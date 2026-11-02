"""Shared sequence/cache loading, paired model inputs, partitions and label noise."""
import logging
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import load_config, select_encoding
from .encodings import AA_TO_INT, seqs_to_onehot, seqs_to_onehot_cnn

ENCODERS = {"onehot": seqs_to_onehot, "onehot_cnn": seqs_to_onehot_cnn}


def validate_sequences(sequences, wt=None, min_length=6):
    values = list(sequences)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("Sequences must be nonempty strings.")
    lengths = {len(value) for value in values}
    if len(lengths) != 1 or (wt is not None and lengths != {len(wt)}):
        raise ValueError("Aligned sequences must share one length and match WT when specified.")
    if next(iter(lengths)) < min_length:
        raise ValueError(f"At least {min_length} residues are required.")
    invalid = set("".join(values)) - set(AA_TO_INT)
    if invalid:
        raise ValueError(f"Unsupported amino-acid symbols: {sorted(invalid)}")


def calc_num_mut(first, second):
    if len(first) != len(second):
        raise ValueError("Mutation counts require equal-length sequences.")
    return sum(a != b for a, b in zip(first, second))


def _labels(frame, column="label"):
    if column not in frame:
        raise ValueError(f"Label column {column!r} not found.")
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError("Labels must be finite numeric values.")
    return values


def load_data_from_csv(file_path, seq_col="sequence", label_col="label", wt=None, require_labels=True):
    frame = pd.read_csv(file_path)
    if seq_col not in frame:
        raise ValueError(f"Sequence column {seq_col!r} not found.")
    sequences = frame[seq_col].copy()
    validate_sequences(sequences, wt)
    frame["sequence"] = sequences
    frame["mut_loci_seq"] = sequences
    if require_labels:
        frame["label"] = _labels(frame, label_col)
    elif label_col in frame:
        frame["label"] = frame[label_col]
    if wt is not None:
        frame["n_mut"] = [calc_num_mut(sequence, wt) for sequence in sequences]
    return frame


def load_embeddings(path, wt=None, require_labels=True):
    """Read original positional HDF5 caches, preserving recorded position order."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Cached ESM2 inputs require the esm2 optional dependencies.") from error
    with h5py.File(path, "r") as handle:
        required = {"embeddings", "mut_loci_seq"} | ({"fitness"} if require_labels else set())
        missing = required - set(handle)
        if missing:
            raise ValueError(f"Embedding file is missing datasets: {sorted(missing)}")
        positions = np.asarray(handle.attrs.get("positions", [])).reshape(-1)
        if not len(positions):
            raise ValueError("Embedding file must record nonempty positions metadata.")
        values = handle["embeddings"][:]
        sequences = handle["mut_loci_seq"].asstr()[:]
        labels = handle["fitness"][:] if "fitness" in handle else None
        attributes = {"positions": positions.tolist(),
                      "model_name": str(handle.attrs.get("model_name", "unknown")),
                      "positions_indexed": str(handle.attrs.get("positions_indexed", "unspecified"))}
    if values.ndim != 2 or not values.shape[1] or values.shape[1] % len(positions):
        raise ValueError("Embedding width must be positive and divisible by the position count.")
    if len(values) != len(sequences) or not len(sequences) or not np.isfinite(values).all():
        raise ValueError("Nonempty, finite embedding rows must align with sequence identities.")
    validate_sequences(sequences, wt, min_length=1)
    if labels is not None and labels.shape != (len(sequences),):
        raise ValueError("Embedding scalar labels must align with sequence identities.")
    if require_labels and not np.isfinite(labels).all():
        raise ValueError("Embedding labels must be finite.")
    positional = values.reshape(len(sequences), len(positions), -1)
    base = {"mut_loci_seq": np.asarray(sequences, dtype=str)}
    if labels is not None:
        base["label"] = labels
    flat = pd.DataFrame({**base, "sequence": list(positional.mean(axis=1))})
    cnn = pd.DataFrame({**base, "sequence": list(positional)})
    if wt is not None:
        mutations = [calc_num_mut(sequence, wt) for sequence in sequences]
        flat["n_mut"], cnn["n_mut"] = mutations, mutations
    flat.attrs.update(attributes)
    cnn.attrs.update(attributes)
    return flat, cnn


def load_data(config, encoding=None, require_labels=True, path=None):
    """Load either representation from one selected case; path overrides its input."""
    active = config["global"].get("encoder", "onehot")
    selected_encoding = encoding or active
    if selected_encoding not in ("onehot", "esm2"):
        raise ValueError("Encoding must be 'onehot' or 'esm2'.")
    # A selected configuration can carry explicit input/column overrides. Do
    # not reselect its canonical baseline and discard those caller changes.
    settings = config["global"] if selected_encoding == active else select_encoding(config, encoding)["global"]
    if selected_encoding == "esm2":
        selected = path or settings.get("esm2_embedding_path") or settings.get("embedding_path")
        if selected is None:
            raise ValueError("ESM2 requires global.esm2_embedding_path or an explicit cache path.")
        return load_embeddings(selected, settings.get("wt"), require_labels=require_labels)
    selected = path or settings.get("data_path")
    if selected is None:
        raise ValueError("Sequence input requires global.data_path or an explicit CSV path.")
    frame = load_data_from_csv(selected, settings.get("seq_col", "sequence"),
                               settings.get("label_col", "label"), settings.get("wt"),
                               require_labels=require_labels)
    return frame, frame.copy()


def load_case(config_path, encoding=None, require_labels=True):
    config = load_config(config_path, encoding=encoding)
    flat, cnn = load_data(config, require_labels=require_labels)
    return config, flat, cnn


class ProteinSeqDataset(Dataset):
    """Paired-loader row positions are local; caller dataframe identities stay intact."""
    def __init__(self, frame, encoder="onehot", include_labels=True):
        if encoder not in (*ENCODERS, "esm2", "esm2_cnn"):
            raise ValueError(f"Unsupported encoder: {encoder}")
        self.values = frame["sequence"].tolist()
        self.encoder = encoder
        self.labels = _labels(frame).to_numpy(dtype=np.float32, copy=True) if include_labels else None
        if encoder.startswith("esm2") and self.values:
            expected = 2 if encoder.endswith("_cnn") else 1
            shape = np.asarray(self.values[0]).shape
            if len(shape) != expected or not all(shape):
                raise ValueError("Cached inputs have the wrong representation rank or empty features.")
            if any(np.asarray(value).shape != shape or not np.isfinite(value).all() for value in self.values):
                raise ValueError("Cached input features must have identical finite shapes.")

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = self.values[index]
        if self.encoder in ENCODERS:
            value = ENCODERS[self.encoder]([value])[0]
        feature = torch.as_tensor(np.asarray(value), dtype=torch.float32)
        if self.labels is None:
            return index, feature
        return index, feature, self.labels[index]


def get_data_loader(frame, batch_size, encoder="onehot", shuffle=False, include_labels=True):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    return DataLoader(ProteinSeqDataset(frame, encoder, include_labels), batch_size=batch_size, shuffle=shuffle)


def split_data_random(frame, n_train=-1, test_size=0.2, val_frac=0.1, random_state=42):
    """Original pandas split identities: 20% test, then 10% development validation."""
    if not frame.index.is_unique:
        raise ValueError("Partitions require unique source row indices.")
    if not 0 < test_size < 1 or not 0 <= val_frac < 1:
        raise ValueError("test_size must be in (0,1), val_frac in [0,1).")
    test = frame.sample(frac=test_size, random_state=random_state)
    development = frame.drop(test.index)
    if n_train not in (None, -1):
        if n_train < 1:
            raise ValueError("n_train must be positive, None, or -1.")
        if n_train < len(development):
            development = development.sample(n=n_train, random_state=random_state)
    validation = development.sample(frac=val_frac, random_state=random_state)
    return development.drop(validation.index).copy(), validation.copy(), test.copy()


def split_train_val(frame, val_frac=0.1, random_state=42):
    validation = frame.sample(frac=val_frac, random_state=random_state)
    return frame.drop(validation.index).copy(), validation.copy()


def parse_mutation_range(part, max_mutation):
    if part.endswith("+"):
        return list(range(int(part[:-1]), max_mutation + 1))
    return [int(value) for value in part.split("_")]


def split_by_mutation_number(frame, setting, test_on_rest=False):
    if "n_mut" not in frame:
        raise ValueError("Mutation-number partitions require n_mut.")
    maximum = int(frame.n_mut.max())
    if test_on_rest:
        if not setting.startswith("train_"):
            raise ValueError("Mutation split must begin with train_.")
        train_counts = parse_mutation_range(setting[6:], maximum)
        test_counts = sorted(set(frame.n_mut) - set(train_counts))
    else:
        match = re.fullmatch(r"train_([0-9+_]+)_test_([0-9+_]+)", setting)
        if match is None:
            raise ValueError("Use a mutation split such as train_0_1_2_test_3+.")
        train_counts, test_counts = [parse_mutation_range(part, maximum) for part in match.groups()]
    if set(train_counts) & set(test_counts):
        raise ValueError("Training and test mutation-number ranges must be disjoint.")
    return frame[frame.n_mut.isin(train_counts)].copy(), frame[frame.n_mut.isin(test_counts)].copy()


def get_data_splits(frame, args, seed):
    cap = getattr(args, "n_train", None)
    fraction = getattr(args, "n_train_frac", None)
    if fraction is not None:
        cap = int(len(frame) * fraction)
    method = getattr(args, "split_method", "random")
    if method == "random":
        return split_data_random(frame, n_train=cap, random_state=seed)
    if method != "mutation_number":
        raise ValueError(f"Unsupported split method: {method}")
    setting = getattr(args, "mutation_split_setting", None)
    if not setting:
        raise ValueError("A mutation_split_setting is required.")
    train, test = split_by_mutation_number(frame, setting, getattr(args, "test_on_rest", False))
    if cap not in (None, -1) and cap < len(train):
        train = train.sample(n=cap, random_state=seed)
    train, validation = split_train_val(train, random_state=seed)
    return train, validation, test


def split_case(flat, cnn, *, seed=42, n_train=None, test_size=0.2, val_frac=0.1):
    if not flat.index.equals(cnn.index) or not flat.index.is_unique:
        raise ValueError("Paired representations require identical unique source row indices.")
    for column in ("mut_loci_seq", "label"):
        if column in flat and column in cnn and not np.array_equal(flat[column], cnn[column]):
            raise ValueError(f"Paired {column} values must have identical row order.")
    partitions = split_data_random(flat, n_train, test_size, val_frac, seed)
    return partitions, tuple(cnn.loc[part.index].copy() for part in partitions)


def add_gaussian_noise(frame, column="label", noise_level=0.0, seed=42):
    if not np.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and nonnegative.")
    result = frame.copy()
    if noise_level == 0:
        return result
    labels = _labels(frame, column)
    scale = labels.std()
    if not np.isfinite(scale):
        raise ValueError("Noise injection requires at least two finite labels.")
    result[column] = labels + np.random.default_rng(seed).normal(0, noise_level * scale, size=len(frame))
    return result


def resolve_wt_fitness(frame, label_col="label"):
    values = frame.loc[frame.n_mut == 0, label_col]
    if not len(values):
        raise ValueError("WT-referenced optimization requires a measured WT sequence.")
    if len(values) > 1:
        logging.warning("Multiple WT rows found; using their mean label.")
    return float(values.mean())
