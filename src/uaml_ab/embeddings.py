"""Optional frozen ESM2 representations in the original cached HDF5 format.

Embed full heavy/light-domain context first, then select target residue tokens.
Cached files preserve their own position order; newly inferred mutation
positions are sorted in sequence order for deterministic regeneration.
"""
import argparse
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

from .config import load_config

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"


def get_positions(mutations):
    """Parse source annotations such as M28A, VH_M28A and VL_S12T."""
    if not isinstance(mutations, str):
        return []
    positions = []
    for mutation in mutations.split(","):
        match = re.search(r"(.+?)(\d+)([A-Za-z])", mutation.strip())
        if match:
            prefix, number, _ = match.groups()
            positions.append((prefix[:-1], int(number)))
    return positions


def _columns(value):
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value) if value is not None else None


def token_positions(positions, one_indexed=True):
    """Map residue coordinates to ESM tokens, accounting for the leading CLS."""
    values = [int(position) for position in positions]
    if not values or min(values) < (1 if one_indexed else 0):
        raise ValueError("Residue positions must be nonempty and nonnegative (positive when 1-indexed).")
    if len(set(values)) != len(values):
        raise ValueError("Residue positions must not contain duplicates.")
    return values if one_indexed else [position + 1 for position in values]


def resolve_embedding_inputs(frame, settings, full_sequence_cols=None,
                             mutation_col=None, positions=None):
    """Prepare context and one-based target coordinates without a dataset registry.

    `full_sequence_cols` gives one full domain/scFv column, or heavy then light
    columns to concatenate. VL mutation coordinates are offset by heavy-chain
    length. `embedding_positions` can explicitly specify concatenated residue
    coordinates, which is preferable when reproducing a cached position order.
    """
    columns = _columns(full_sequence_cols or settings.get("full_sequence_cols"))
    if columns is None:
        if "sequence_vh" in frame:
            columns = ["sequence_vh"]
            if settings.get("wt_vl") and "sequence_vl" in frame:
                columns.append("sequence_vl")
        elif "scFv" in frame:
            columns = ["scFv"]
        else:
            raise ValueError("Set global.full_sequence_cols or --full_sequence_cols to full-domain columns.")
    if not 1 <= len(columns) <= 2 or any(column not in frame for column in columns):
        raise ValueError("Provide one full context column, or two columns ordered heavy then light.")
    result = frame.copy()
    for column in columns:
        if result[column].isna().any() or not result[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Full-domain column {column!r} must contain nonempty sequence strings.")
        result[column] = result[column].str.strip()
        if result[column].eq("").any() or result[column].str.contains(r"\s", regex=True).any():
            raise ValueError("Protein contexts must contain amino-acid strings without internal whitespace.")
    result["_esm2_context"] = result[columns[0]]
    if len(columns) == 2:
        heavy_lengths = result[columns[0]].str.len().unique()
        if len(heavy_lengths) != 1:
            raise ValueError("Concatenated heavy/light contexts require a fixed heavy-chain length.")
        result["_esm2_context"] = result[columns[0]] + result[columns[1]]
    if not result["_esm2_context"].is_unique:
        raise ValueError("Full context sequences must be unique, as in the original embedding pipeline.")

    selected = positions if positions is not None else settings.get("embedding_positions")
    if selected is None:
        column = mutation_col or settings.get("mutation_col")
        if column is None:
            column = "mut_vh_nr" if "mut_vh_nr" in result else "mut"
        inferred = set()
        if column in result:
            for annotations in result[column]:
                for chain, position in get_positions(annotations):
                    if chain.upper().rstrip("_") == "VL":
                        if len(columns) != 2:
                            raise ValueError("VL mutation annotations require explicit heavy/light context columns.")
                        position += len(result[columns[0]].iloc[0])
                    elif chain.upper().rstrip("_") not in ("", "VH"):
                        raise ValueError(f"Unrecognized mutation-chain prefix {chain!r}; supply --positions explicitly.")
                    inferred.add(position)
        selected = sorted(inferred)
        if not selected:
            # For annotation-free contiguous regions, locate the WT region in
            # the full WT sequence. Never embed the isolated CDR as context.
            wild_type = settings.get("wt")
            full_wt = str(settings.get("wt_vh", "")) + str(settings.get("wt_vl", ""))
            if wild_type and full_wt.count(wild_type) == 1:
                start = full_wt.index(wild_type) + 1
                selected = list(range(start, start + len(wild_type)))
            else:
                raise ValueError("Cannot infer target residues; supply global.embedding_positions or --positions.")
    selected = [int(position) for position in selected]
    token_positions(selected)
    lengths = result["_esm2_context"].str.len()
    if max(selected) > lengths.min():
        raise ValueError("A selected residue exceeds at least one full context's length.")
    return result, selected


def get_safe_batch_size(seq_length, vram_gb=20.0):
    if seq_length < 1 or vram_gb <= 0:
        raise ValueError("Sequence length and memory budget must be positive.")
    return max(1, int(vram_gb / (seq_length * 0.0002)))


def verify_alignment(emb_dataset, sequences, positions, tokenizer, model, device,
                     n_check=8, seed=0):
    """Re-embed sampled identities to detect a length-sort/write-order mix-up."""
    import torch
    indices = np.sort(np.random.default_rng(seed).choice(
        len(sequences), min(n_check, len(sequences)), replace=False))
    inputs = tokenizer([sequences[index] for index in indices],
                       return_tensors="pt", padding=True, truncation=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        fresh = model(**inputs).last_hidden_state[:, positions, :].flatten(start_dim=1)
    fresh = fresh.cpu().numpy().astype(np.float32)
    stored = emb_dataset[indices].astype(np.float32)
    distances = np.linalg.norm(fresh[:, None, :] - stored[None, :, :], axis=2)
    own = np.diag(distances)
    # Different contexts can yield identical selected features. Ties are valid.
    tolerance = np.maximum(np.linalg.norm(fresh, axis=1) * 0.01, 1e-4)
    if np.any(own > distances.min(axis=1) + tolerance) or np.any(own > tolerance):
        raise RuntimeError("Embedding alignment check failed: stored rows do not match their sequence identities.")


def get_and_save_embeddings(df, sequence_col, fitness_col, mutated_seq_col, positions,
                            output_path, one_indexed=True, model_name=DEFAULT_MODEL,
                            vram_gb=20.0, batch_size=None, device=None, verify=True):
    """Generate the original float16 N x (positions*hidden) cached layout.

    This call may download model weights. Merely importing the module or reading
    cached embeddings does not load transformers or download a model.
    """
    try:
        import h5py
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as error:
        raise ImportError("Embedding generation requires the esm2 optional dependencies.") from error
    if len(df) == 0:
        raise ValueError("Cannot embed an empty dataset.")
    sequences = df[sequence_col].tolist()
    if not all(isinstance(sequence, str) and sequence for sequence in sequences):
        raise ValueError("Full context sequences must be nonempty strings.")
    if len(set(sequences)) != len(sequences):
        raise ValueError("Full context sequences must be unique.")
    labels = None
    if fitness_col is not None:
        labels = pd.to_numeric(df[fitness_col], errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(labels).all():
            raise ValueError("Embedding labels must be finite numeric values.")
    residue_tokens = token_positions(positions, one_indexed)
    if max(residue_tokens) > min(map(len, sequences)):
        raise ValueError("Selected residues exceed at least one context sequence.")
    if max(map(len, sequences)) > 1022:
        raise ValueError("Full contexts exceed ESM2's 1024-token limit; explicit context curation is required.")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive.")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    if target_device.type == "cuda":
        model = model.half()
    model = model.to(target_device)
    order = np.argsort([len(sequence) for sequence in sequences], kind="stable")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=destination.name + ".", suffix=".tmp",
                                     dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with h5py.File(temporary_path, "w") as handle:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset("mut_loci_seq", data=np.asarray(df[mutated_seq_col], dtype=object),
                                  dtype=string_dtype)
            if labels is not None:
                handle.create_dataset("fitness", data=labels)
            handle.attrs["model_name"] = model_name
            handle.attrs["positions"] = list(positions)
            handle.attrs["positions_indexed"] = "1-indexed" if one_indexed else "0-indexed"
            embeddings = None
            offset = 0
            current_size = batch_size or get_safe_batch_size(max(map(len, sequences)), vram_gb)
            while offset < len(sequences):
                rows = order[offset:offset + current_size]
                batch = [sequences[index] for index in rows]
                try:
                    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
                    inputs = {key: value.to(target_device) for key, value in inputs.items()}
                    with torch.no_grad():
                        hidden = model(**inputs).last_hidden_state
                    values = hidden[:, residue_tokens, :].flatten(start_dim=1).cpu().numpy().astype(np.float16)
                    if embeddings is None:
                        embeddings = handle.create_dataset("embeddings", shape=(len(sequences), values.shape[1]),
                                                           dtype=np.float16)
                    sorted_rows = np.argsort(rows)
                    embeddings[rows[sorted_rows]] = values[sorted_rows]
                    offset += len(rows)
                    print(f"Embedded {offset}/{len(sequences)} sequences", flush=True)
                except torch.cuda.OutOfMemoryError:
                    if current_size == 1:
                        raise
                    current_size = max(1, current_size // 2)
                    torch.cuda.empty_cache()
            if verify:
                verify_alignment(embeddings, sequences, residue_tokens, tokenizer, model, target_device)
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination



def generate(config_path, output=None, input_path=None, require_labels=True, **options):
    """Generate a cache from a case's full-domain contexts without loading models elsewhere.

    Input paths supplied explicitly resolve in the caller's current directory;
    configured paths have already resolved beside their YAML file. Set
    require_labels=False for a prediction-only cache without synthetic labels.
    """
    config = load_config(config_path)
    settings = config["global"]
    frame = pd.read_csv(input_path or settings["data_path"])
    frame, selected = resolve_embedding_inputs(
        frame, settings, options.pop("full_sequence_cols", None),
        options.pop("mutation_col", None), options.pop("positions", None))
    label_col = settings.get("label_col", "label")
    if label_col not in frame:
        if require_labels:
            raise ValueError(f"Label column {label_col!r} is missing; use unlabeled mode for prediction caches.")
        label_col = None
    sequence_col = settings.get("seq_col", "sequence")
    if sequence_col not in frame:
        raise ValueError(f"Target-region sequence column {sequence_col!r} is missing.")
    if output is not None:
        destination = output
    elif input_path is not None:
        destination = Path(input_path).with_suffix(".embeddings.h5")
    else:
        destination = settings.get("esm2_embedding_path")
    if destination is None:
        destination = Path(input_path or settings["data_path"]).with_name("embeddings.h5")
    options.setdefault("model_name", settings.get("esm2_model_name", DEFAULT_MODEL))
    return get_and_save_embeddings(frame, "_esm2_context", label_col, sequence_col,
                                    selected, destination, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path, dest="input_path")
    parser.add_argument("--unlabeled", action="store_true", help="Allow a prediction-only cache without labels")
    parser.add_argument("--full-sequence-cols", "--full_sequence_cols", dest="full_sequence_cols", nargs="+", help="One full context column or heavy then light columns")
    parser.add_argument("--mutation-col", "--mutation_col", dest="mutation_col")
    parser.add_argument("--positions", nargs="+", type=int, help="Explicit one-based positions in concatenated context")
    parser.add_argument("--model-name", "--model_name", dest="model_name")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    parser.add_argument("--vram-gb", "--vram_gb", dest="vram_gb", type=float, default=20.0)
    parser.add_argument("--device")
    args = vars(parser.parse_args(argv))
    args["require_labels"] = not args.pop("unlabeled")
    if args["model_name"] is None:
        args.pop("model_name")
    path = generate(args.pop("config"), **args)
    print(f"Saved embeddings to {path}")
    return 0
