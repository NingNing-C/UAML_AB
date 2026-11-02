"""Shared data contracts, cached representations and context-to-token mapping."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from uaml_ab.config import load_config, select_encoding
from uaml_ab.data import (add_gaussian_noise, get_data_loader, get_data_splits, load_case,
                          load_data, load_embeddings, split_by_mutation_number, split_case)
from uaml_ab.embeddings import generate, get_and_save_embeddings, resolve_embedding_inputs, token_positions
from uaml_ab.encodings import seqs_to_onehot, seqs_to_onehot_cnn

HAS_H5PY = importlib.util.find_spec("h5py") is not None


class DataTests(unittest.TestCase):
    def test_configuration_paths_and_encoding_roundtrip(self):
        source = {"global": {"encoder": "onehot", "data_path": "../data/case.csv",
                             "esm2_embedding_path": "../cache/case.h5"},
                  "methods": {"MVE": {"lr": 0.001, "hidden_size": 128}},
                  "encoding_overrides": {"esm2": {"methods": {"MVE": {"lr": 0.002}}}}}
        original = deepcopy(source)
        embedded = select_encoding(source, "esm2")
        self.assertEqual(source, original)
        self.assertEqual(embedded["methods"]["MVE"], {"lr": 0.002, "hidden_size": 128})
        restored = select_encoding(embedded, "onehot")
        self.assertEqual(restored["methods"], source["methods"])
        self.assertEqual(embedded["global"]["encoder"], "esm2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            path = root / "configs" / "case.yaml"
            path.write_text(yaml.safe_dump(source))
            config = load_config(path, encoding="esm2")
            self.assertEqual(Path(config["global"]["data_path"]), (root / "data/case.csv").resolve())
            self.assertEqual(Path(config["global"]["esm2_embedding_path"]), (root / "cache/case.h5").resolve())

    def test_checkpoint_compatible_onehot_layout(self):
        expected = np.zeros((1, 4, 24), dtype=int)
        expected[0, 1, 1] = expected[0, 2, 15] = expected[0, 3, 23] = 1
        np.testing.assert_array_equal(seqs_to_onehot_cnn(["MAX"]), expected)
        np.testing.assert_array_equal(seqs_to_onehot(["MAX"]), expected.reshape(1, -1))

    def test_unlabeled_csv_and_loader_never_make_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.csv"
            pd.DataFrame({"cdr": ["AAAAAAA", "CAAAAAA"]}).to_csv(path, index=False)
            config = {"global": {"data_path": str(path), "seq_col": "cdr", "label_col": "score",
                                  "wt": "AAAAAAA", "encoder": "onehot"}}
            flat, cnn = load_data(config, require_labels=False)
            self.assertNotIn("label", flat)
            self.assertNotIn("label", cnn)
            self.assertEqual(flat.n_mut.tolist(), [0, 1])
            original_indices = flat.index.tolist()
            batch = next(iter(get_data_loader(flat, 2, include_labels=False)))
            self.assertEqual(len(batch), 2)
            self.assertEqual(tuple(batch[1].shape), (2, 8 * 24))
            self.assertEqual(flat.index.tolist(), original_indices)
            with self.assertRaisesRegex(ValueError, "Label column"):
                get_data_loader(flat, 2)
            with self.assertRaisesRegex(ValueError, "Label column"):
                load_data(config, require_labels=True)

    def test_pairing_and_source_indices_survive_partitions(self):
        frame = pd.DataFrame({"sequence": ["AAAAAAA"] * 100, "label": np.arange(100.)},
                             index=np.arange(100) + 1000)
        cnn = frame.copy()
        cnn["sequence"] = [np.full((7, 3), number) for number in range(100)]
        partitions, paired = split_case(frame, cnn, seed=9, n_train=50)
        self.assertEqual([len(part) for part in partitions], [45, 5, 20])
        for flat, positional in zip(partitions, paired):
            self.assertEqual(flat.index.tolist(), positional.index.tolist())
            np.testing.assert_array_equal(flat.label, positional.label)
        identities = [set(part.index) for part in partitions]
        self.assertFalse(identities[0] & identities[1] or identities[0] & identities[2] or identities[1] & identities[2])
        repeated, _ = split_case(frame, cnn, seed=9, n_train=50)
        self.assertEqual([part.index.tolist() for part in partitions], [part.index.tolist() for part in repeated])
        with self.assertRaisesRegex(ValueError, "identical unique"):
            split_case(frame, cnn.iloc[::-1])
        full = get_data_splits(frame, SimpleNamespace(split_method="random"), 42)
        self.assertEqual([len(part) for part in full], [72, 8, 20])

    def test_overlapping_mutation_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            split_by_mutation_number(pd.DataFrame({"n_mut": [0, 1, 2, 3]}), "train_0_1_2_test_2_3")

    def test_noise_reuses_source_pattern_and_sample_standard_deviation(self):
        frame = pd.DataFrame({"label": np.linspace(-2, 3, 20)})
        half = add_gaussian_noise(frame, noise_level=0.5)
        full = add_gaussian_noise(frame, noise_level=1.)
        np.testing.assert_allclose(full.label - frame.label, 2 * (half.label - frame.label), atol=1e-15)
        expected = np.random.default_rng(42).normal(0, 0.5 * frame.label.std(), size=len(frame))
        np.testing.assert_allclose(half.label - frame.label, expected, atol=1e-15)
        np.testing.assert_array_equal(frame.label, np.linspace(-2, 3, 20))

    def test_full_domain_heavy_light_coordinates(self):
        frame = pd.DataFrame({"sequence_vh": ["MACDEFGH", "MTCDEFGH"],
                              "sequence_vl": ["STVWYA", "ATVWYA"], "mut": ["VH_A2T,VL_S1A", np.nan]})
        prepared, positions = resolve_embedding_inputs(frame, {}, ["sequence_vh", "sequence_vl"])
        self.assertEqual(positions, [2, 9])
        self.assertEqual(prepared._esm2_context.tolist(), ["MACDEFGHSTVWYA", "MTCDEFGHATVWYA"])
        self.assertEqual(token_positions(positions), [2, 9])
        self.assertEqual(token_positions([1, 8], one_indexed=False), [2, 9])
        _, explicit = resolve_embedding_inputs(frame, {}, ["sequence_vh", "sequence_vl"], positions=[9, 2])
        self.assertEqual(explicit, [9, 2])
        _, located = resolve_embedding_inputs(pd.DataFrame({"sequence_vh": ["MACDEF"]}),
                                              {"wt": "ACD", "wt_vh": "MACDEF"})
        self.assertEqual(located, [2, 3, 4])

    def test_optional_dependencies_stay_lazy(self):
        code = """
import sys
import uaml_ab.data, uaml_ab.config, uaml_ab.embeddings
for name in ('h5py', 'transformers', 'gpytorch', 'torchbnn', 'sklearn', 'matplotlib'):
    assert name not in sys.modules, name
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_new_embedding_input_does_not_replace_configured_training_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_cache = root / "training.h5"
            training_cache.write_bytes(b"unchanged training cache")
            inputs = root / "new.csv"
            pd.DataFrame({"sequence": ["CAAAAAA", "ACAAAAA"],
                          "sequence_vh": ["MCAAAAAA", "MACAAAAA"]}).to_csv(inputs, index=False)
            config = root / "case.yaml"
            config.write_text(yaml.safe_dump({"global": {
                "data_path": "training.csv", "esm2_embedding_path": "training.h5",
                "seq_col": "sequence", "label_col": "label", "wt": "AAAAAAA",
                "full_sequence_cols": ["sequence_vh"], "embedding_positions": list(range(2, 9)),
            }}))

            def writer(*args, **kwargs):
                return Path(args[5])

            with patch("uaml_ab.embeddings.get_and_save_embeddings", side_effect=writer) as mocked:
                destination = generate(config, input_path=inputs, require_labels=False)
                self.assertEqual(destination, inputs.with_suffix(".embeddings.h5"))
                self.assertEqual(Path(mocked.call_args.args[5]), destination)
                explicit = root / "custom.h5"
                self.assertEqual(generate(config, input_path=inputs, output=explicit, require_labels=False), explicit)
            self.assertEqual(training_cache.read_bytes(), b"unchanged training cache")


@unittest.skipUnless(HAS_H5PY, "Cached representations require optional h5py")
class EmbeddingDataTests(unittest.TestCase):
    def cache(self, path, include_labels=True, bad_width=False):
        import h5py
        values = np.arange(6 * 7 * 3, dtype=np.float16).reshape(6, 21)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("embeddings", data=values[:, :-1] if bad_width else values)
            handle.create_dataset("mut_loci_seq", data=np.array(["AAAAAAA", "CAAAAAA", "ACAAAAA",
                                                                 "AACAAAA", "AAACAAA", "AAAACAA"], dtype=object),
                                  dtype=h5py.string_dtype())
            if include_labels:
                handle.create_dataset("fitness", data=np.arange(6, dtype=np.float32))
            handle.attrs["positions"] = [7, 1, 6, 2, 5, 3, 4]
        return values.reshape(6, 7, 3)

    def test_cached_position_order_pooling_and_unlabeled_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.h5"
            expected = self.cache(path)
            flat, cnn = load_embeddings(path, "AAAAAAA")
            np.testing.assert_array_equal(np.stack(cnn.sequence), expected)
            np.testing.assert_array_equal(np.stack(flat.sequence), expected.mean(axis=1))
            self.assertEqual(flat.attrs["positions"], [7, 1, 6, 2, 5, 3, 4])
            self.assertEqual(flat.n_mut.tolist(), [0, 1, 1, 1, 1, 1])
            _, features, labels = next(iter(get_data_loader(cnn.iloc[::-1], 2, "esm2_cnn")))
            self.assertEqual(tuple(features.shape), (2, 7, 3))
            np.testing.assert_array_equal(labels.numpy(), [5., 4.])
            unlabeled = Path(directory) / "unlabeled.h5"
            self.cache(unlabeled, include_labels=False)
            config_path = Path(directory) / "case.yaml"
            config_path.write_text(yaml.safe_dump({"global": {"encoder": "esm2", "wt": "AAAAAAA",
                                                               "esm2_embedding_path": "unlabeled.h5"}}))
            _, flat, cnn = load_case(config_path, require_labels=False)
            self.assertNotIn("label", flat)
            self.assertEqual(len(next(iter(get_data_loader(cnn, 2, "esm2_cnn", include_labels=False)))), 2)
            with self.assertRaisesRegex(ValueError, "fitness"):
                load_embeddings(unlabeled)

    def test_invalid_cache_width_and_feature_rank_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.h5"
            self.cache(path, bad_width=True)
            with self.assertRaisesRegex(ValueError, "divisible"):
                load_embeddings(path)
        frame = pd.DataFrame({"sequence": [np.zeros(3), np.zeros(4)]})
        with self.assertRaisesRegex(ValueError, "identical finite shapes"):
            get_data_loader(frame, 2, "esm2", include_labels=False)

    def test_generator_preserves_original_rows_without_synthetic_labels(self):
        class Tokenizer:
            def __call__(self, sequences, **kwargs):
                ids = torch.zeros(len(sequences), max(map(len, sequences)) + 2)
                for row, sequence in enumerate(sequences):
                    ids[row, 1:len(sequence) + 1] = torch.tensor([ord(letter) for letter in sequence])
                return {"input_ids": ids}
        class Model(torch.nn.Module):
            def forward(self, input_ids):
                return SimpleNamespace(last_hidden_state=torch.stack((input_ids, input_ids / 2), dim=-1))
        module = ModuleType("transformers")
        module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name: Tokenizer())
        module.AutoModel = SimpleNamespace(from_pretrained=lambda name: Model())
        frame = pd.DataFrame({"context": ["MRAAAAA", "MACDEFGHI", "MKLLLL"], "region": ["RA", "AC", "KL"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.h5"
            with patch.dict(sys.modules, {"transformers": module}):
                get_and_save_embeddings(frame, "context", None, "region", [2, 3], path, batch_size=2, device="cpu")
            flat, cnn = load_embeddings(path, require_labels=False)
            self.assertNotIn("label", flat)
            self.assertEqual(flat.mut_loci_seq.tolist(), frame.region.tolist())
            for sequence, matrix in zip(frame.context, cnn.sequence):
                np.testing.assert_array_equal(matrix, [[ord(sequence[1]), ord(sequence[1]) / 2],
                                                       [ord(sequence[2]), ord(sequence[2]) / 2]])


if __name__ == "__main__":
    unittest.main()
