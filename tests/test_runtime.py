"""Regression coverage for one model lifecycle across all reported methods.

The retained trainer evaluation bodies supply reference scientific aggregation.
Public prediction must match them while never evaluating labels or losses.
Optional methods are skipped only when their declared model dependency is absent.
"""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import torch

from uaml_ab.models.registry import METHODS, method_spec
from uaml_ab.runtime import fit, load, _loader


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        rng = np.random.default_rng(8)
        sequences = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 24)) for _ in range(26)]
        base = {"mut_loci_seq": sequences, "label": rng.normal(0, .1, 26)}
        index = np.arange(26)*2+100
        cls.onehot = pd.DataFrame({**base, "sequence": sequences}, index=index)
        values = rng.normal(0, .1, (26, 10, 8)).astype(np.float32)
        cls.flat = pd.DataFrame({**base, "sequence": list(values.mean(1))}, index=index)
        cls.positional = pd.DataFrame({**base, "sequence": list(values)}, index=index)
        cls.settings = {"hidden_size": 8, "num_filters": 4, "num_models": 3,
                        "num_samples": 3, "batch_size": 4, "epochs": 100}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def frames(self, encoding):
        return (self.onehot, self.onehot.copy()) if encoding == "onehot" else (self.flat, self.positional)

    def model(self, method, encoding="onehot", initialization_seed=None):
        flat, cnn = self.frames(encoding)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit(method, flat.iloc[:18], flat.iloc[18:22], self.settings,
                       "cpu", 7, 1, encoding=encoding, train_cnn=cnn.iloc[:18],
                       val_cnn=cnn.iloc[18:22], initialization_seed=initialization_seed)

    def exercise_method(self, method, encoding):
        fitted = self.model(method, encoding)
        flat, cnn = self.frames(encoding)
        test, positional = flat.iloc[22:], cnn.iloc[22:]
        original_index = test.index.copy()
        self.assertEqual(fitted.settings["epochs"], 1)
        loader = _loader(positional if method == "CNN" else test, 4, encoding, cnn=method == "CNN")
        torch.manual_seed(123)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "Linreg":
                _, features, _ = next(iter(loader))
                reference = {"pred": fitted.model.predict(features.numpy())}
            else:
                output = (fitted.trainer.evaluate(loader, _loader(positional, 4, encoding, cnn=True))
                          if method == "ModelArcEnsemble" else fitted.trainer.evaluate(loader))
                reference = {"pred": output[1]}
                if method != "Deterministic":
                    reference.update(epistd=output[2], alestd=output[3], totalstd=output[4])
        unlabeled, unlabeled_cnn = test.drop(columns="label"), positional.drop(columns="label")
        guard = patch.object(fitted.trainer, "evaluate", side_effect=AssertionError("Prediction evaluated a loss")) if fitted.trainer else nullcontext()
        with guard:
            before = torch.get_rng_state()
            prediction = fitted.predict(unlabeled, unlabeled_cnn, seed=123)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            poisoned, poisoned_cnn = test.copy(), positional.copy()
            poisoned["label"], poisoned_cnn["label"] = "invalid unused label", np.nan
            again = fitted.predict(poisoned, poisoned_cnn, seed=123)
        for name, expected in reference.items():
            if expected is not None:
                np.testing.assert_allclose(prediction[name], expected, rtol=1e-5, atol=1e-7)
        for name in prediction:
            self.assertEqual(prediction[name].shape, (len(test),))
            self.assertTrue(np.isfinite(prediction[name]).all())
            np.testing.assert_array_equal(prediction[name], again[name])
        if method in ("Linreg", "Deterministic"):
            self.assertEqual(set(prediction), {"pred"})
        elif "epistd" in prediction:
            np.testing.assert_allclose(prediction["totalstd"]**2,
                                       prediction["epistd"]**2+prediction["alestd"]**2, rtol=1e-5)
        fitted.metadata = {"config": {"global": {"encoder": encoding}}, "source_fingerprint": "fixture",
                           "splits": {"test": np.array(test.index)}, "seed": np.int64(7)}
        path = self.directory / f"{method}-{encoding}.pt"
        fitted.save(path)
        payload = torch.load(path, weights_only=True, map_location="cpu")
        self.assertEqual(payload["format_version"], 2)
        self.assertEqual(payload["metadata"]["splits"]["test"], list(test.index))
        restored = load("not-read-when-payload-is-provided", payload=payload)
        self.assertEqual(restored.metadata["source_fingerprint"], "fixture")
        actual = restored.predict(unlabeled, unlabeled_cnn, seed=123)
        for name in prediction:
            np.testing.assert_allclose(actual[name], prediction[name], rtol=1e-5, atol=1e-7)
        legacy = {"format_version": 1, "method": method, "settings": fitted.settings, "spec": fitted.spec,
                  "encoding": encoding, "seed": fitted.seed}
        if method == "Linreg":
            legacy.update(linear_coef=torch.as_tensor(fitted.model.coef_),
                          linear_intercept=torch.as_tensor(fitted.model.intercept_))
        else:
            legacy["state_dict"] = fitted.model.state_dict()
        if method == "GP":
            legacy.update(gp_train_x=fitted.gp_train_x, gp_train_y=fitted.gp_train_y)
        old = load("unused", payload=legacy)
        self.assertTrue(old.metadata["legacy"])
        for name, values in old.predict(unlabeled, unlabeled_cnn, seed=123).items():
            np.testing.assert_allclose(values, prediction[name], rtol=1e-5, atol=1e-7)
        self.assertTrue(test.index.equals(original_index))

    def test_registry_contains_exactly_reported_methods(self):
        self.assertEqual(METHODS, ("Linreg", "Deterministic", "MVE", "CNN", "BootstrapEnsemble",
                                  "ModelArcEnsemble", "BNN", "NeuralLinear", "ConcreteDropout", "GP", "Evidential"))

    def test_raw_legacy_dimensions_need_no_reference_labels(self):
        for encoding in ("onehot", "esm2"):
            with self.subTest(encoding=encoding):
                model = self.model("ModelArcEnsemble", encoding)
                self.assertEqual(model.initialization_seed, 42)
                restored = load("unused", config={"global": {"encoder": encoding}}, payload=model.model.state_dict())
                flat, cnn = self.frames(encoding)
                expected = model.predict(flat.iloc[22:].drop(columns="label"), cnn.iloc[22:].drop(columns="label"))
                actual = restored.predict(flat.iloc[22:].drop(columns="label"), cnn.iloc[22:].drop(columns="label"))
                np.testing.assert_allclose(actual["pred"], expected["pred"], rtol=1e-5, atol=1e-7)
                self.assertTrue(restored.metadata["legacy"])
        explicit = self.model("ModelArcEnsemble", initialization_seed=7)
        self.assertEqual(explicit.initialization_seed, 7)

    def test_paired_input_order_and_zero_batch_are_rejected(self):
        model = self.model("ModelArcEnsemble", "esm2")
        with self.assertRaisesRegex(ValueError, "row indices"):
            model.predict(self.flat.iloc[22:], self.positional.iloc[22:].iloc[::-1])
        with self.assertRaisesRegex(ValueError, "positive"):
            model.predict(self.flat.iloc[22:], self.positional.iloc[22:], batch_size=0)


def _method_test(method, encoding):
    def test(self):
        self.exercise_method(method, encoding)
    missing = [name for name in method_spec(method).optional_dependencies if importlib.util.find_spec(name) is None]
    if missing:
        test = unittest.skip("Optional dependency absent: " + ", ".join(missing))(test)
    return test


for _method in METHODS:
    for _encoding in ("onehot", "esm2"):
        setattr(RuntimeTests, f"test_lifecycle_{_method}_{_encoding}", _method_test(_method, _encoding))


if __name__ == "__main__":
    unittest.main()
