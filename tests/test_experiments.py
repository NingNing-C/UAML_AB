"""Small end-to-end checks of the shared experiment and checkpoint services."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

from uaml_ab.experiments import benchmark, evaluate, predict, train
from uaml_ab.runtime import load


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, directory):
        directory = Path(directory)
        sequences = ["".join("C" if number & (1 << bit) else "A" for bit in range(8))
                     for number in range(50)]
        frame = pd.DataFrame({"sequence": sequences,
                              "label": [sequence.count("C") + number/100 for number, sequence in enumerate(sequences)]})
        frame.to_csv(directory / "data.csv", index=False)
        config = directory / "case.yaml"
        config.write_text(yaml.safe_dump({"global": {"dataset": "fixture", "data_path": "data.csv",
                "seq_col": "sequence", "label_col": "label", "wt": "A"*8, "encoder": "onehot"},
                "methods": {"ModelArcEnsemble": {"batch_size": 16},
                            "Deterministic": {"batch_size": 16, "hidden_size": 8}}}))
        return config, frame

    def test_single_run_checkpoint_reload_and_unlabeled_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            config, original = self.fixture(folder)
            record = train(config, epochs=1, n_train=20, output_dir=Path(folder)/"runs", device="cpu")
            run = Path(record["run_dir"])
            fitted = load(run/"checkpoint.pt")
            self.assertEqual(len(fitted.metadata["splits"]["train"]), 18)
            self.assertEqual(len(fitted.metadata["splits"]["validation"]), 2)
            self.assertEqual(len(fitted.metadata["splits"]["test"]), 10)
            # The checkpoint itself is sufficient; sidecar deletion is harmless.
            (run/"run.json").unlink()
            held_out, metrics = evaluate(run/"checkpoint.pt", output_dir=Path(folder)/"evaluation")
            expected = pd.read_csv(run/"predictions.csv")
            pd.testing.assert_frame_equal(held_out, expected, check_dtype=False, rtol=1e-6, atol=1e-6)
            self.assertEqual(metrics["n_evaluated"], 10)
            inputs = Path(folder)/"unlabeled.csv"
            original[["sequence"]].iloc[:3].to_csv(inputs, index=False)
            predictions = predict(run/"checkpoint.pt", inputs, output=Path(folder)/"new.csv")
            self.assertNotIn("label", predictions)
            self.assertEqual(len(predictions), 3)
            self.assertTrue(np.isfinite(predictions[["pred", "epistd", "alestd", "totalstd"]]).all().all())

    def test_sweep_shares_heldout_rows_and_never_adds_test_noise(self):
        with tempfile.TemporaryDirectory() as folder:
            config, original = self.fixture(folder)
            results = benchmark([config], ["Deterministic", "ModelArcEnsemble"],
                                seeds=[7], noise_levels=[0.0, 0.5], epochs=1, n_train=20,
                                device="cpu", output_dir=Path(folder)/"runs")
            self.assertEqual(len(results), 4)
            heldout = None
            for row in results.to_dict("records"):
                run = Path(row["run_dir"])
                table = pd.read_csv(run/"predictions.csv")
                indices = table.row_id.tolist()
                heldout = indices if heldout is None else heldout
                self.assertEqual(indices, heldout)
                np.testing.assert_allclose(table.label, original.loc[indices, "label"])
                metadata = json.loads((run/"run.json").read_text())
                self.assertFalse(set(metadata["splits"]["train"]) & set(indices))
                if row["method"] == "Deterministic":
                    self.assertNotIn("totalstd", table)
                    self.assertFalse(row["uq_available"])

    def test_changed_source_rejected_and_external_evaluation_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            config, frame = self.fixture(folder)
            record = train(config, "Deterministic", epochs=1, n_train=20,
                           output_dir=Path(folder)/"runs", device="cpu")
            checkpoint = Path(record["run_dir"])/"checkpoint.pt"
            altered = frame.copy()
            altered.loc[0, "label"] += 1
            altered.to_csv(Path(folder)/"data.csv", index=False)
            with self.assertRaisesRegex(ValueError, "changed since training"):
                evaluate(checkpoint)
            table, _ = evaluate(checkpoint, external_data=Path(folder)/"data.csv",
                                output_dir=Path(folder)/"external")
            self.assertEqual(len(table), 50)

    def test_installed_cli_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            completed = subprocess.run([sys.executable, "-m", "uaml_ab", "models"],
                                       cwd=folder, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            names = [line.split("\t")[0] for line in completed.stdout.splitlines()]
            self.assertEqual(len(names), 11)
            self.assertIn("ModelArcEnsemble", names)


if __name__ == "__main__":
    unittest.main()
