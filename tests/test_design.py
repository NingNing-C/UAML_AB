"""Design regressions through the same canonical runtime as ordinary prediction."""
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from uaml_ab.design.acquisition import propose_candidates
from uaml_ab.design.ga.engine import GAEngine
from uaml_ab.design.ga.fitness import make_predict_fn
from uaml_ab.design.ga.run import parse_args as ga_args, run_ga
from uaml_ab.design.ga.select import main as select_main
from uaml_ab.design.bo.run import parse_args as bo_args, run_bo
from uaml_ab.design.bo.round import fit_round
from uaml_ab.runtime import fit, load, default_settings
from uaml_ab.data import split_train_val, split_data_random
from uaml_ab.config import load_config
from uaml_ab.records import source_fingerprint


class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.wt = "AAAAAAAA"
        sequences = [cls.wt]
        for count in (1, 2):
            for positions in itertools.combinations(range(8), count):
                sequence = list(cls.wt)
                for position in positions:
                    sequence[position] = "C"
                sequences.append("".join(sequence))
        cls.configs, cls.checkpoints = [], []
        for index, antigen in enumerate(("M22", "SI06")):
            labels = np.array([sum(c != "A" for c in sequence) for sequence in sequences], dtype=float)
            labels = 1 + index + .25 * labels
            frame = pd.DataFrame({"sequence": sequences, "label": labels})
            frame.to_csv(cls.directory / f"{antigen}.csv", index=False)
            settings = default_settings("ModelArcEnsemble")
            settings.update(batch_size=16, lr=.001, weight_decay=1e-5)
            train, val, held_out = split_data_random(frame, random_state=42)
            fitted = fit("ModelArcEnsemble", train, val, settings, "cpu", 42, 1)
            checkpoint = cls.directory / f"{antigen}.pt"
            cls.checkpoints.append(checkpoint)
            config = {"global": {"dataset": f"toy-{antigen}", "data_path": f"{antigen}.csv",
                                  "seq_col": "sequence", "label_col": "label", "encoder": "onehot", "wt": cls.wt},
                      "methods": {"ModelArcEnsemble": settings}}
            path = cls.directory / f"{antigen}.yaml"
            path.write_text(yaml.safe_dump(config))
            cls.configs.append(path)
            fitted.metadata.update({"split_seed": 42, "splits": {"train": train.index.tolist(),
                                    "validation": val.index.tolist(), "test": held_out.index.tolist()},
                                    "source_fingerprint": source_fingerprint(load_config(path), frame, "onehot")})
            fitted.save(checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_gate_margin_scores_use_weighted_uncertainty(self):
        frame = pd.DataFrame({"split_label": ["test", "test"], "pred": [1., 2.], "totalstd": [.1, .4]})
        scored = propose_candidates(frame, 1., normalize="gate_margin", ei_threshold=1.,
                                    unc_weight=.5, norm_ref={"pred": (0., 2.)})
        np.testing.assert_allclose(scored.greedy_score, [0., .5])
        np.testing.assert_allclose(scored.lcb_score, [-.025, .4])
        np.testing.assert_allclose(scored.ucb_score, [.025, .6])
        self.assertGreater(scored.ei_score.iloc[0], 0)

    def test_prospective_predictions_do_not_require_dummy_labels(self):
        fitted = load(self.checkpoints[0])
        frame = pd.DataFrame({"sequence": [self.wt, "CCCAAAAA"]})
        direct = fitted.predict(frame)
        wrapped = make_predict_fn(fitted)(frame.sequence.tolist())
        for name in ("pred", "epistd", "alestd", "totalstd"):
            np.testing.assert_allclose(wrapped[name], direct[name], rtol=1e-6, atol=1e-7)
        bad_labels = frame.assign(label=[np.nan, np.nan])
        for name, values in direct.items():
            np.testing.assert_allclose(fitted.predict(bad_labels)[name], values)

    def test_bo_round_passes_shared_initialization_seed(self):
        frame = pd.DataFrame({"sequence": [self.wt] * 12, "label": np.arange(12.)})
        class Result:
            metadata = {}
            settings = {}
        with patch("uaml_ab.design.bo.round.fit", return_value=Result()) as common_fit:
            fit_round(frame, {"encoder": "onehot"}, {}, "cpu", 1, 51)
        self.assertEqual(common_fit.call_args.kwargs["initialization_seed"], 51)
        train, validation = common_fit.call_args.args[1:3]
        self.assertTrue(set(train.index).isdisjoint(validation.index))

    def test_ga_rejects_changed_source_before_calibrating(self):
        changed = pd.read_csv(self.directory / "M22.csv")
        changed["label"] += .01
        changed.to_csv(self.directory / "M22_changed.csv", index=False)
        config = yaml.safe_load(self.configs[0].read_text())
        config["global"]["data_path"] = "M22_changed.csv"
        path = self.directory / "changed.yaml"
        path.write_text(yaml.safe_dump(config))
        args = ga_args(["--config", str(path), "--checkpoint", str(self.checkpoints[0]),
                        "--device", "cpu", "--out_dir", str(self.directory / "rejected")])
        with self.assertRaisesRegex(ValueError, "differs from the checkpoint"):
            run_ga(args)

    def test_tiny_canonical_ga_has_qualifiers_and_final_shortlist(self):
        output = self.directory / "ga"
        args = ga_args(["--configs", *(str(p) for p in self.configs),
                        "--checkpoints", *(str(p) for p in self.checkpoints),
                        "--antigen_names", "M22", "SI06", "--acquisitions", "greedy", "ei", "lcb",
                        "--sigma_types", "total", "--weight_modes", "calibrated", "--seedings", "top_binders",
                        "--ga_seeds", "42", "--pop", "16", "--generations", "2", "--seed_pool_size", "12",
                        "--min_mut", "3", "--trust_radius", "5", "--alphabet_mode", "all",
                        "--gate_percentile", "0", "--n_select", "8", "--device", "cpu", "--out_dir", str(output)])
        summary = run_ga(args)
        self.assertEqual(len(summary), 3)
        self.assertGreater(summary.n_shortlist.sum(), 0)
        select_main(["--run_dir", str(output), "--n_per_acquisition", "8", "--n_experimental", "6"])
        selected = pd.read_csv(output / "candidates_by_acquisition.csv")
        experimental = pd.read_csv(output / "candidates_by_acquisition_top6.csv")
        self.assertGreater(len(experimental), 0)
        for acquisition, group in selected.groupby("acquisition"):
            self.assertEqual(experimental[experimental.acquisition == acquisition].sequence.tolist(), group.head(6).sequence.tolist())
        self.assertTrue((experimental.num_mut >= 3).all())

    def test_single_and_two_objective_bo_save_canonical_round_models(self):
        for count in (1, 2):
            with self.subTest(objectives=count):
                output = self.directory / f"bo{count}"
                args = bo_args(["--configs", *(str(p) for p in self.configs[:count]),
                                "--epochs", "1", "--n_init", "12", "--n_rounds", "1", "--batch_size", "3",
                                "--device", "cpu", "--out_dir", str(output)])
                results = run_bo(args)
                self.assertEqual(len(results), 1)
                state = pd.read_csv(output / "pool_state.csv")
                self.assertEqual((state.round_acquired == 0).sum(), 3)
                self.assertEqual((state.pool == "held_out").sum(), 7)
                checkpoints = list((output / "checkpoints").rglob("model.pt"))
                self.assertEqual(len(checkpoints), count)
                for checkpoint in checkpoints:
                    model = load(checkpoint)
                    self.assertEqual(model.method, "ModelArcEnsemble")
                    self.assertEqual(model.initialization_seed, 42)
                    self.assertEqual(model.metadata["workflow"], "retrospective_bo")
                    self.assertTrue(model.metadata["no_saved_test_partition"])
                    self.assertIn("global", model.metadata["config"])
                    self.assertIn("ModelArcEnsemble", model.metadata["config"]["methods"])


if __name__ == "__main__":
    unittest.main()
