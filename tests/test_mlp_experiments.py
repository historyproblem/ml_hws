import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from spend_baselines.experiments import (
    inner_temporal_split, experiment_plan, guard_run_directory, completed_job, write_summaries,
)
from spend_baselines.models import TorchMLP


class ExperimentProtocolTests(unittest.TestCase):
    def test_inner_split_purges_labels_overlapping_stopping_date(self):
        data = pd.DataFrame({
            "as_of_date": pd.to_datetime(["2018-06-01", "2018-07-20", "2018-08-01", "2018-09-01", "2018-10-01"]),
            "target_end": pd.to_datetime(["2018-07-01", "2018-08-19", "2018-08-31", "2018-10-01", "2018-10-31"]),
        })
        split = inner_temporal_split(data, "2018-09-01")
        self.assertEqual(split["inner_date"], pd.Timestamp("2018-08-01"))
        self.assertEqual(split["inner_fit"].tolist(), [True, False, False, False, False])
        self.assertEqual(split["inner_valid"].tolist(), [False, False, True, False, False])
        self.assertEqual(split["outer_fit"].tolist(), [True, True, True, False, False])
        self.assertEqual(split["outer_valid"].tolist(), [False, False, False, True, False])
        self.assertEqual(split["n_purged"], 1)

    def test_plan_is_explicit_and_does_not_mutate_baseline_configuration(self):
        base = json.loads(Path("configs/baselines.json").read_text())
        specification = json.loads(Path("configs/mlp_experiments.json").read_text())
        plan = experiment_plan(base, specification, variants=["dropout_10"], seeds=[7], epochs=3)
        self.assertEqual(plan["variants"][0]["settings"]["dropout"], 0.1)
        self.assertEqual(plan["variants"][0]["settings"]["epochs"], 3)
        self.assertEqual(base["mlp"]["epochs"], 60)
        self.assertNotIn("dropout", base["mlp"])
        with self.assertRaises(ValueError):
            experiment_plan(base, specification, variants=["unknown"])
        with self.assertRaises(ValueError):
            experiment_plan(base, specification, seeds=[42, 42])

    def test_resume_refuses_changed_run_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = {"code": "abc", "seed": 42}
            guard_run_directory(directory, manifest)
            guard_run_directory(directory, manifest, resume=True)
            with self.assertRaises(ValueError):
                guard_run_directory(directory, manifest)
            with self.assertRaises(ValueError):
                guard_run_directory(directory, {"code": "different", "seed": 42}, resume=True)

    def test_completed_job_requires_all_artifacts_and_matching_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.assertIsNone(completed_job(directory))
            names = ["checkpoint.pt", "selection_history.csv", "refit_history.csv", "validation_predictions.csv"]
            for name in names:
                (directory / name).write_text("verified artifact")
            result = {"artifact_sha256": {n: hashlib.sha256((directory / n).read_bytes()).hexdigest() for n in names}}
            (directory / "result.json").write_text(json.dumps(result))
            self.assertEqual(completed_job(directory), result)
            (directory / "checkpoint.pt").write_text("truncated")
            with self.assertRaises(ValueError):
                completed_job(directory)

    def test_pooled_metric_uses_row_weights_in_log_squared_error(self):
        rows = [{"variant": "manual", "seed": 42, "date": "2018-09-01", "RMSLE": 1.0,
                 "n_valid": 1, "selected_epoch": 5},
                {"variant": "manual", "seed": 42, "date": "2018-10-01", "RMSLE": 3.0,
                 "n_valid": 3, "selected_epoch": 8}]
        with tempfile.TemporaryDirectory() as tmp:
            write_summaries(rows, Path(tmp))
            pooled = pd.read_csv(Path(tmp) / "pooled_by_seed.csv")
            self.assertAlmostEqual(pooled.pooled_rmsle.iloc[0], np.sqrt(7))
            self.assertEqual(pooled.n_folds.iloc[0], 2)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency unavailable")
class EarlyStoppingTests(unittest.TestCase):
    def setUp(self):
        self.settings = {"hidden_sizes": [8, 4], "epochs": 20, "batch_size": 16,
                         "learning_rate": 0.001, "weight_decay": 0.0001, "dropout": 0.1}
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(48, 3))
        self.z = np.maximum(0, self.x[:, 0] + 2)

    def test_patience_restores_selected_epoch_weights(self):
        # A large required improvement forces epoch 1 to remain the selected epoch.
        stopped = TorchMLP(self.settings, threads=1).fit(
            self.x[:32], self.z[:32], validation=(self.x[32:], self.z[32:]),
            early_stopping={"patience": 2, "min_delta": 100.0})
        self.assertEqual(stopped.best_epoch_, 1)
        self.assertEqual(stopped.epochs_run_, 3)
        self.assertEqual(stopped.stop_reason_, "patience_reached")
        once = TorchMLP({**self.settings, "epochs": 1}, threads=1).fit(self.x[:32], self.z[:32])
        np.testing.assert_allclose(stopped.predict(self.x), once.predict(self.x), atol=1e-7, rtol=0)

    def test_validation_does_not_fit_scaling_or_imputation(self):
        shifted = self.x[32:] + 1e6
        model = TorchMLP({**self.settings, "epochs": 1}, threads=1).fit(
            self.x[:32], self.z[:32], validation=(shifted, self.z[32:]),
            early_stopping={"patience": 2, "min_delta": 0.0})
        np.testing.assert_allclose(model.preprocessor.steps[1][1].mean_, self.x[:32].mean(axis=0))

    def test_portable_checkpoint_roundtrip_with_and_without_missing_indicators(self):
        for has_missing in [False, True]:
            with self.subTest(has_missing=has_missing):
                x = self.x.copy()
                if has_missing:
                    x[::3, 1] = np.nan
                model = TorchMLP({**self.settings, "epochs": 2}, threads=1).fit(x, self.z)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "checkpoint.pt"
                    model.save(path, metadata={"purpose": "unit test"})
                    loaded = TorchMLP.load(path, threads=1)
                    self.assertEqual(loaded.metadata_["purpose"], "unit test")
                    np.testing.assert_allclose(loaded.predict(x), model.predict(x), rtol=0, atol=1e-7)
                    with self.assertRaises(ValueError):
                        loaded.predict(np.ones((2, 8)))


if __name__ == "__main__":
    unittest.main()
