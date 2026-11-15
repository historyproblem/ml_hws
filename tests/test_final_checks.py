"""Feature-time safety, frozen-suite compatibility and final artifact checks."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scripts.final_check_features import (
    SEASON_COLUMNS, TARGET_COLUMNS, build_extra_features, causal_target_prior,
)
from scripts.final_check_models import FinalJoint, weighted_joint_loss
from scripts.final_check_io import FrozenReference, export_locked_submissions
from scripts.run_final_checks import read_final_config, plan_jobs
from spend_baselines.suite import job_key
from spend_baselines.suite_data import digest
from spend_baselines.suite_models import Neural, neural_loss


class FinalFeatureTests(unittest.TestCase):
    def fixtures(self):
        days = pd.DataFrame({"customer_id": ["a"]*5,
                             "date": pd.to_datetime(["2016-12-15", "2017-11-10", "2018-10-01", "2018-10-02", "2018-10-31"]),
                             "amount": [2.0, 100.0, 1000.0, 3.0, 5.0]})
        objects = pd.DataFrame({"customer_id": ["a", "a", "b"],
                                "as_of_date": pd.to_datetime(["2017-11-01", "2018-11-01", "2018-11-01"])})
        official = pd.DataFrame({"target_end": pd.to_datetime(["2017-10-31", "2017-12-01", "2018-12-01"]),
                                 "spend_30d": [4.0, 100.0, 1e9]})
        return objects, days, official

    def test_missing_year_history_is_not_zero_purchase(self):
        objects, days, official = self.fixtures()
        a = build_extra_features(objects, days, official, history_start="2016-12-01")
        c = SEASON_COLUMNS.index
        self.assertEqual(a["season"][0, c("yearago_available")], 0)
        self.assertTrue(np.isnan(a["season"][0, c("yearago_log_spend30")]))
        self.assertEqual(a["season"][1, c("yearago_available")], 1)
        self.assertAlmostEqual(a["season"][1, c("yearago_log_spend30")], np.log1p(100), places=6)
        self.assertEqual(a["season"][2, c("yearago_available")], 1)
        self.assertEqual(a["season"][2, c("yearago_log_spend30")], 0)

    def test_future_transactions_and_unfinished_targets_cannot_change_features(self):
        objects, days, official = self.fixtures()
        before = build_extra_features(objects, days, official, history_start="2016-12-01")
        poisoned = pd.concat([days, pd.DataFrame({"customer_id": ["a"], "date": [pd.Timestamp("2018-11-01")], "amount": [1e12]})])
        official.loc[official.target_end.gt(objects.as_of_date.max()), "spend_30d"] = 1e15
        after = build_extra_features(objects, poisoned, official, history_start="2016-12-01")
        for key in before:
            np.testing.assert_allclose(before[key], after[key], equal_nan=True)
        self.assertAlmostEqual(before["target_history"][1, TARGET_COLUMNS.index("target_log_lag_01")], np.log1p(8), places=6)

    def test_calendar_and_prior_boundaries(self):
        objects, days, official = self.fixtures()
        result = build_extra_features(objects, days, official, history_start="2016-12-01")
        self.assertEqual(result["season"][1, SEASON_COLUMNS.index("forecast_month_share_11")], 1)
        pm, pp, positive, available = causal_target_prior(official, pd.to_datetime(["2017-10-30", "2017-10-31", "2017-12-01"]))
        self.assertFalse(available[0])
        self.assertTrue(np.isnan(pm[0]))
        self.assertAlmostEqual(pm[1], np.log1p(4))
        self.assertAlmostEqual(pm[2], (np.log1p(4)+np.log1p(100))/2)

    def test_final_plan_is_small_and_does_not_change_frozen_config(self):
        c = read_final_config("configs/final_checks.json")
        jobs = plan_jobs(c)
        self.assertEqual(len(jobs), 85)
        self.assertEqual(len({job_key(j) for j in jobs}), 85)
        self.assertTrue(all(j["date"] in {"2018-09-01", "2018-10-01"} for j in jobs[:34]))
        self.assertEqual(len(SEASON_COLUMNS), 24)
        self.assertEqual(len(TARGET_COLUMNS), 15)


class FinalLossTests(unittest.TestCase):
    def settings(self):
        return {"hidden_sizes": [8, 4], "weight_decay": .0001, "dropout": 0,
                "learning_rate": .001, "batch_size": 16, "max_epochs": 3,
                "patience": 1, "min_delta": .0001,
                "loss_weights": {"purchase_bce": 1., "positive_log_mse": 1., "total_log_mse": 1.}}

    def test_default_weighted_loss_and_training_match_frozen_joint(self):
        import torch
        torch.manual_seed(5)
        raw = torch.randn(16, 2, requires_grad=True)
        y = torch.tensor([0., 1., 3., 0.]*4)
        w = torch.linspace(.2, 1., 16)
        a = neural_loss(raw, y, w, "hurdle_joint")
        b = weighted_joint_loss(raw, y, w, self.settings()["loss_weights"])
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        rng = np.random.default_rng(42)
        x = rng.normal(size=(40, 4)).astype(np.float32)
        yy = np.expm1(np.maximum(x[:, 0]+1, 0))
        weights = np.linspace(.2, 1., len(x))
        old = Neural("mlp", "hurdle_joint", self.settings(), 42, "cpu", 1).fit(x, yy, weights, epochs=2)
        new = FinalJoint("mlp", "hurdle_joint", self.settings(), 42, "cpu", 1).fit(x, yy, weights, epochs=2)
        np.testing.assert_allclose(old.predict(x), new.predict(x), rtol=0, atol=1e-7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"
            new.save(path)
            loaded = Neural.load(path)
            np.testing.assert_allclose(new.predict(x), loaded.predict(x), rtol=0, atol=1e-7)


class LockedArtifactTests(unittest.TestCase):
    def test_primary_file_can_be_exported_before_wide_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);data = root/"data";data.mkdir()
            sample = pd.DataFrame({"Id": ["a", "b"], "spend_30d": [0., 0.]})
            sample.to_csv(data/"sample_submission.csv", index=False)
            pd.DataFrame({"Id": ["a", "b"], "as_of_date": ["2018-11-01"]*2}).to_csv(data/"test.csv", index=False)
            c = read_final_config("configs/final_checks.json")
            reference = root/"reference"
            for name, values in [(c["locked_primary"], [1., 3.]), (c["locked_reserve"], [5., 8.])]:
                key = job_key({"candidate": name, "dataset": "D2", "representation": "F2", "date": "2018-11-01"})
                directory = reference/"candidates"/key;directory.mkdir(parents=True)
                pd.DataFrame({"Id": ["a", "b"], "spend_30d": values}).to_csv(directory/"submission.csv", index=False)
                (directory/"result.json").write_text(json.dumps({"candidate": name, "dataset": "D2", "representation": "F2",
                    "sha256": {"submission.csv": digest(directory/"submission.csv")}}))
            destination = root/"submissions"
            path = export_locked_submissions(reference, {**c, "refit_reserve": False}, data, destination)
            self.assertTrue(path.is_file())
            export_locked_submissions(reference, c, data, destination)
            blended = pd.read_csv(destination/"joint75_wide25.csv")
            expected = np.expm1(.75*np.log1p([1., 3.])+.25*np.log1p([5., 8.]))
            np.testing.assert_allclose(blended.spend_30d, expected)
            np.testing.assert_allclose(pd.read_csv(path).spend_30d, [1., 3.])


if __name__ == "__main__":
    unittest.main()
