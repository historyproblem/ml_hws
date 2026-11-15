"""Adversarial time-boundary, censoring and statistical-model checks."""

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from spend_behavior.data import COLUMNS, calendar_month_features, positive_lines, transaction_features
from spend_behavior.models import (
    DiscreteHazard, SegmentRidge, WeibullAFT, causal_histories,
    chronological_residuals, survival_observations, weibull_objective,
)
from spend_behavior.report import cluster_interval
from spend_behavior.suite import complete, read_config
from spend_baselines.suite_data import atomic_json, digest


class CausalityTests(unittest.TestCase):
    def test_calendar_month_includes_day_31_and_marks_incomplete_coverage(self):
        objects = pd.DataFrame({"customer_id": ["a", "a"],
                                "as_of_date": pd.to_datetime(["2018-10-01", "2017-10-01"])})
        lines = pd.DataFrame({"customer_id": ["a"], "event_ts": pd.to_datetime(["2017-10-31"]),
                              "amount": [9.], "order_id": ["order"]})
        values = calendar_month_features(objects, lines, pd.Timestamp("2017-01-01"))
        self.assertAlmostEqual(values[0, 0], np.log1p(9), places=6)
        self.assertTrue(np.isnan(values[1]).all())

    def test_chronological_oof_prediction_does_not_fit_its_own_label(self):
        dates = pd.date_range("2018-01-01", periods=6, freq="MS")
        objects = pd.DataFrame({"Id": [str(i) for i in range(6)], "customer_id": ["a"] * 6,
                                "as_of_date": dates, "target_end": dates + pd.Timedelta(days=30),
                                "month": dates.strftime("%Y-%m"), "official": True,
                                "spend_30d": np.arange(6) * 10.})
        data = SimpleNamespace(objects=objects, matrix=lambda ix: np.ones((len(ix), 1)))
        class Constant:
            def __init__(self, mean):
                self.mean = mean
            def predict(self, x):
                return np.full(len(x), self.mean)
        with patch("spend_behavior.models.ridge", side_effect=lambda x, y, w, alpha: Constant(np.average(np.log1p(y), weights=w))):
            before, _ = chronological_residuals(data, min_dates=1)
            old_prediction = np.log1p(objects.spend_30d.iloc[3]) - before[3]
            objects.loc[3, "spend_30d"] = 1e10
            after, folds = chronological_residuals(data, min_dates=1)
            new_prediction = np.log1p(objects.spend_30d.iloc[3]) - after[3]
        self.assertAlmostEqual(old_prediction, new_prediction)
        for fold in folds:
            self.assertLessEqual(pd.Timestamp(fold["max_target_end"]), pd.Timestamp(fold["date"]))

    def test_transaction_features_ignore_poisoned_future(self):
        t = pd.Timestamp("2018-08-01")
        rows = pd.DataFrame({"customer_id": ["a"], "as_of_date": [t]})
        records = []
        for month in range(1, 9):
            for day in (1, 8, 15):
                records.append({"customer_id": "a", "event_ts": pd.Timestamp(2017, month, day),
                                "item_id": f"i{day}", "order_id": f"{month}-{day}",
                                "qty": 1, "unit_price": 20})
                records.append({**records[-1], "event_ts": pd.Timestamp(2018, month, day),
                                "order_id": f"2018-{month}-{day}"})
        raw = pd.DataFrame(records)
        raw["amount"] = raw.qty * raw.unit_price
        a = raw.sort_values("event_ts")
        b = a.copy()
        future = b.event_ts.ge(t)
        b.loc[future, "amount"] *= 100000
        b.loc[future, "item_id"] = "future_poison"
        fa, _ = transaction_features(rows, a, a.event_ts.min())
        fb, _ = transaction_features(rows, b, b.event_ts.min())
        np.testing.assert_allclose(fa, fb, equal_nan=True)
        self.assertEqual(fa[0, COLUMNS.index("year_coverage")], 1)

    def test_labels_available_at_target_end_and_no_oof_means_no_effect(self):
        objects = pd.DataFrame({"customer_id": ["a"] * 4,
                                "as_of_date": pd.to_datetime(["2018-06-01", "2018-06-30", "2018-07-01", "2018-08-01"]),
                                "target_end": pd.to_datetime(["2018-07-01", "2018-07-30", "2018-07-31", "2018-08-31"]),
                                "spend_30d": [100., 999999., 200., np.nan],
                                "official": [True, False, True, False]})
        h, effects = causal_histories(objects, np.array([2., 1e6, np.nan, np.nan]), [2])
        self.assertEqual(h[1, 0], 0)
        self.assertAlmostEqual(h[2, 0], np.log1p(1), places=6)
        self.assertAlmostEqual(effects[2][2], 2 / 3)
        self.assertAlmostEqual(effects[2][3], 2 / 3)
        objects.loc[2, "spend_30d"] = 1e10
        hh, ee = causal_histories(objects, np.array([2., 1e6, 1e10, np.nan]), [2])
        np.testing.assert_array_equal(h[:3], hh[:3])
        np.testing.assert_array_equal(effects[2][:3], ee[2][:3])

    def test_partial_future_is_censored_and_cutoff_event_excluded(self):
        objects = pd.DataFrame({"as_of_date": pd.to_datetime(["2018-09-01", "2018-09-22", "2018-11-01"]),
                                "official": [True, True, False], "is_test": [False, False, True]})
        epoch = pd.Timestamp("1970-01-01")
        events = np.array([(pd.Timestamp("2018-10-01") - epoch).days,
                           (pd.Timestamp("2018-10-02") - epoch).days, np.inf])
        rows, duration, event = survival_observations(objects, events, "2018-10-01")
        np.testing.assert_array_equal(rows, [0, 1])
        np.testing.assert_array_equal(duration, [30, 9])
        np.testing.assert_array_equal(event, [False, False])

    def test_cleaning_matches_target_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"customer_id": "1", "event_ts": "2018-01-01", "order_id": o,
                     "item_id": i, "qty": q, "unit_price": 2}
                    for o, i, q in [("a", "p", 1), ("Ra", "p", 1), ("b", "SVC00", 1),
                                    ("c", "GFT1", 1), ("d", "p", -1)]]
            pd.DataFrame(rows).to_csv(Path(tmp) / "transactions.csv", index=False)
            lines, _ = positive_lines(tmp)
            self.assertEqual(lines.amount.sum(), 2)


class ModelTests(unittest.TestCase):
    def test_weibull_gradient_and_probability(self):
        rng = np.random.default_rng(1)
        a = np.c_[np.ones(40), rng.normal(size=(40, 3))]
        duration = rng.uniform(1, 30, 40)
        event, weights = rng.random(40) < .5, rng.uniform(.5, 1.5, 40)
        theta = np.r_[3., .1, -.2, .3, .1]
        fun = lambda t: weibull_objective(t, a, duration, event, weights, .1)[0]
        grad = lambda t: weibull_objective(t, a, duration, event, weights, .1)[1]
        self.assertLess(check_grad(fun, grad, theta), 1e-5)
        model = WeibullAFT().fit(a[:, 1:], duration, event, weights)
        self.assertTrue(np.all((model.predict(a[:, 1:]) > 0) & (model.predict(a[:, 1:]) < 1)))

    def test_hazard_drops_incomplete_negative_period(self):
        x = np.arange(20).reshape(-1, 1).astype(float)
        duration = np.full(20, 9.)
        event = np.arange(20) % 2 == 0
        model = DiscreteHazard().fit(x, duration, event, np.ones(20))
        # Everybody completed bin 0; only observed events enter bin 1.
        self.assertEqual(model.n_periods, 30)
        self.assertTrue(np.all((model.predict(x) >= 0) & (model.predict(x) <= 1)))

    def test_small_segments_fall_back_to_global(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(40, 5))
        model = SegmentRidge("kmeans", 3, "soft", min_customers=100).fit(
            x, x, np.exp(x[:, 0]), np.ones(40), np.arange(40))
        np.testing.assert_allclose(model.predict(x, x), model.global_model.predict(x))
        self.assertTrue(all(s["fallback"] for s in model.summary))

    def test_cluster_bootstrap_keeps_customer_dates_together(self):
        rows = pd.DataFrame({"customer_id": ["a", "a", "b", "b"],
                             "target_z": [1, 1, 2, 2], "z": [1, 1, 2, 2],
                             "reference_z": [2, 2, 3, 3]})
        self.assertEqual(cluster_interval(rows, 100), [-1., -1.])

    def test_config_is_bounded(self):
        cfg = read_config("configs/behavior_suite.json")
        self.assertLessEqual(len(cfg["joint_blocks"]), 2)
        self.assertEqual(cfg["segment_k"], [2, 3, 4, 5])

    def test_completed_result_rejects_corrupt_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "predictions" / "a" / "2018-10-01.csv.gz"
            result = root / "results" / "a" / "2018-10-01.json"
            pred.parent.mkdir(parents=True)
            result.parent.mkdir(parents=True)
            pred.write_bytes(b"correct predictions")
            atomic_json(result, {"predictions_sha256": digest(pred)})
            args = SimpleNamespace(out_dir=root, resume=True)
            self.assertTrue(complete(args, "a", "2018-10-01"))
            pred.write_bytes(b"corruption")
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                complete(args, "a", "2018-10-01")


if __name__ == "__main__":
    unittest.main()
