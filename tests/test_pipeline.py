from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from spend_baselines.data import clean_transactions, PurchaseHistory, feature_matrix
from spend_baselines.cli import temporal_masks, write_submission
from spend_baselines.models import fit_model, predict_logs, rmsle_from_logs


def objects(date="2018-09-01", customer="7"):
    start = pd.Timestamp(date)
    return pd.DataFrame({"Id": ["example"], "customer_id": [customer],
                         "as_of_date": [start], "target_end": [start + pd.Timedelta(days=30)]})


def purchases():
    return pd.DataFrame([
        {"event_ts": d, "order_id": f"order-{i}", "customer_id": "7.0",
         "item_id": "P00001", "qty": 2, "unit_price": v}
        for i, (d, v) in enumerate([
            ("2018-03-05 12:00", 5.0), ("2018-08-31 18:00", 10.0),
            ("2018-09-01 10:00", 100.0), ("2018-09-02 10:00", 200.0),
        ])
    ])


class DataTests(unittest.TestCase):
    def test_cleaning_contract_and_service_boundaries(self):
        base = purchases().iloc[[0]]
        frames = [base, base.copy()]
        for i, updates in enumerate([
            {"order_id": "R-cancel"}, {"qty": -1}, {"unit_price": 0},
            {"customer_id": None}, {"item_id": "SVC00"}, {"item_id": "SVC13"},
            {"item_id": "GFT123"}, {"item_id": "SVC14"},
        ]):
            row = base.copy()
            row["order_id"] = f"special-{i}"
            for k, v in updates.items():
                row[k] = v
            frames.append(row)
        days, stats = clean_transactions(pd.concat(frames, ignore_index=True))
        self.assertEqual(stats["exact_duplicates"], 1)
        self.assertEqual(stats["clean_rows"], 2)  # Regular product + SVC14.
        self.assertEqual(days.customer_id.tolist(), ["7"])
        self.assertEqual(days.amount.iloc[0], 20.0)

    def test_future_and_boundary_cannot_change_past_features(self):
        raw = purchases()
        full, _ = clean_transactions(raw)
        past, _ = clean_transactions(raw.iloc[:2])
        a = PurchaseHistory(full).build_features(objects())
        b = PurchaseHistory(past).build_features(objects())
        assert_frame_equal(a, b)
        self.assertEqual(a.n_180.iloc[0], 2)  # Includes exact lower window boundary.
        self.assertEqual(a.sum_30.iloc[0], 20.0)
        self.assertEqual(a.sum_180.iloc[0], 30.0)
        self.assertEqual(a.recency.iloc[0], 1)

    def test_target_window_includes_start_excludes_end(self):
        raw = purchases()
        extra = raw.iloc[[0]].copy()
        extra["order_id"] = "end-boundary"
        extra["event_ts"] = "2018-10-01 10:00"
        days, _ = clean_transactions(pd.concat([raw, extra]))
        y = PurchaseHistory(days).reconstruct_targets(objects())
        self.assertEqual(y.tolist(), [600.0])

    def test_unknown_or_not_yet_seen_customer_has_no_future_information(self):
        days, _ = clean_transactions(purchases())
        h = PurchaseHistory(days)
        f = h.build_features(objects("2018-01-01"))
        self.assertEqual(f.n_180.iloc[0], 0)
        self.assertEqual(f.tenure.iloc[0], 0)
        self.assertEqual(f.logmean_prediction.iloc[0], 0)
        self.assertTrue(np.isnan(f.median_gap.iloc[0]))
        self.assertTrue(np.isfinite(feature_matrix(f)).all())

    def test_label_availability_controls_training_split(self):
        frame = pd.DataFrame({"as_of_date": pd.to_datetime(["2018-08-01", "2018-08-15", "2018-09-01"]),
                              "target_end": pd.to_datetime(["2018-08-31", "2018-09-14", "2018-10-01"])})
        fit, valid = temporal_masks(frame, "2018-09-01")
        self.assertEqual(fit.tolist(), [True, False, False])
        self.assertEqual(valid.tolist(), [False, False, True])

    def test_submission_is_original_units_and_follows_sample_order(self):
        test = pd.DataFrame({"Id": ["a", "b"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer.csv"
            out = write_submission(test, np.log1p([4.0, 9.0]), path, pd.Series(["b", "a"]))
            self.assertEqual(out.Id.tolist(), ["b", "a"])
            np.testing.assert_allclose(out.spend_30d, [9.0, 4.0])
            self.assertEqual(pd.read_csv(path).columns.tolist(), ["Id", "spend_30d"])
            with self.assertRaises(ValueError):
                write_submission(test, np.array([0, np.inf]), path)

    def test_rmsle_uses_log1p_including_zero(self):
        self.assertEqual(rmsle_from_logs([0, 10], np.log1p([0, 10])), 0.0)
        with self.assertRaises(ValueError):
            rmsle_from_logs([0, 10], [0, -1])


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("configs/baselines.json").read_text())
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(40, 4))
        self.y = np.expm1(np.maximum(0, self.x[:, 0] + 1))

    def test_ridge_and_forest_predict_finite_nonnegative_amounts(self):
        self.config["forest"]["n_estimators"] = 3
        f = pd.DataFrame(index=range(len(self.x)))
        for name in ["ridge", "forest"]:
            model = fit_model(name, self.x, self.y, self.config, threads=1)
            pred = predict_logs(name, model, self.x, f)
            self.assertTrue(np.isfinite(np.expm1(pred)).all())
            self.assertTrue((pred >= 0).all())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency unavailable")
    def test_mlp_seed_and_cpu_training_are_reproducible(self):
        self.config["mlp"].update({"epochs": 2, "batch_size": 16, "hidden_sizes": [8, 4]})
        a = fit_model("mlp", self.x, self.y, self.config, threads=1)
        b = fit_model("mlp", self.x, self.y, self.config, threads=1)
        np.testing.assert_allclose(a.predict(self.x), b.predict(self.x), rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
