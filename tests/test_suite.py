"""Temporal correctness, numeric CLV checks and complete neural smoke coverage."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import gammaln

from spend_baselines.data import PurchaseHistory, clean_transactions
from spend_baselines.suite import (
    combine,
    existing_result,
    job_key,
    make_plan,
    metric,
    read_config,
)
from spend_baselines.suite_clv import CustomerModel, OrderHistory, likelihood
from spend_baselines.suite_data import (
    BASE_COLUMNS,
    digest,
    prepare_cache,
    row_weights,
    split_indices,
)
from spend_baselines.suite_models import Neural, output_prediction


def synthetic_dataset(directory, poison=False):
    rows = []
    for cid in range(1, 7):
        for month in range(1, 11):
            for day in (3, 19):
                rows.append(
                    {
                        "customer_id": str(cid),
                        "order_id": f"{cid}-{month}-{day}",
                        "event_ts": f"2018-{month:02d}-{day:02d} 12:00",
                        "item_id": f"P{cid}",
                        "qty": cid,
                        "unit_price": 2.0,
                        "region": "A",
                    }
                )
    # The same order ID has a future line: it must not change its past amount.
    rows.append(
        dict(
            rows[0],
            event_ts="2018-08-20 13:00",
            qty=100000 if poison else 2,
            region="Z",
        )
    )
    rows.append(
        dict(
            rows[0],
            order_id="R-return",
            event_ts="2018-08-21 13:00",
            qty=-100000 if poison else -2,
            region="Z",
        )
    )
    raw = pd.DataFrame(rows)
    raw.to_csv(directory / "transactions.csv", index=False)
    days, _ = clean_transactions(raw)
    h = PurchaseHistory(days)
    objects = []
    for date in pd.to_datetime(
        ["2018-06-01", "2018-07-01", "2018-08-01", "2018-09-01", "2018-10-01"]
    ):
        for cid in range(1, 7):
            objects.append(
                {
                    "Id": f"{date.date()}:{cid}",
                    "customer_id": str(cid),
                    "as_of_date": date,
                    "target_end": date + pd.Timedelta(days=30),
                }
            )
    train = pd.DataFrame(objects)
    train["spend_30d"] = h.reconstruct_targets(train)
    train.to_csv(directory / "train.csv", index=False)
    test = train[train.as_of_date.eq("2018-10-01")].drop(columns="spend_30d").copy()
    test["as_of_date"] = pd.Timestamp("2018-11-01")
    test["target_end"] = pd.Timestamp("2018-12-01")
    test["Id"] = "test:" + test.customer_id
    test.to_csv(directory / "test.csv", index=False)
    pd.DataFrame({"Id": test.Id, "spend_30d": 0}).to_csv(
        directory / "sample_submission.csv", index=False
    )


class SnapshotTests(unittest.TestCase):
    def test_all_features_and_sequences_ignore_future_transactions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a"
            b = root / "b"
            a.mkdir()
            b.mkdir()
            synthetic_dataset(a)
            synthetic_dataset(b, poison=True)
            config = {"datasets": ["D0", "D1", "D2", "D3"]}
            ca = prepare_cache(a, root / "ca", config, {"test": "a"})
            cb = prepare_cache(b, root / "cb", config, {"test": "b"})
            mask = ca.objects.as_of_date.le("2018-08-01")
            self.assertEqual(ca.objects.Id.tolist(), cb.objects.Id.tolist())
            np.testing.assert_allclose(
                ca.features[mask], cb.features[mask], equal_nan=True
            )
            np.testing.assert_array_equal(ca.sequence[mask], cb.sequence[mask])
            self.assertEqual(ca.objects.official.sum(), 30)
            self.assertTrue(ca.objects.as_of_date.dt.day.eq(2).any())
            self.assertTrue(
                ca.objects.loc[~ca.objects.is_test, "spend_30d"].notna().all()
            )
            with self.assertRaises(ValueError):
                prepare_cache(a, root / "ca", config, {"test": "changed"})
            # Monetary summaries must also ignore a future line of an old order.
            sa = OrderHistory(a).summary("2018-07-01", "all")
            sb = OrderHistory(b).summary("2018-07-01", "all")
            pd.testing.assert_frame_equal(sa, sb)

    def test_weights_are_normalised_after_purge(self):
        objects = pd.DataFrame(
            {
                "Id": ["a", "b", "c", "d"],
                "customer_id": ["1", "1", "1", "2"],
                "as_of_date": pd.to_datetime(
                    ["2018-08-01", "2018-08-08", "2018-08-22", "2018-08-01"]
                ),
                "target_end": pd.to_datetime(
                    ["2018-08-31", "2018-09-07", "2018-09-21", "2018-08-31"]
                ),
                "month": ["2018-08"] * 4,
                "official": [True, False, False, True],
            }
        )
        ix = split_indices(objects, "D2", "2018-09-10")
        np.testing.assert_array_equal(ix, [0, 1, 3])
        w = row_weights(objects, ix, "D2")
        self.assertAlmostEqual(w.sum(), 2.0)
        self.assertAlmostEqual(w[0] + w[1], w[2])
        wb = row_weights(objects, ix, "D2", {"1": 2, "2": 1})
        self.assertAlmostEqual(wb[0] + wb[1], 2 * wb[2])
        np.testing.assert_array_equal(
            split_indices(objects, "D0", "2018-09-10"), [0, 3]
        )


class ProbabilityModelTests(unittest.TestCase):
    def test_pareto_integral_matches_independent_adaptive_quadrature(self):
        r, alpha, s, beta = 1.3, 4.2, 0.8, 6.1
        x = np.array([0.0, 1.0, 5.0])
        tx = np.array([0.0, 3.0, 9.0])
        age = np.array([20.0, 12.0, 15.0])
        log, alive = likelihood([r, alpha, s, beta], x, tx, age, "pareto", 128)
        for i in range(len(x)):
            constant = np.exp(
                gammaln(r + x[i]) - gammaln(r) + r * np.log(alpha) + s * np.log(beta)
            )
            active = constant / (alpha + age[i]) ** (r + x[i]) / (beta + age[i]) ** s
            dead = (
                constant
                * s
                * quad(
                    lambda t, exponent=-r - x[i]: (
                        (alpha + t) ** exponent * (beta + t) ** (-s - 1)
                    ),
                    tx[i],
                    age[i],
                    epsabs=1e-13,
                )[0]
            )
            self.assertAlmostEqual(log[i], np.log(active + dead), places=9)
            self.assertAlmostEqual(alive[i], np.log(active), places=9)

    def test_bg_survival_and_monetary_shrinkage(self):
        summary = pd.DataFrame(
            {
                "customer_id": ["a", "b"],
                "x": [0.0, 4.0],
                "tx": [0.0, 20.0],
                "age": [30.0, 30.0],
                "n": [1.0, 5.0],
                "mean": [10.0, 100.0],
            }
        )
        model = CustomerModel("bg", {"quadrature_order": 64})
        model.parameters = np.array([1.0, 20.0, 2.0, 4.0])
        model.money = np.array([2.0, 3.0, 20.0])
        values = model.values(summary)
        from scipy.stats import beta

        for i in range(len(summary)):
            row = summary.iloc[i]
            expected = (
                quad(
                    lambda p, frequency=row.x: (
                        (1 - (50 / (50 + 30 * p)) ** (1 + frequency))
                        / p
                        * beta.pdf(p, 2, 4 + frequency)
                    ),
                    0,
                    1,
                )[0]
                * values[i, 0]
            )
            self.assertAlmostEqual(values[i, 1], expected, places=9)
        self.assertEqual(values[0, 0], 1.0)
        self.assertTrue(((values[:, 0] >= 0) & (values[:, 0] <= 1)).all())
        self.assertAlmostEqual(values[0, 2], 15.0)
        self.assertTrue(20 < values[1, 2] < 100)
        z, mean, se = model.predict(summary, draws=8192, seed=5)
        self.assertTrue(np.all(z <= np.log1p(mean) + 5 * se))
        self.assertTrue((z >= 0).all())


class PlanAndPersistenceTests(unittest.TestCase):
    def test_both_plans_are_unique_and_count_fits_explicitly(self):
        for name, n in [("v100_suite.json", 515), ("full_suite.json", 49880)]:
            plan = make_plan(read_config(Path("configs") / name))
            self.assertEqual(len(plan), n)
            self.assertEqual(len({job_key(j) for j in plan}), n)
        smoke = make_plan(read_config("configs/full_suite.json", True))
        self.assertEqual(len(smoke), 128)
        self.assertEqual(
            {j["kind"] for j in smoke}, {"tabular", "neural", "bagging", "clv"}
        )

    def test_focused_plan_tests_specific_questions_and_shares_overlapping_fits(self):
        plan = make_plan(read_config("configs/v100_suite.json"))
        self.assertTrue(all(j.get("dataset") not in {"D1", "D3"} for j in plan))
        self.assertTrue(
            all(j.get("representation") not in {"F3", "F4", "S0", "S1"} for j in plan)
        )
        self.assertEqual({j["window"] for j in plan if j["kind"] == "clv"}, {"all"})
        self.assertEqual({j["family"] for j in plan if j["kind"] == "bagging"}, {"mlp"})
        self.assertEqual(
            {j["mode"] for j in plan if j["kind"] == "bagging"}, {"seed", "customer"}
        )
        for dataset, representation, seeds in [
            ("D0", "F0", {42}),
            ("D2", "F2", {42, 123, 2026}),
        ]:
            actual = [
                j
                for j in plan
                if j.get("dataset") == dataset
                and j.get("representation") == representation
                and j["kind"] == "neural"
                and j["family"] == "mlp"
                and j["task"] == "direct"
                and j["profile"] == "compact_l2"
                and j["date"] == "2018-10-01"
            ]
            self.assertEqual({j["seed"] for j in actual}, seeds)
            self.assertEqual(len(actual), len(seeds))
        invalid = read_config("configs/v100_suite.json")
        invalid["experiments"][0]["cells"] = [["D3", "F4"]]
        with self.assertRaises(ValueError):
            make_plan(invalid)

    def test_resume_checks_every_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            job = {"date": "2018-09-01"}
            self.assertIsNone(existing_result(dest, job))
            (dest / "predictions.npz").write_bytes(b"complete")
            (dest / "result.json").write_text(
                json.dumps(
                    {
                        "job": job,
                        "sha256": {"predictions.npz": digest(dest / "predictions.npz")},
                    }
                )
            )
            self.assertIsNotNone(existing_result(dest, job))
            (dest / "predictions.npz").write_bytes(b"truncated")
            with self.assertRaises(ValueError):
                existing_result(dest, job)

    def test_metric_is_from_log_squared_errors(self):
        a = metric(
            np.array([0.0, 3.0, 0.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([0.1, 0.9, 0.2]),
        )
        self.assertAlmostEqual(a["squared_log_error"], 1 + (np.log(4) - 2) ** 2 + 9)
        self.assertAlmostEqual(a["Brier"], (0.01 + 0.01 + 0.04) / 3)

    def test_router_never_trains_on_current_or_future_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            results = []
            rows = []
            rng = np.random.default_rng(4)
            for date in ["2018-06-01", "2018-07-01", "2018-08-01"]:
                ids = np.array([date + ":" + str(i) for i in range(16)])
                for i, identifier in enumerate(ids):
                    rows.append(
                        {
                            "Id": identifier,
                            "customer_id": str(i),
                            "as_of_date": pd.Timestamp(date),
                            "target_end": pd.Timestamp(date) + pd.Timedelta(days=30),
                            "spend_30d": float(np.expm1(rng.uniform(0, 3))),
                        }
                    )
                for family, kind in [
                    ("linear", "tabular"),
                    ("forest", "tabular"),
                    ("mlp", "neural"),
                ]:
                    job = {
                        "date": date,
                        "dataset": "D0",
                        "representation": "F0",
                        "kind": kind,
                        "family": family,
                        "task": "direct",
                        "seed": 42,
                    }
                    if kind == "neural":
                        job["profile"] = "compact_l2"
                    dest = directory / "jobs" / job_key(job)
                    dest.mkdir(parents=True)
                    np.savez(
                        dest / "predictions.npz",
                        Id=ids,
                        z=rng.uniform(0.1, 3, len(ids)),
                        probability=np.array([]),
                    )
                    results.append({"job": job})
            objects = pd.DataFrame(rows)
            cache = SimpleNamespace(
                objects=objects,
                features=rng.uniform(size=(len(objects), len(BASE_COLUMNS))),
            )
            config = {
                "datasets": ["D0"],
                "tabular_features": ["F0"],
                "validation_dates": ["2018-06-01", "2018-07-01", "2018-08-01"],
                "seeds": [42],
                "component_profile": "compact_l2",
                "bagging": {"prefixes": [1, 2]},
            }
            combine(results, config, cache, directory)

            def router():
                key = job_key(
                    {
                        "name": "oot_soft_router",
                        "date": "2018-07-01",
                        "dataset": "D0",
                        "representation": "F0",
                    }
                )
                with np.load(directory / "derived" / f"{key}.npz") as p:
                    return p["z"]

            before = router()
            cache.objects.loc[
                cache.objects.as_of_date.ge("2018-07-01"), "spend_30d"
            ] *= 100000
            combine(results, config, cache, directory)
            np.testing.assert_array_equal(before, router())


class NeuralSuiteTests(unittest.TestCase):
    def test_all_architectures_train_and_restore_predictions(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(30, 55)).astype(np.float32)
        x[0, 0] = np.nan
        y = np.where(np.arange(30) % 3 == 0, 0.0, np.expm1(rng.uniform(0.5, 2, 30)))
        w = np.ones(30)
        w[1] = 0
        settings = {
            "hidden_sizes": [8, 4],
            "dropout": 0.1,
            "weight_decay": 0.001,
            "learning_rate": 0.001,
            "max_epochs": 2,
            "batch_size": 16,
            "patience": 1,
            "min_delta": 0.0,
        }
        for architecture in ("mlp", "wide", "residual", "tcn", "gru", "lstm"):
            for task in ("direct", "hurdle_joint", "ziln"):
                with self.subTest(architecture=architecture, task=task):
                    shape = (26, 2) if architecture in {"tcn", "gru", "lstm"} else None
                    model = Neural(
                        architecture, task, settings, threads=1, sequence_shape=shape
                    )
                    model.fit(x[:24], y[:24], w[:24], validation=(x[24:], y[24:]))
                    predicted = model.predict(x)
                    self.assertTrue(np.isfinite(predicted).all())
                    self.assertTrue((predicted >= 0).all())
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "model.pt"
                        model.save(path)
                        loaded = Neural.load(path, threads=1)
                        np.testing.assert_allclose(
                            predicted, loaded.predict(x), rtol=0, atol=1e-7
                        )
                    if task != "direct":
                        p = model.predict_probability(x)
                        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_ziln_prediction_is_expected_log1p_not_expected_money(self):
        import torch

        raw = torch.tensor(
            [[-np.log(4), np.log(100), np.log(np.expm1(0.95))]], dtype=torch.float64
        )
        predicted = output_prediction(raw, "ziln").item()
        expected = (
            0.2
            * quad(
                lambda z: (
                    np.logaddexp(0, np.log(100) + z)
                    * np.exp(-z * z / 2)
                    / np.sqrt(2 * np.pi)
                ),
                -10,
                10,
            )[0]
        )
        self.assertAlmostEqual(predicted, expected, places=7)


if __name__ == "__main__":
    unittest.main()
