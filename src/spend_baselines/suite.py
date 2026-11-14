"""Run the explicit non-boosting study: plan, prepare, run, combine, refit.

python -m spend_baselines.suite plan --config configs/full_suite.json
"""

import argparse
import fcntl
import hashlib
import itertools
import json
import platform
import time
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from .suite_data import (
    BASE_COLUMNS,
    FEATURES,
    SEQUENCES,
    atomic_json,
    digest,
    prepare_cache,
    row_weights,
    split_indices,
)
from .suite_models import Neural, Tabular


def job_key(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]


def gate_objective(theta, features, predictions, target):
    weights = softmax(
        features @ theta.reshape(features.shape[1], predictions.shape[1]), axis=1
    )
    error = (weights * predictions).sum(1) - target
    return np.mean(error**2) + 0.01 * np.mean(theta**2)


def read_config(path, smoke=False):
    config = json.loads(Path(path).read_text())
    if config["version"] != 1 or not config["validation_dates"]:
        raise ValueError("Unsupported/empty suite configuration")
    for key, allowed in (
        ("datasets", {"D0", "D1", "D2", "D3"}),
        ("tabular_features", set(FEATURES)),
        ("sequence_features", set(SEQUENCES)),
    ):
        if len(set(config[key])) != len(config[key]) or not set(config[key]) <= allowed:
            raise ValueError(f"Invalid {key}")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Duplicate seeds")
    if smoke:
        config = deepcopy(config)
        config.update(
            datasets=["D0"],
            tabular_features=["F0", "F2"],
            sequence_features=["S0_F0"],
            smoke_customers=64,
            validation_dates=["2018-09-01", "2018-10-01"],
            seeds=[42],
        )
        config["neural"].update(max_epochs=2, patience=1, batch_size=64)
        config["profiles"] = [
            {
                "name": "compact_l2",
                "hidden_sizes": [8, 4],
                "weight_decay": 0.0001,
                "dropout": 0.0,
            }
        ]
        config["tabular"]["forest"].update(
            n_estimators=3, max_depth=3, min_samples_leaf=2
        )
        config["bagging"].update(
            members=2, prefixes=[1, 2], features=["F0"], datasets=["D0"]
        )
        config["clv"].update(draws=256)
    return config


def make_plan(config):
    jobs = []
    for date, dataset, representation in itertools.product(
        config["validation_dates"], config["datasets"], config["tabular_features"]
    ):
        for family, task in itertools.product(
            ("linear", "spline", "forest"), ("direct", "probability", "positive")
        ):
            jobs.append(
                {
                    "date": date,
                    "dataset": dataset,
                    "representation": representation,
                    "kind": "tabular",
                    "family": family,
                    "task": task,
                    "seed": 42,
                }
            )
        for profile, seed in itertools.product(config["profiles"], config["seeds"]):
            for architecture, task in [
                ("mlp", "direct"),
                ("wide", "direct"),
                ("residual", "direct"),
                ("mlp", "probability"),
                ("mlp", "positive"),
                ("mlp", "hurdle"),
                ("mlp", "hurdle_joint"),
                ("mlp", "ziln"),
            ]:
                jobs.append(
                    {
                        "date": date,
                        "dataset": dataset,
                        "representation": representation,
                        "kind": "neural",
                        "family": architecture,
                        "task": task,
                        "seed": seed,
                        "profile": profile["name"],
                    }
                )
    for (
        date,
        dataset,
        representation,
        profile,
        seed,
        architecture,
        task,
    ) in itertools.product(
        config["validation_dates"],
        config["datasets"],
        config["sequence_features"],
        config["profiles"],
        config["seeds"],
        ("mlp", "tcn", "gru", "lstm"),
        ("direct", "hurdle_joint"),
    ):
        jobs.append(
            {
                "date": date,
                "dataset": dataset,
                "representation": representation,
                "kind": "neural",
                "family": architecture,
                "task": task,
                "seed": seed,
                "profile": profile["name"],
            }
        )
    bag = config["bagging"]
    for date, dataset, representation, mode, member, pair in itertools.product(
        config["validation_dates"],
        bag["datasets"],
        bag["features"],
        ("seed", "customer", "patch"),
        range(bag["members"]),
        (("mlp", "direct"), ("wide", "direct"), ("mlp", "hurdle_joint")),
    ):
        if (
            dataset not in config["datasets"]
            or representation
            not in config["tabular_features"] + config["sequence_features"]
        ):
            continue
        architecture, task = pair
        jobs.append(
            {
                "date": date,
                "dataset": dataset,
                "representation": representation,
                "kind": "bagging",
                "family": architecture,
                "task": task,
                "seed": bag["seed"] + member,
                "profile": bag["profile"],
                "mode": mode,
                "member": member,
            }
        )
    if config["clv"]["enabled"]:
        for date, window, family in itertools.product(
            config["validation_dates"], ("180", "all"), ("bg", "pareto")
        ):
            jobs.append(
                {
                    "date": date,
                    "kind": "clv",
                    "family": family,
                    "window": window,
                    "seed": 42,
                }
            )
    keys = [job_key(j) for j in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate planned jobs")
    return jobs


def identity(config, data_dir):
    return {
        "config": config,
        "source_sha256": {
            p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "input_sha256": {
            n: digest(Path(data_dir) / n)
            for n in (
                "transactions.csv",
                "train.csv",
                "test.csv",
                "sample_submission.csv",
            )
        },
    }


def existing_result(directory, job):
    marker = directory / "result.json"
    if not marker.exists():
        return None
    result = json.loads(marker.read_text())
    if result["job"] != job:
        raise ValueError(f"Job specification mismatch in {directory}")
    for name, expected in result["sha256"].items():
        if not (directory / name).exists() or digest(directory / name) != expected:
            raise ValueError(f"Missing/corrupt completed artifact: {directory / name}")
    return result


def metric(y, z, p=None):
    if len(y) != len(z) or not np.isfinite(z).all() or (z < 0).any():
        raise ValueError("Invalid predictions")
    error = (np.log1p(y) - z) ** 2
    result = {
        "n": len(y),
        "squared_log_error": float(error.sum()),
        "RMSLE": float(np.sqrt(error.mean())),
    }
    for name, mask in (("zero", y == 0), ("positive", y > 0)):
        result[f"n_{name}"] = int(mask.sum())
        result[f"RMSLE_{name}"] = (
            float(np.sqrt(error[mask].mean())) if mask.any() else None
        )
    if p is not None:
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid probabilities")
        from sklearn.metrics import roc_auc_score

        result["Brier"] = float(np.mean((p - (y > 0)) ** 2))
        result["AUC"] = (
            float(roc_auc_score(y > 0, p)) if len(np.unique(y > 0)) == 2 else None
        )
    return result


def diagnostics(directory, cache, indices, z, probability, task):
    target = cache.objects.iloc[indices].spend_30d.to_numpy(float)
    groups = []
    for column, edges, labels in (
        ("n_180", [0, 1, 2, 3, 5, 10, np.inf], ["1", "2", "3", "4-5", "6-10", "11+"]),
        (
            "recency",
            [-1, 7, 30, 60, 90, np.inf],
            ["0-7", "8-30", "31-60", "61-90", "91+"],
        ),
    ):
        values = np.expm1(cache.features[indices, BASE_COLUMNS.index(column)])
        values = np.rint(
            values
        )  # Float32 log/exp must not move integer bin boundaries.
        buckets = pd.cut(values, edges, labels=labels)
        for label in labels:
            mask = np.asarray(buckets == label)
            if task == "positive":
                mask &= target > 0
            if not mask.any():
                continue
            scores = metric(
                target[mask],
                z[mask],
                None if probability is None else probability[mask],
            )
            if task == "probability":
                for key in (
                    "RMSLE",
                    "RMSLE_zero",
                    "RMSLE_positive",
                    "squared_log_error",
                ):
                    scores[key] = None
            groups.append({"feature": column, "group": label, **scores})
    pd.DataFrame(groups).to_csv(directory / "segments.csv", index=False)
    if probability is not None:
        bins = np.minimum((probability * 10).astype(int), 9)
        rows = [
            {
                "bin": i,
                "n": int((bins == i).sum()),
                "mean_probability": float(probability[bins == i].mean()),
                "purchase_rate": float((target[bins == i] > 0).mean()),
            }
            for i in range(10)
            if (bins == i).any()
        ]
        pd.DataFrame(rows).to_csv(directory / "calibration.csv", index=False)


def bootstrap_weights(objects, indices, dataset, seed, enabled):
    bootstrap = None
    if enabled:
        customers = np.sort(objects.iloc[indices].customer_id.unique())
        draw = np.random.default_rng(seed).choice(
            customers, size=len(customers), replace=True
        )
        bootstrap = dict(Counter(draw))
    return row_weights(objects, indices, dataset, bootstrap)


def patch_columns(representation, n, shape, seed):
    rng = np.random.default_rng(seed)
    if representation in FEATURES:
        columns = FEATURES[representation]

        def block(c):
            if c in {"recency", "tenure", "month_sin", "month_cos"}:
                return "always"
            if c.startswith("region_"):
                return "region"
            if c.startswith(("bg_", "pareto_")):
                return "clv"
            if c in FEATURES["F0"]:
                return "rfm"
            if c in FEATURES["F1"]:
                return "activity"
            if c in FEATURES["F2"]:
                return "basket"
            return "rhythm"

        groups = np.array([block(c) for c in columns])
    else:
        tab, channels = SEQUENCES[representation]
        tabn = 0 if tab is None else len(FEATURES[tab])
        groups = np.array(
            ["tabular"] * tabn + [f"channel_{c}" for _ in range(26) for c in channels]
        )
    candidates = sorted(set(groups) - {"always", "channel_6"})
    selected = [g for g in candidates if rng.random() < 0.7]
    if not selected:
        selected = [candidates[0]]
    return np.flatnonzero(np.isin(groups, selected + ["always", "channel_6"]))


def execute_job(job, config, cache, args):
    directory = args.out_dir / "jobs" / job_key(job)
    old = existing_result(directory, job) if args.resume else None
    if old is not None:
        return old
    if (directory / "result.json").exists():
        raise FileExistsError(f"Completed job exists; use --resume: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    objects = cache.objects
    valid = np.flatnonzero(
        (
            objects.as_of_date.eq(pd.Timestamp(job["date"]))
            & (objects.is_test if job.get("test") else objects.official)
        ).to_numpy()
    )
    if len(valid) == 0:
        raise ValueError(f"No evaluation rows for {job['date']}")
    metadata = {}
    probability = None
    if job["kind"] == "clv":
        from .suite_clv import CustomerModel, OrderHistory

        history = OrderHistory(args.data_dir)
        summary = history.summary(
            job["date"], job["window"], objects.customer_id.unique()
        )
        model = CustomerModel(job["family"], config["clv"]).fit(summary)
        z, mean, se = model.predict(
            summary.loc[objects.iloc[valid].customer_id],
            seed=job["seed"],
            draws=config["clv"]["draws"],
        )
        metadata = {
            "mc_log_standard_error_mean": float(se.mean()),
            "mc_log_standard_error_max": float(se.max()),
            "state": model.state(),
        }
        atomic_json(directory / "model.json", model.state())
        np.savez_compressed(
            directory / "diagnostics.npz", money_mean=mean, mc_log_standard_error=se
        )
    else:
        dataset, representation = job["dataset"], job["representation"]
        outer = split_indices(objects, dataset, job["date"])
        x, shape = cache.matrix(representation, outer)
        vx, _ = cache.matrix(representation, valid)
        y = objects.iloc[outer].spend_30d.to_numpy(float)
        boot = job.get("mode") in {"customer", "patch"}
        weights = bootstrap_weights(objects, outer, dataset, job["seed"], boot)
        columns = np.arange(x.shape[1])
        if job.get("mode") == "patch":
            columns = patch_columns(representation, x.shape[1], shape, job["seed"])
            x, vx = x[:, columns], vx[:, columns]
            shape = None  # Bagging uses dense encoders; retain exact column map.
        if job["kind"] == "tabular":
            model = Tabular(
                job["family"], job["task"], config["tabular"], job["seed"], args.threads
            ).fit(x, y, weights)
            import joblib

            joblib.dump(model, directory / "model.joblib", compress=3)
        else:
            profile = next(p for p in config["profiles"] if p["name"] == job["profile"])
            settings = {
                **config["neural"],
                **{k: v for k, v in profile.items() if k != "name"},
            }
            # February's 30-day target may extend past March 1: use the latest
            # monthly check whose labels are actually available at outer time.
            available_inner = objects.official & objects.target_end.le(
                pd.Timestamp(job["date"])
            )
            inner_date = objects.loc[available_inner, "as_of_date"].max()
            inner = split_indices(objects, dataset, inner_date)
            iv = np.flatnonzero(
                (objects.as_of_date.eq(inner_date) & objects.official).to_numpy()
            )
            if not len(inner) or not len(iv) or not len(outer):
                raise ValueError("Empty temporal split")
            ix, ishape = cache.matrix(representation, inner)
            ivx, _ = cache.matrix(representation, iv)
            if job.get("mode") == "patch":
                ix, ivx = ix[:, columns], ivx[:, columns]
                ishape = None
            iw = bootstrap_weights(objects, inner, dataset, job["seed"], boot)
            selector = Neural(
                job["family"],
                job["task"],
                settings,
                job["seed"],
                args.device,
                args.threads,
                ishape,
            )
            selector.fit(
                ix,
                objects.iloc[inner].spend_30d.to_numpy(float),
                iw,
                validation=(ivx, objects.iloc[iv].spend_30d.to_numpy(float)),
            )
            chosen = selector.best_epoch
            pd.DataFrame(selector.history).to_csv(
                directory / "selection.csv", index=False
            )
            del selector, ix, ivx
            metadata.update(
                inner_date=str(inner_date.date()),
                selected_epoch=chosen,
                n_inner=len(inner),
                n_inner_valid=len(iv),
            )
            model = Neural(
                job["family"],
                job["task"],
                settings,
                job["seed"],
                args.device,
                args.threads,
                shape,
            )
            model.fit(x, y, weights, epochs=chosen)
            pd.DataFrame(model.history).to_csv(directory / "refit.csv", index=False)
            model.save(directory / "model.pt")
        prediction = model.predict(vx)
        if job["task"] == "probability":
            probability = prediction
            # This component is scored by Brier/AUC; zero is only its placeholder
            # in the common container, never interpreted as a spend candidate.
            z = np.zeros(len(prediction))
        else:
            z = prediction
            if job["task"].startswith("hurdle") or job["task"] == "ziln":
                probability = model.predict_probability(vx)
        metadata.update(
            n_train=len(outer),
            weight_sum=float(weights.sum()),
            input_columns=columns.tolist(),
        )
    np.savez_compressed(
        directory / "predictions.npz",
        Id=objects.iloc[valid].Id.to_numpy(str),
        customer_id=objects.iloc[valid].customer_id.to_numpy(str),
        z=z,
        probability=np.array([]) if probability is None else probability,
    )
    if job.get("test"):
        scores = {"n": len(valid)}
        if job.get("task") not in {"probability", "positive"}:
            from .cli import write_submission

            sample = pd.read_csv(
                args.data_dir / "sample_submission.csv", dtype={"Id": "string"}
            )
            sample = sample[sample.Id.isin(objects.iloc[valid].Id)]
            write_submission(
                objects.iloc[valid], z, directory / "submission.csv", sample.Id
            )
    else:
        scores = metric(objects.iloc[valid].spend_30d.to_numpy(float), z, probability)
        if job.get("task") == "probability":
            scores.update(
                RMSLE=None, RMSLE_zero=None, RMSLE_positive=None, squared_log_error=None
            )
        elif job.get("task") == "positive":
            target = objects.iloc[valid].spend_30d.to_numpy(float)
            mask = target > 0
            scores = metric(target[mask], z[mask])
            scores["n_validation_total"] = len(valid)
        diagnostics(directory, cache, valid, z, probability, job.get("task", "direct"))
    result = {"job": job, **scores, **metadata, "seconds": time.monotonic() - start}
    result["sha256"] = {
        p.name: digest(p)
        for p in directory.iterdir()
        if p.is_file()
        and p.name not in {"result.json", "error.json"}
        and not p.name.endswith(".tmp")
    }
    atomic_json(directory / "result.json", result)
    (directory / "error.json").unlink(missing_ok=True)
    return result


def write_scores(results, directory):
    rows = []
    for r in results:
        if r["job"].get("test"):
            continue
        row = {
            **r["job"],
            **{
                k: r.get(k)
                for k in (
                    "n",
                    "RMSLE",
                    "RMSLE_zero",
                    "RMSLE_positive",
                    "Brier",
                    "AUC",
                    "seconds",
                    "squared_log_error",
                )
            },
        }
        row["job_id"] = job_key(r["job"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "scores.csv", index=False)
    if rows:
        keys = [
            c
            for c in (
                "kind",
                "dataset",
                "representation",
                "family",
                "task",
                "profile",
                "seed",
                "mode",
                "member",
                "window",
            )
            if c in frame
        ]
        spend = frame[
            frame.get("task", pd.Series(index=frame.index, dtype=str)).ne("probability")
        ]
        pooled = (
            spend.groupby(keys, dropna=False)
            .agg(
                n=("n", "sum"),
                sum_log_squared_error=("squared_log_error", "sum"),
                folds=("date", "nunique"),
            )
            .reset_index()
        )
        pooled["RMSLE"] = np.sqrt(pooled.sum_log_squared_error / pooled.n)
        pooled.to_csv(directory / "pooled_scores.csv", index=False)


def load_prediction(directory, result):
    with np.load(
        directory / "jobs" / job_key(result["job"]) / "predictions.npz",
        allow_pickle=False,
    ) as data:
        return {k: data[k] for k in data.files}


def aligned(predictions):
    first = predictions[0]["Id"]
    for p in predictions[1:]:
        if not np.array_equal(first, p["Id"]):
            raise ValueError("Prediction identifiers/order do not match")
    return first


def combine(results, config, cache, directory):
    """Predeclared recombinations; no best-score/seed selection."""
    derived = []
    groups = defaultdict(list)
    objects = cache.objects.set_index("Id")
    for result in results:
        j = result["job"]
        if j["kind"] in {"neural", "tabular"}:
            groups[
                (
                    j["date"],
                    j["dataset"],
                    j["representation"],
                    j["family"],
                    j["task"],
                    j.get("profile", "fixed"),
                )
            ].append(result)
    values = {}

    def save(name, date, dataset, representation, predictions, z, prob=None):
        ids = aligned(predictions)
        selected = objects.loc[ids]
        info = {
            "name": name,
            "date": date,
            "dataset": dataset,
            "representation": representation,
            **metric(selected.spend_30d.to_numpy(float), z, prob),
        }
        key = job_key(
            {k: info[k] for k in ("name", "date", "dataset", "representation")}
        )
        dest = directory / "derived"
        dest.mkdir(exist_ok=True)
        np.savez_compressed(
            dest / f"{key}.npz",
            Id=ids,
            z=z,
            probability=np.array([]) if prob is None else prob,
        )
        derived.append(info)
        return {
            "Id": ids,
            "z": z,
            "probability": np.array([]) if prob is None else prob,
        }

    for key, items in groups.items():
        date, ds, rep, family, task, profile = key
        required = 1 if items[0]["job"]["kind"] == "tabular" else len(config["seeds"])
        if len(items) != required:
            continue
        ps = [load_prediction(directory, r) for r in items]
        aligned(ps)
        name = f"{family}:{task}:{profile}"
        prediction = np.mean(
            [p["probability"] if task == "probability" else p["z"] for p in ps], axis=0
        )
        values[(date, ds, rep, family, task, profile)] = {
            "Id": ps[0]["Id"],
            "z": prediction,
        }
        if task not in {"probability", "positive"}:
            save(name, date, ds, rep, ps, prediction)
    component_profile = config["component_profile"]
    for date, ds, rep in itertools.product(
        config["validation_dates"], config["datasets"], config["tabular_features"]
    ):
        for classifier, regressor in itertools.product(
            ("linear", "spline", "forest", "mlp"), repeat=2
        ):
            p = values.get(
                (
                    date,
                    ds,
                    rep,
                    classifier,
                    "probability",
                    component_profile if classifier == "mlp" else "fixed",
                )
            )
            m = values.get(
                (
                    date,
                    ds,
                    rep,
                    regressor,
                    "positive",
                    component_profile if regressor == "mlp" else "fixed",
                )
            )
            if p is None or m is None:
                continue
            save(
                f"hurdle:{classifier}+{regressor}",
                date,
                ds,
                rep,
                [p, m],
                p["z"] * m["z"],
                p["z"],
            )
            if classifier == regressor == "linear":
                save(
                    "hard_gate_0.5:linear",
                    date,
                    ds,
                    rep,
                    [p, m],
                    (p["z"] >= 0.5) * m["z"],
                    p["z"],
                )
        members = []
        for family, task, profile in [
            ("linear", "direct", "fixed"),
            ("forest", "direct", "fixed"),
            ("mlp", "direct", component_profile),
        ]:
            p = values.get((date, ds, rep, family, task, profile))
            if p is not None:
                members.append(p)
        if len(members) == 3:
            save(
                "equal_linear_forest_mlp",
                date,
                ds,
                rep,
                members,
                np.mean([p["z"] for p in members], axis=0),
            )
            save(
                "equal_linear_forest",
                date,
                ds,
                rep,
                members[:2],
                (members[0]["z"] + members[1]["z"]) / 2,
            )
    bags = defaultdict(list)
    for r in results:
        j = r["job"]
        if j["kind"] == "bagging":
            bags[
                tuple(
                    j[k]
                    for k in (
                        "date",
                        "dataset",
                        "representation",
                        "family",
                        "task",
                        "mode",
                    )
                )
            ].append(r)
    for (date, ds, rep, family, task, mode), items in bags.items():
        items.sort(key=lambda r: r["job"]["member"])
        for count in config["bagging"]["prefixes"]:
            chosen = [r for r in items if r["job"]["member"] < count]
            if len(chosen) != count:
                continue
            ps = [load_prediction(directory, r) for r in chosen]
            save(
                f"ensemble:{mode}:{family}:{task}:n{count}",
                date,
                ds,
                rep,
                ps,
                np.mean([p["z"] for p in ps], axis=0),
            )
    # Learned combinations train only on earlier OOT predictions with available labels.
    from sklearn.linear_model import Ridge

    for ds, rep in itertools.product(config["datasets"], config["tabular_features"]):
        past_x = []
        past_y = []
        past_g = []
        past_end = []
        for date in sorted(config["validation_dates"]):
            ps = [
                values.get((date, ds, rep, f, t, p))
                for f, t, p in [
                    ("linear", "direct", "fixed"),
                    ("forest", "direct", "fixed"),
                    ("mlp", "direct", component_profile),
                ]
            ]
            if any(p is None for p in ps):
                continue
            ids = aligned(ps)
            x = np.stack([p["z"] for p in ps], axis=1)
            rows = objects.loc[ids]
            ix = cache.objects.index[cache.objects.Id.isin(ids)].to_numpy()
            if not np.array_equal(cache.objects.iloc[ix].Id.to_numpy(str), ids):
                lookup = pd.Series(cache.objects.index, index=cache.objects.Id)
                ix = lookup.loc[ids].to_numpy()
            g = np.asarray(
                cache.features[
                    np.ix_(
                        ix,
                        [BASE_COLUMNS.index(c) for c in ("recency", "n_180", "tenure")],
                    )
                ],
                dtype=float,
            )
            if past_x:
                px = np.concatenate(past_x)
                py = np.concatenate(past_y)
                pg = np.concatenate(past_g)
                pe = np.concatenate(past_end)
                eligible = pe <= np.datetime64(date)
                if eligible.any():
                    linear = Ridge(alpha=10).fit(px[eligible], py[eligible])
                    save(
                        "oot_stack_ridge",
                        date,
                        ds,
                        rep,
                        ps,
                        np.maximum(linear.predict(x), 0),
                    )
                    mean, scale = pg[eligible].mean(0), pg[eligible].std(0).clip(0.01)
                    train_g = np.c_[
                        np.ones(eligible.sum()), (pg[eligible] - mean) / scale
                    ]
                    gate = np.c_[np.ones(len(g)), (g - mean) / scale]

                    opt = minimize(
                        gate_objective,
                        np.zeros(12),
                        args=(train_g, px[eligible], py[eligible]),
                        method="L-BFGS-B",
                        options={"maxiter": 500},
                    )
                    if not opt.success:
                        raise RuntimeError(f"Router failed: {opt.message}")
                    prediction = (softmax(gate @ opt.x.reshape(4, 3), axis=1) * x).sum(
                        1
                    )
                    save("oot_soft_router", date, ds, rep, ps, prediction)
            past_x.append(x)
            past_y.append(np.log1p(rows.spend_30d.to_numpy(float)))
            past_g.append(g)
            past_end.append(rows.target_end.to_numpy())
    frame = pd.DataFrame(derived)
    frame.to_csv(directory / "derived_scores.csv", index=False)
    if not frame.empty:
        pooled = (
            frame.groupby(["name", "dataset", "representation"])
            .agg(
                n=("n", "sum"),
                sum_log_squared_error=("squared_log_error", "sum"),
                folds=("date", "nunique"),
            )
            .reset_index()
        )
        pooled["RMSLE"] = np.sqrt(pooled.sum_log_squared_error / pooled.n)
        pooled.to_csv(directory / "derived_pooled.csv", index=False)
    return len(derived)


def refit_candidate(name, dataset, representation, config, plan, cache, args):
    """Refit an explicitly named derived candidate without selecting a winner."""
    from sklearn.linear_model import Ridge

    from .cli import write_submission

    objects = cache.objects
    cutoff = str(objects.loc[objects.is_test, "as_of_date"].min().date())
    anchor = max(config["validation_dates"])
    component = config["component_profile"]
    groups = []

    def add_base(family, task, profile=None):
        profile = profile or (
            component if family not in {"linear", "spline", "forest"} else "fixed"
        )
        found = [
            j
            for j in plan
            if j["date"] == anchor
            and j.get("dataset") == dataset
            and j.get("representation") == representation
            and j["kind"] in {"tabular", "neural"}
            and j["family"] == family
            and j["task"] == task
            and j.get("profile", "fixed") == profile
        ]
        expected = (
            1 if family in {"linear", "spline", "forest"} else len(config["seeds"])
        )
        if len(found) != expected:
            raise ValueError(f"Unavailable component: {family}/{task}/{profile}")
        groups.append(found)

    if name.startswith("hurdle:"):
        classifier, regressor = name.split(":", 1)[1].split("+")
        add_base(classifier, "probability")
        add_base(regressor, "positive")
    elif name == "hard_gate_0.5:linear":
        add_base("linear", "probability")
        add_base("linear", "positive")
    elif name in {
        "equal_linear_forest",
        "equal_linear_forest_mlp",
        "oot_stack_ridge",
        "oot_soft_router",
    }:
        add_base("linear", "direct")
        add_base("forest", "direct")
        if name != "equal_linear_forest":
            add_base("mlp", "direct")
    elif name.startswith("ensemble:"):
        _, mode, family, task, count = name.split(":")
        count = int(count.removeprefix("n"))
        if count not in config["bagging"]["prefixes"]:
            raise ValueError("Unplanned ensemble prefix")
        found = [
            j
            for j in plan
            if j["date"] == anchor
            and j.get("dataset") == dataset
            and j.get("representation") == representation
            and j["kind"] == "bagging"
            and j["family"] == family
            and j["task"] == task
            and j["mode"] == mode
            and j["member"] < count
        ]
        if len(found) != count:
            raise ValueError("Incomplete ensemble definition")
        groups.append(found)
    else:
        family, task, profile = name.split(":")
        if task in {"probability", "positive"}:
            raise ValueError("A component is not a spend candidate")
        add_base(family, task, profile)
    key = job_key(
        {
            "candidate": name,
            "dataset": dataset,
            "representation": representation,
            "date": cutoff,
        }
    )
    dest = args.out_dir / "candidates" / key
    dest.mkdir(parents=True, exist_ok=True)
    predictions = []
    members = []
    for group in groups:
        pp = []
        for template in group:
            job = {**template, "date": cutoff, "test": True}
            result = execute_job(job, config, cache, args)
            members.append(job_key(job))
            pp.append(load_prediction(args.out_dir, result))
        ids = aligned(pp)
        field = "probability" if group[0]["task"] == "probability" else "z"
        predictions.append({"Id": ids, "z": np.mean([p[field] for p in pp], axis=0)})
    ids = aligned(predictions)
    if name.startswith("hurdle:"):
        z = predictions[0]["z"] * predictions[1]["z"]
    elif name.startswith("hard_gate"):
        z = (predictions[0]["z"] >= 0.5) * predictions[1]["z"]
    elif name in {"oot_stack_ridge", "oot_soft_router"}:
        history_x = []
        history_y = []
        history_g = []
        lookup = pd.Series(objects.index, index=objects.Id)
        router_columns = [BASE_COLUMNS.index(c) for c in ("recency", "n_180", "tenure")]
        for date in sorted(config["validation_dates"]):
            pp = []
            for group in groups:
                rows = []
                for template in group:
                    job = {**template, "date": date}
                    result = existing_result(args.out_dir / "jobs" / job_key(job), job)
                    if result is None:
                        raise ValueError(
                            "Meta refit requires all earlier OOT component predictions"
                        )
                    rows.append(load_prediction(args.out_dir, result))
                aligned(rows)
                pp.append(
                    {"Id": rows[0]["Id"], "z": np.mean([p["z"] for p in rows], axis=0)}
                )
            historic_ids = aligned(pp)
            ix = lookup.loc[historic_ids].to_numpy()
            eligible = objects.iloc[ix].target_end.le(cutoff).to_numpy()
            history_x.append(np.stack([p["z"] for p in pp], axis=1)[eligible])
            history_y.append(
                np.log1p(objects.iloc[ix].spend_30d.to_numpy(float)[eligible])
            )
            history_g.append(
                np.asarray(cache.features[np.ix_(ix, router_columns)])[eligible]
            )
        px, py, pg = (
            np.concatenate(history_x),
            np.concatenate(history_y),
            np.concatenate(history_g),
        )
        x = np.stack([p["z"] for p in predictions], axis=1)
        if name == "oot_stack_ridge":
            model = Ridge(alpha=10).fit(px, py)
            z = np.maximum(model.predict(x), 0)
            atomic_json(
                dest / "combiner.json",
                {
                    "coefficients": model.coef_.tolist(),
                    "intercept": float(model.intercept_),
                },
            )
        else:
            mean, scale = pg.mean(0), pg.std(0).clip(0.01)
            train_g = np.c_[np.ones(len(pg)), (pg - mean) / scale]
            ix = lookup.loc[ids].to_numpy()
            g = np.asarray(cache.features[np.ix_(ix, router_columns)])
            g = np.c_[np.ones(len(g)), (g - mean) / scale]

            def objective(theta):
                error = (softmax(train_g @ theta.reshape(4, 3), axis=1) * px).sum(
                    1
                ) - py
                return np.mean(error**2) + 0.01 * np.mean(theta**2)

            opt = minimize(
                objective, np.zeros(12), method="L-BFGS-B", options={"maxiter": 500}
            )
            if not opt.success:
                raise RuntimeError(opt.message)
            z = (softmax(g @ opt.x.reshape(4, 3), axis=1) * x).sum(1)
            atomic_json(
                dest / "combiner.json",
                {
                    "theta": opt.x.tolist(),
                    "mean": mean.tolist(),
                    "scale": scale.tolist(),
                },
            )
    else:
        z = np.mean([p["z"] for p in predictions], axis=0)
    test = objects.set_index("Id").loc[ids].reset_index()
    sample = pd.read_csv(
        args.data_dir / "sample_submission.csv", dtype={"Id": "string"}
    )
    sample = sample[sample.Id.isin(test.Id)]
    write_submission(test, z, dest / "submission.csv", sample.Id)
    np.savez_compressed(dest / "predictions.npz", Id=ids, z=z)
    atomic_json(
        dest / "result.json",
        {
            "candidate": name,
            "dataset": dataset,
            "representation": representation,
            "date": cutoff,
            "members": members,
            "sha256": {
                p.name: digest(p)
                for p in dest.iterdir()
                if p.is_file() and p.name != "result.json"
            },
        },
    )
    print(f"Wrote {dest / 'submission.csv'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["plan", "prepare", "run", "combine", "refit"]
    )
    parser.add_argument("--config", type=Path, default=Path("configs/full_suite.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/full-suite"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="All families, small target-independent customer subset, 2 epochs; separate output directory",
    )
    parser.add_argument(
        "--kinds", nargs="+", choices=["tabular", "neural", "bagging", "clv"]
    )
    parser.add_argument("--datasets", nargs="+", choices=["D0", "D1", "D2", "D3"])
    parser.add_argument("--representations", nargs="+")
    parser.add_argument("--dates", nargs="+")
    parser.add_argument(
        "--limit", type=int, help="Explicitly run only a prefix for diagnostics"
    )
    parser.add_argument(
        "--job-ids",
        nargs="+",
        help="Explicit candidates for refit; no automatic leaderboard-based selection",
    )
    parser.add_argument(
        "--candidate",
        help="Explicit derived name, e.g. hurdle:linear+forest or oot_soft_router",
    )
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Positive threads/limit required")
    config = read_config(args.config, args.smoke)
    plan = make_plan(config)
    counts = Counter(j["kind"] for j in plan)
    print(
        json.dumps(
            {
                "jobs": len(plan),
                "by_kind": counts,
                "neural_fit_calls": 2 * (counts["neural"] + counts["bagging"]),
                "dates": config["validation_dates"],
                "smoke": args.smoke,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.command == "plan":
        return 0
    selected = [
        j
        for j in plan
        if (not args.kinds or j["kind"] in args.kinds)
        and (not args.datasets or j.get("dataset") in args.datasets)
        and (
            not args.representations or j.get("representation") in args.representations
        )
        and (not args.dates or j["date"] in args.dates)
    ]
    if args.limit:
        selected = selected[: args.limit]
    if args.command == "run" and not selected:
        parser.error("Filters selected no planned jobs")
    if args.command in {"run", "refit"} and args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; use --device cpu only for a local smoke check"
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.out_dir / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    info = identity(config, args.data_dir)
    cache = prepare_cache(args.data_dir, args.out_dir / "cache", config, info)
    needs_f4 = args.command == "prepare"
    if args.command == "run":
        needs_f4 = any(j.get("representation") == "F4" for j in selected)
    elif args.command == "refit":
        needs_f4 = bool(
            args.candidate and args.representations and "F4" in args.representations
        )
        needs_f4 |= any(
            job_key(j) in (args.job_ids or []) and j.get("representation") == "F4"
            for j in plan
        )
    if "F4" in config["tabular_features"] and needs_f4:
        from .suite_clv import prepare_clv

        prepare_clv(cache, args.data_dir, config["clv"], info)
    if args.command == "prepare":
        return 0
    runtime = {
        "definition": info,
        "device": args.device,
        "threads": args.threads,
        "python": platform.python_version(),
        "libraries": {
            n: version(n) for n in ("numpy", "pandas", "scikit-learn", "scipy", "torch")
        },
    }
    marker = args.out_dir / "run_manifest.json"
    if marker.exists():
        if json.loads(marker.read_text()) != runtime:
            raise ValueError(
                "Run code/data/config/runtime changed; use a new --out-dir"
            )
        if args.command == "run" and not args.resume:
            raise ValueError("Existing run requires --resume")
    else:
        atomic_json(marker, runtime)
    atomic_json(args.out_dir / "plan.json", plan)
    if args.command == "refit":
        if args.candidate:
            if (
                args.job_ids
                or not args.datasets
                or len(args.datasets) != 1
                or not args.representations
                or len(args.representations) != 1
            ):
                parser.error(
                    "--candidate requires exactly one --datasets and --representations, without --job-ids"
                )
            refit_candidate(
                args.candidate,
                args.datasets[0],
                args.representations[0],
                config,
                plan,
                cache,
                args,
            )
            return 0
        if not args.job_ids:
            parser.error("refit requires explicit --job-ids")
        chosen = [j for j in plan if job_key(j) in args.job_ids]
        if len(chosen) != len(set(args.job_ids)):
            raise ValueError("Unknown/duplicate refit job IDs")
        if any(j.get("task") in {"probability", "positive"} for j in chosen):
            raise ValueError(
                "Refit standalone spend predictors; components require their paired hurdle candidate"
            )
        test_date = str(
            cache.objects.loc[cache.objects.is_test, "as_of_date"].min().date()
        )
        for job in chosen:
            j = {**job, "date": test_date, "test": True}
            execute_job(j, config, cache, args)
        return 0
    failures = []
    running_results = {}
    if args.command == "run":
        for i, job in enumerate(selected):
            print(
                f"[{i + 1}/{len(selected)}] {job_key(job)} {json.dumps(job, sort_keys=True)}",
                flush=True,
            )
            try:
                result = execute_job(job, config, cache, args)
                running_results[job_key(job)] = result
                score = (
                    result.get("Brier")
                    if job.get("task") == "probability"
                    else result.get("RMSLE")
                )
                print(f"  complete: metric={score}", flush=True)
            except Exception as error:
                directory = args.out_dir / "jobs" / job_key(job)
                directory.mkdir(exist_ok=True, parents=True)
                atomic_json(
                    directory / "error.json",
                    {
                        "job": job,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                failures.append(job_key(job))
                print(f"  FAILED: {error}", flush=True)
                if not config.get("continue_on_error", False):
                    raise
            if (i + 1) % config.get("progress_every", 25) == 0:
                write_scores(list(running_results.values()), args.out_dir)
                atomic_json(
                    args.out_dir / "status.json",
                    {
                        "planned": len(plan),
                        "completed_in_this_selection": len(running_results),
                        "selected": len(selected),
                        "visited": i + 1,
                        "failed_in_this_invocation": failures,
                    },
                )
    results = []
    for job in plan:
        result = existing_result(args.out_dir / "jobs" / job_key(job), job)
        if result is not None:
            results.append(result)
    write_scores(results, args.out_dir)
    nderived = combine(results, config, cache, args.out_dir)
    atomic_json(
        args.out_dir / "status.json",
        {
            "planned": len(plan),
            "completed": len(results),
            "remaining": len(plan) - len(results),
            "failed_in_this_invocation": failures,
            "derived": nderived,
            "selected_in_this_invocation": len(selected),
        },
    )
    print(
        f"Completed {len(results)}/{len(plan)} jobs; {nderived} derived evaluations; failures={len(failures)}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
