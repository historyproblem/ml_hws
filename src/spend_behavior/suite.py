"""Bounded CPU screening followed by at most two explicitly chosen Joint inputs.

No leaderboard access, submission writer, architecture grid or external data.
"""

import argparse
import fcntl
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from spend_baselines.suite_data import atomic_json, digest, row_weights, split_indices
from spend_baselines.suite_models import Neural

from .data import BLOCKS, COLUMNS, prepare
from .models import (
    DiscreteHazard, SegmentRidge, WeibullAFT, causal_histories,
    chronological_residuals, inner_date, ridge, segment_matrix,
    survival_observations, validation_rows,
)
from .report import Reference, write_report


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["version"] != 1 or cfg["dataset"] != "D2":
        raise ValueError("Behaviour suite requires version 1 / D2")
    if len(cfg["joint_blocks"]) > 2 or not set(cfg["joint_blocks"]) <= set(BLOCKS) - {"segment"}:
        raise ValueError("Choose at most two explicit feature directions for Joint")
    if cfg["seeds"] != [42, 123, 2026]:
        raise ValueError("Joint comparison requires the three frozen seeds")
    if not cfg["validation_dates"] or cfg["bootstrap_repeats"] < 1:
        raise ValueError("Empty dates/bootstrap")
    if not set(cfg["segment_k"]) <= {2, 3, 4, 5}:
        raise ValueError("Segment counts must be 2–5")
    if any(w <= 0 or w >= 1 for w in cfg["blend_weights"]):
        raise ValueError("Blend weights must be in (0,1)")
    return cfg


def candidates(config):
    return (["ridge_F2", "ridge_individual", "ridge_history", "ridge_regularity",
             "ridge_season", "ridge_season_only", "ridge_items"] +
            [f"{m}_{positive}_positive" for m in ("hazard", "weibull") for positive in ("ridge", "joint")] +
            [f"segment_{method}_{mode}" for method in ("kmeans", "gmm", "rules")
             for mode in ("feature", "hard", "soft")] +
            [f"season_shift_{strength:g}" for strength in config["season_strengths"]])


def log(message):
    print(message, flush=True)


class Context:
    def __init__(self, data, config, args):
        self.data, self.objects, self.cfg, self.args = data, data.objects, config, args
        self.models = {}
        self.effects = None

    def train(self, cutoff):
        ix = split_indices(self.objects, "D2", cutoff)
        return ix, self.objects.iloc[ix].spend_30d.to_numpy(float), row_weights(self.objects, ix, "D2")

    def fit_ridge(self, cutoff, blocks=(), task="direct", season_only=False):
        key = ("ridge", str(cutoff), tuple(blocks), task, season_only)
        if key not in self.models:
            start = time.monotonic()
            ix, y, w = self.train(cutoff)
            x = self.data.matrix(ix, blocks)
            if season_only:
                x = x[:, -len(BLOCKS["season"]):]
            model = ridge(x, y, w, self.cfg["ridge_alpha"], task)
            self.models[key] = model, time.monotonic() - start
        return self.models[key]

    def ridge_prediction(self, cutoff, valid, blocks=(), task="direct", season_only=False):
        model, seconds = self.fit_ridge(cutoff, blocks, task, season_only)
        x = self.data.matrix(valid, blocks)
        if season_only:
            x = x[:, -len(BLOCKS["season"]):]
        start = time.monotonic()
        return model.predict(x), model, seconds, time.monotonic() - start

    def fit_survival(self, cutoff, kind, parameter):
        key = (kind, str(cutoff), parameter)
        if key not in self.models:
            start = time.monotonic()
            rows, duration, event = survival_observations(self.objects, self.data.next_event, cutoff)
            w = row_weights(self.objects, rows, "D0")
            cls = DiscreteHazard if kind == "hazard" else WeibullAFT
            model = cls(parameter).fit(self.data.matrix(rows, ["regularity"]), duration, event, w)
            self.models[key] = (model, time.monotonic() - start,
                                {"n_origins": len(rows), "n_events": int(event.sum()),
                                 "n_censored": int((~event).sum()), "max_origin": str(self.objects.iloc[rows].as_of_date.max())})
        return self.models[key]

    def fit_segment(self, cutoff, method, k, mode):
        key = ("segment", str(cutoff), method, k, mode)
        if key not in self.models:
            start = time.monotonic()
            ix, y, w = self.train(cutoff)
            model = SegmentRidge(method, k, mode, self.cfg["segment_min_customers"],
                                 self.cfg["segment_shrink"], self.cfg["ridge_alpha"])
            model.fit(self.data.matrix(ix), segment_matrix(self.data, ix), y, w,
                      self.objects.iloc[ix].customer_id.to_numpy())
            self.models[key] = model, time.monotonic() - start
        return self.models[key]

    def cheap(self, name, cutoff, valid, reference=None, positive=None):
        """Inner-date model selection followed by a fresh outer-cutoff fit."""
        selection_start = time.monotonic()
        inner = inner_date(self.objects, cutoff)
        iv = validation_rows(self.objects, inner)
        iy = self.objects.iloc[iv].spend_30d.to_numpy(float)
        iz = np.log1p(iy)
        selection = []
        metadata = {"inner_date": str(inner.date())}
        model = None
        if name.startswith("ridge_"):
            block = name.removeprefix("ridge_")
            blocks = [] if block in {"F2", "individual"} else ["season" if block == "season_only" else block]
            z, model, train_seconds, infer_seconds = self.ridge_prediction(cutoff, valid, blocks, season_only=block == "season_only")
            if block == "individual":
                baseline, _, _, _ = self.ridge_prediction(inner, iv)
                for lam in self.cfg["effect_lambdas"]:
                    score = float(np.mean((np.maximum(baseline + self.effects[lam][iv], 0) - iz) ** 2))
                    selection.append({"lambda": lam, "mse": score})
                chosen = min(selection, key=lambda r: r["mse"])["lambda"]
                start = time.monotonic()
                z = np.maximum(z + self.effects[chosen][valid], 0)
                infer_seconds += time.monotonic() - start
                metadata["lambda"] = chosen
        elif name.startswith(("hazard_", "weibull_")):
            kind, component, _ = name.split("_")
            params = self.cfg["hazard_C"] if kind == "hazard" else self.cfg["weibull_penalty"]
            for p in params:
                candidate, _, _ = self.fit_survival(inner, kind, p)
                probability = candidate.predict(self.data.matrix(iv, ["regularity"]))
                selection.append({"parameter": p, "brier": float(np.mean((probability - (iy > 0)) ** 2))})
            chosen = min(selection, key=lambda r: r["brier"])["parameter"]
            model, train_seconds, survival_meta = self.fit_survival(cutoff, kind, chosen)
            start = time.monotonic()
            probability = model.predict(self.data.matrix(valid, ["regularity"]))
            infer_seconds = time.monotonic() - start
            if component == "ridge":
                mu, positive_model, pt, pi = self.ridge_prediction(cutoff, valid, task="positive")
                train_seconds += pt
                infer_seconds += pi
                model = {"survival": model, "positive": positive_model}
            else:
                if positive is None:
                    raise ValueError("Joint positive head is required")
                mu = positive
            z = probability * mu
            metadata.update(parameter=chosen, **survival_meta)
        elif name.startswith("segment_"):
            _, method, mode = name.split("_")
            ks = self.cfg["segment_k"] if method == "kmeans" else [3]
            for k in ks:
                candidate, _ = self.fit_segment(inner, method, k, mode)
                pred = candidate.predict(self.data.matrix(iv), segment_matrix(self.data, iv))
                selection.append({"k": k, "mse": float(np.mean((pred - iz) ** 2))})
            chosen = min(selection, key=lambda r: r["mse"])["k"]
            model, train_seconds = self.fit_segment(cutoff, method, chosen, mode)
            start = time.monotonic()
            z = model.predict(self.data.matrix(valid), segment_matrix(self.data, valid))
            infer_seconds = time.monotonic() - start
            metadata.update(k=chosen, segments=model.summary)
        elif name.startswith("season_shift_"):
            if reference is None:
                raise ValueError("Joint reference is required")
            strength = float(name.removeprefix("season_shift_"))
            start = time.monotonic()
            shift = self.data.extra[valid, COLUMNS.index("global_season")]
            z = np.maximum(reference + strength * shift, 0)
            train_seconds, infer_seconds = 0.0, time.monotonic() - start
            metadata["strength"] = strength
        else:
            raise ValueError(f"Unknown candidate {name}")
        metadata.update(selection=selection, train_seconds=train_seconds,
                        inference_seconds=infer_seconds,
                        total_with_selection_seconds=time.monotonic() - selection_start)
        return np.asarray(z, float), metadata, model

    def joint(self, cutoff, valid, block, destination):
        """Same compact Joint model/profile, with exactly one new feature block."""
        destination.mkdir(parents=True, exist_ok=True)
        inner = inner_date(self.objects, cutoff)
        iv = validation_rows(self.objects, inner)
        ti, iy, iw = self.train(inner)
        tr, y, w = self.train(cutoff)
        settings = self.cfg["joint_settings"]
        blocks = [] if block == "baseline" else [block]
        zs, mus, epochs = [], [], []
        train_seconds, infer_seconds = 0.0, 0.0
        for seed in self.cfg["seeds"]:
            log(f"Joint + {block}, {cutoff}, seed {seed}: inner selection")
            start = time.monotonic()
            selector = Neural("mlp", "hurdle_joint", settings, seed, self.args.device, self.args.threads)
            selector.fit(self.data.matrix(ti, blocks), iy, iw,
                         validation=(self.data.matrix(iv, blocks), self.objects.iloc[iv].spend_30d.to_numpy()))
            chosen = selector.best_epoch
            pd.DataFrame(selector.history).to_csv(destination / f"selection_{seed}.csv", index=False)
            del selector
            model = Neural("mlp", "hurdle_joint", settings, seed, self.args.device, self.args.threads)
            model.fit(self.data.matrix(tr, blocks), y, w, epochs=chosen)
            model.save(destination / f"model_{seed}.pt")
            pd.DataFrame(model.history).to_csv(destination / f"refit_{seed}.csv", index=False)
            train_seconds += time.monotonic() - start
            start = time.monotonic()
            x = self.data.matrix(valid, blocks)
            z, p = model.predict(x), model.predict_probability(x)
            if (p <= 1e-10).any():
                raise ValueError("Joint probability too small for head reconstruction")
            zs.append(z)
            mus.append(z / p)
            infer_seconds += time.monotonic() - start
            epochs.append(chosen)
            del model
        return np.mean(zs, axis=0), np.mean(mus, axis=0), {
            "inner_date": str(inner.date()), "epochs": epochs,
            "train_seconds": train_seconds, "inference_seconds": infer_seconds}


def save_prediction(args, ctx, name, date, valid, z, reference, metadata, model=None):
    if not np.isfinite(z).all() or (z < 0).any():
        raise ValueError("Invalid log predictions")
    rows = ctx.objects.iloc[valid]
    frame = rows[["Id", "customer_id"]].copy()
    frame["candidate"], frame["date"] = name, str(pd.Timestamp(date).date())
    frame["target_z"] = np.log1p(rows.spend_30d)
    frame["z"], frame["reference_z"] = z, reference
    frame["delta_squared_log_error"] = (z - frame.target_z) ** 2 - (reference - frame.target_z) ** 2
    directory = args.out_dir / "predictions" / name
    results = args.out_dir / "results" / name
    directory.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    path = directory / f"{date}.csv.gz"
    frame.to_csv(path, index=False)
    if model is not None:
        model_path = results / f"{date}.joblib"
        joblib.dump(model, model_path, compress=3)
        metadata["model_sha256"] = digest(model_path)
    metadata.update(candidate=name, date=str(date), predictions_sha256=digest(path))
    atomic_json(results / f"{date}.json", metadata)
    score = np.sqrt(np.mean((z - frame.target_z.to_numpy()) ** 2))
    log(f"{date} {name}: {score:.6f} ({metadata['train_seconds']:.2f}s fit)")


def complete(args, name, date):
    path = args.out_dir / "results" / name / f"{date}.json"
    if not path.exists():
        return False
    if not args.resume:
        raise FileExistsError("Completed results exist; use --resume")
    r = json.loads(path.read_text())
    prediction = args.out_dir / "predictions" / name / f"{date}.csv.gz"
    if digest(prediction) != r["predictions_sha256"]:
        raise ValueError("Corrupt completed predictions")
    if "model_sha256" in r and digest(path.with_suffix(".joblib")) != r["model_sha256"]:
        raise ValueError("Corrupt completed model")
    if "joint_sha256" in r:
        for n, sha in r["joint_sha256"].items():
            if digest(args.out_dir / n) != sha:
                raise ValueError("Corrupt Joint checkpoint/curve")
    return True


def blend_predictions(args, ctx, name, date, valid, z, reference):
    for weight in ctx.cfg["blend_weights"]:
        blend = f"blend_{name}_{int(round(weight * 100))}pct"
        if not complete(args, blend, date):
            start = time.monotonic()
            prediction = weight * z + (1 - weight) * reference
            seconds = time.monotonic() - start
            save_prediction(args, ctx, blend, date, valid, prediction,
                            reference, {"weight_new": weight, "train_seconds": 0.0,
                                        "inference_seconds": seconds, "component": name})


def source_identity(args, config):
    paths = sorted(Path(__file__).parent.glob("*.py")) + sorted((Path(__file__).parents[1] / "spend_baselines").glob("*.py"))
    return {"config": config, "source_sha256": {f"{p.parent.name}/{p.name}": digest(p) for p in paths},
            "inputs": {n: digest(args.data_dir / n) for n in
                       ("transactions.csv", "train.csv", "test.csv", "sample_submission.csv")},
            "python": platform.python_version(), "libraries": {n: version(n) for n in
                       ("numpy", "pandas", "scikit-learn", "scipy", "joblib")},
            "device": args.device, "threads": args.threads,
            "base_cache": str(args.base_cache.resolve()) if args.base_cache else None,
            "reference": str(args.reference.resolve()),
            "reference_manifest_sha256": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "prepare", "scout", "joint", "run", "report", "refit"])
    parser.add_argument("--config", type=Path, default=Path("configs/behavior_suite.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/behavior-v1"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/v100-focused"), help="Frozen suite output directory or exported ZIP")
    parser.add_argument("--base-cache", type=Path, help="Optional verified old suite/cache, read only")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate", help="Explicit candidate for refit; no score-based winner selection")
    parser.add_argument("--smoke", action="store_true", help="64 customers, two neural epochs; separate output required")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if args.smoke:
        cfg["smoke_customers"] = 64
        cfg["bootstrap_repeats"] = 20
        cfg["joint_settings"].update(max_epochs=2, patience=1, batch_size=64)
        cfg["segment_min_customers"] = 5
    if args.command == "plan":
        print(json.dumps({"dataset": "D2", "dates": cfg["validation_dates"],
                          "CPU_candidates_per_date": candidates(cfg), "Joint_blocks": cfg["joint_blocks"],
                          "Joint_evaluations": len(cfg["joint_blocks"]) * len(cfg["seeds"]) * len(cfg["validation_dates"]),
                          "neural_fits": 2 * len(cfg["joint_blocks"]) * len(cfg["seeds"]) * len(cfg["validation_dates"]),
                          "full_suite": False, "submission": False}, indent=2))
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "runner.lock").open("w") as lock, threadpool_limits(limits=args.threads):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reference = Reference(args.reference, args.data_dir)
        identity = source_identity(args, cfg)
        import hashlib
        identity["reference_manifest_sha256"] = hashlib.sha256(reference.read("run_manifest.json")).hexdigest()
        # Same environment must be used for scout and joint to share the output.
        # Torch's version is included whenever installed, including CPU scouting.
        try:
            identity["libraries"]["torch"] = version("torch")
        except Exception:
            pass
        manifest = args.out_dir / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("Run definition changed; use a fresh output directory")
        atomic_json(manifest, identity)
        start = time.monotonic()
        data = prepare(args.data_dir, args.out_dir, cfg, args.base_cache)
        if args.command == "prepare":
            return
        ctx = Context(data, cfg, args)
        if args.command == "report":
            write_report(args.out_dir, data, cfg)
            return
        history_path = args.out_dir / "histories.npz"
        history_marker = args.out_dir / "histories.json"
        if history_marker.exists():
            if digest(history_path) != json.loads(history_marker.read_text())["sha256"]:
                raise ValueError("Corrupt OOF histories")
            with np.load(history_path, allow_pickle=False) as h:
                data.history = h["history"]
                ctx.effects = {lam: h[f"effect_{lam}"] for lam in cfg["effect_lambdas"]}
        else:
            residuals, folds = chronological_residuals(data, cfg["ridge_alpha"], progress=log)
            data.history, ctx.effects = causal_histories(data.objects, residuals, cfg["effect_lambdas"],
                                                        cfg["history_prior"], cfg["history_half_life"])
            np.savez_compressed(history_path, history=data.history, residuals=residuals,
                                **{f"effect_{lam}": value for lam, value in ctx.effects.items()})
            atomic_json(history_marker, {"sha256": digest(history_path), "folds": folds,
                                         "shared_prepare_oof_seconds": time.monotonic() - start})
        if args.command == "refit":
            refit(args, ctx)
            return
        for date in cfg["validation_dates"]:
            valid = validation_rows(data.objects, date)
            ref_objects = data.objects.iloc[valid]
            # Smoke is a subset of the full reference; validate full keys first.
            if args.smoke:
                from spend_baselines.data import read_objects
                official = read_objects(args.data_dir / "train.csv")
                all_valid = official.loc[official.as_of_date.eq(date)]
                full_z, full_mu = reference.predictions(date, all_valid)
                zz = pd.DataFrame({"Id": all_valid.Id, "z": full_z, "mu": full_mu}).set_index("Id").loc[ref_objects.Id]
                ref, positive = zz.z.to_numpy(), zz.mu.to_numpy()
            else:
                ref, positive = reference.predictions(date, ref_objects)
            if args.command in {"scout", "run"}:
                for name in candidates(cfg):
                    if complete(args, name, date):
                        frame = pd.read_csv(args.out_dir / "predictions" / name / f"{date}.csv.gz")
                        z = frame.z.to_numpy()
                    else:
                        log(f"Fitting {name} at {date}")
                        z, meta, model = ctx.cheap(name, date, valid, ref, positive)
                        save_prediction(args, ctx, name, date, valid, z, ref, meta, model)
                    if name in cfg["blend_candidates"]:
                        blend_predictions(args, ctx, name, date, valid, z, ref)
            if args.command in {"joint", "run"}:
                for block in cfg["joint_blocks"]:
                    name = f"joint_{block}_mean3"
                    if complete(args, name, date):
                        frame = pd.read_csv(args.out_dir / "predictions" / name / f"{date}.csv.gz")
                        z = frame.z.to_numpy()
                    else:
                        dest = args.out_dir / "joint_models" / block / date
                        z, _, meta = ctx.joint(date, valid, block, dest)
                        meta["joint_sha256"] = {str(p.relative_to(args.out_dir)): digest(p) for p in dest.iterdir() if p.is_file()}
                        save_prediction(args, ctx, name, date, valid, z, ref, meta)
                    blend_predictions(args, ctx, name, date, valid, z, ref)
            atomic_json(args.out_dir / "status.json", {"last_complete_date": date, "phase": args.command})
        write_report(args.out_dir, data, cfg)


def refit(args, ctx):
    """Explicit final candidate; predictions/weights only, no submission CSV."""
    name = args.candidate
    if not name:
        raise ValueError("refit requires --candidate; no automatic winner selection")
    valid = np.flatnonzero(ctx.objects.is_test.to_numpy())
    date = str(ctx.objects.iloc[valid].as_of_date.min().date())
    if ctx.objects.iloc[valid].as_of_date.nunique() != 1:
        raise ValueError("Refit requires one test cutoff")
    dest = args.out_dir / "refit" / name
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "result.json"
    if marker.exists():
        if not args.resume:
            raise FileExistsError("Completed refit exists; use --resume")
        old = json.loads(marker.read_text())
        for n, sha in old["sha256"].items():
            if digest(dest / n) != sha:
                raise ValueError("Corrupt completed refit")
        log(f"Verified completed refit: {dest}")
        return
    component = name
    blend_weight = None
    if name.startswith("blend_"):
        component, suffix = name.removeprefix("blend_").rsplit("_", 1)
        blend_weight = int(suffix.removesuffix("pct")) / 100
        if blend_weight not in ctx.cfg["blend_weights"]:
            raise ValueError("Unplanned blend weight")
    reference = positive = None
    if blend_weight is not None or "joint_positive" in component or component.startswith("season_shift_"):
        # Baseline weights are absent from exported ZIP. Refit all seeds
        # causally at test time; an empty feature block gives the frozen F2.
        reference, positive, baseline_meta = ctx.joint(date, valid, "baseline", dest / "baseline")
    if component.startswith("joint_") and component.endswith("_mean3"):
        block = component.removeprefix("joint_").removesuffix("_mean3")
        if block not in ctx.cfg["joint_blocks"]:
            raise ValueError("Unplanned Joint feature direction")
        z, _, meta = ctx.joint(date, valid, block, dest)
    elif component in candidates(ctx.cfg):
        if blend_weight is not None and component not in ctx.cfg["blend_candidates"]:
            raise ValueError("Unplanned blended component")
        z, meta, model = ctx.cheap(component, date, valid, reference, positive)
        joblib.dump({"model": model, "metadata": meta}, dest / "model.joblib", compress=3)
    else:
        raise ValueError("Unknown refit candidate; use the exact name in scores.csv")
    if blend_weight is not None:
        z = blend_weight * z + (1 - blend_weight) * reference
        meta.update(weight_new=blend_weight, baseline=baseline_meta)
    if not np.isfinite(np.expm1(z)).all():
        raise ValueError("Non-finite money prediction")
    np.savez_compressed(dest / "predictions.npz", Id=ctx.objects.iloc[valid].Id.to_numpy(str),
                        customer_id=ctx.objects.iloc[valid].customer_id.to_numpy(str),
                        z=z, spend_30d=np.maximum(np.expm1(z), 0))
    meta.update(candidate=name, cutoff=date, submission_created=False,
                sha256={str(p.relative_to(dest)): digest(p) for p in dest.rglob("*") if p.is_file() and p != marker})
    atomic_json(dest / "result.json", meta)
    log(f"Refit saved at {dest}; no submission CSV created")
