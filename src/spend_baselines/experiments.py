"""Explicit MLP experiments with inner temporal stopping and outer evaluation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from .data import load_dataset, feature_matrix
from .models import TorchMLP, rmsle_from_logs


def inner_temporal_split(objects, outer_date):
    """Return global masks; overlapping target windows are purged from inner fit."""
    outer_date = pd.Timestamp(outer_date)
    outer_fit = objects.target_end.le(outer_date)
    outer_valid = objects.as_of_date.eq(outer_date)
    if not outer_fit.any() or not outer_valid.any() or (outer_fit & outer_valid).any():
        raise ValueError("Invalid outer temporal split")
    inner_date = objects.loc[outer_fit, "as_of_date"].max()
    inner_fit = outer_fit & objects.target_end.le(inner_date)
    inner_valid = outer_fit & objects.as_of_date.eq(inner_date)
    if inner_date >= outer_date or not inner_fit.any() or not inner_valid.any():
        raise ValueError("Not enough history for an inner temporal validation date")
    if (inner_fit & inner_valid).any() or (outer_valid & (inner_fit | inner_valid)).any():
        raise ValueError("Inner/outer temporal roles overlap")
    return {"inner_date": inner_date, "inner_fit": inner_fit, "inner_valid": inner_valid,
            "outer_fit": outer_fit, "outer_valid": outer_valid,
            "n_purged": int((outer_fit & ~inner_fit & ~inner_valid).sum())}


def experiment_plan(base, specification, *, variants=None, seeds=None, epochs=None):
    records = specification["variants"]
    names = [r["name"] for r in records]
    if len(set(names)) != len(names) or not all(re.fullmatch(r"[a-z][a-z0-9_-]*", n) for n in names):
        raise ValueError("Experiment names must be unique simple identifiers")
    chosen = names if variants is None else variants
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(names):
        raise ValueError(f"Choose explicit known variants: {names}")
    chosen_seeds = specification["seeds"] if seeds is None else seeds
    if not chosen_seeds or len(set(chosen_seeds)) != len(chosen_seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if any(not isinstance(s, int) or not 0 <= s < 2**32 for s in chosen_seeds):
        raise ValueError("Seeds must be integers in [0, 2**32)")
    maximum = specification["max_epochs"] if epochs is None else epochs
    stopping = specification["early_stopping"]
    if maximum < 1 or stopping["patience"] < 1 or stopping["min_delta"] < 0:
        raise ValueError("Invalid maximum epochs or stopping settings")
    settings_by_name = {r["name"]: r["settings"] for r in records}
    resolved = []
    for name in chosen:
        settings = deepcopy(base["mlp"])
        settings.update(settings_by_name[name])
        settings["epochs"] = maximum
        resolved.append({"name": name, "settings": settings})
    return {"variants": resolved, "seeds": list(chosen_seeds), "early_stopping": deepcopy(stopping),
            "validation_dates": list(base["validation_dates"]), "feature_group": base["feature_group"]}


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def guard_run_directory(directory, expected, resume=False):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_manifest.json"
    if path.exists():
        if not resume:
            raise ValueError("Output already has a run; use --resume or a different --out-dir")
        if json.loads(path.read_text()) != expected:
            raise ValueError("Resume needs identical code, inputs, settings, versions and device; use a new --out-dir")
    else:
        if any(directory.iterdir()):
            raise ValueError("Nonempty output without a run manifest; use a new --out-dir")
        atomic_json(path, expected)


def completed_job(directory):
    path = directory / "result.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    required = {"checkpoint.pt", "selection_history.csv", "refit_history.csv", "validation_predictions.csv"}
    if set(result.get("artifact_sha256", {})) != required:
        raise ValueError(f"Incomplete saved job manifest: {directory.name}")
    for name, expected in result["artifact_sha256"].items():
        artifact = directory / name
        if Path(name).name != name or not artifact.is_file():
            raise ValueError(f"Incomplete saved job: {directory.name}")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed saved artifact: {directory.name}/{name}")
    return result


def write_summaries(results, directory):
    scores = pd.DataFrame([{k: v for k, v in r.items() if k != "artifact_sha256"} for r in results])
    scores.to_csv(directory / "scores.csv", index=False)
    summary = scores.groupby(["variant", "date"], sort=True).agg(
        n_seeds=("seed", "size"), mean_rmsle=("RMSLE", "mean"), std_rmsle=("RMSLE", "std"),
        min_rmsle=("RMSLE", "min"), max_rmsle=("RMSLE", "max"),
        mean_selected_epoch=("selected_epoch", "mean"),
    ).reset_index()
    summary.to_csv(directory / "summary.csv", index=False)
    pooled = []
    for (variant, seed), group in scores.groupby(["variant", "seed"], sort=True):
        pooled.append({"variant": variant, "seed": int(seed), "n_folds": len(group),
                       "n_valid": int(group.n_valid.sum()),
                       "pooled_rmsle": float(np.sqrt(np.average(group.RMSLE**2, weights=group.n_valid)))})
    pd.DataFrame(pooled).to_csv(directory / "pooled_by_seed.csv", index=False)
    return summary


def run_mlp_experiments(args, base_config):
    # Imported here to keep CLI startup and the ordinary baseline path independent.
    from .cli import manifest
    specification = json.loads(args.experiment_config.read_text())
    plan = experiment_plan(base_config, specification, variants=args.variants,
                           seeds=args.seeds, epochs=args.epochs)
    print("Loading official data and checking every known training target", flush=True)
    train, _, history, checks = load_dataset(args.data_dir)
    info = manifest(args, base_config, checks)
    info["experiment_plan"] = plan
    info["protocol"] = "select epoch on latest inner date; refit from scratch on all outer-fit rows; evaluate outer once"
    guard_run_directory(args.out_dir, info, args.resume)
    print("Building strictly historical features", flush=True)
    features = history.build_features(train)
    x = feature_matrix(features, plan["feature_group"])
    results = []
    for date in plan["validation_dates"]:
        split = inner_temporal_split(train, date)
        fit, inner = split["inner_fit"], split["inner_valid"]
        outer_fit, outer_valid = split["outer_fit"], split["outer_valid"]
        for variant in plan["variants"]:
            for seed in plan["seeds"]:
                job_id = f"{pd.Timestamp(date).date()}__{variant['name']}__seed{seed}"
                directory = args.out_dir / job_id
                existing = completed_job(directory) if args.resume else None
                if existing is not None:
                    if (existing["date"], existing["variant"], existing["seed"]) != (date, variant["name"], seed):
                        raise ValueError(f"Saved job metadata differs: {job_id}")
                    print(f"Resuming: completed {job_id}", flush=True)
                    results.append(existing)
                    write_summaries(results, args.out_dir)
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                print(f"{job_id}: inner date={split['inner_date'].date()}, epoch selection", flush=True)
                selector = TorchMLP(variant["settings"], seed, args.device, args.threads)
                selector.fit(x[fit], np.log1p(train.loc[fit, "spend_30d"].to_numpy()),
                             validation=(x[inner], np.log1p(train.loc[inner, "spend_30d"].to_numpy())),
                             early_stopping=plan["early_stopping"])
                chosen_epoch = selector.best_epoch_
                pd.DataFrame(selector.history).to_csv(directory / "selection_history.csv", index=False)
                refit_settings = deepcopy(variant["settings"])
                refit_settings["epochs"] = chosen_epoch
                print(f"{job_id}: refit on {int(outer_fit.sum())} rows for {chosen_epoch} epochs", flush=True)
                model = TorchMLP(refit_settings, seed, args.device, args.threads)
                model.fit(x[outer_fit], np.log1p(train.loc[outer_fit, "spend_30d"].to_numpy()))
                # Outer targets are read only now, after epoch selection and refitting.
                prediction = np.maximum(model.predict(x[outer_valid]), 0.0)
                score = rmsle_from_logs(train.loc[outer_valid, "spend_30d"].to_numpy(), prediction)
                checkpoint_metadata = {"variant": variant["name"], "seed": seed,
                                       "outer_date": date, "inner_date": str(split["inner_date"].date()),
                                       "feature_group": plan["feature_group"], "selected_epoch": chosen_epoch}
                model.save(directory / "checkpoint.pt", metadata=checkpoint_metadata)
                pd.DataFrame(model.history).to_csv(directory / "refit_history.csv", index=False)
                rows = train.loc[outer_valid, ["Id", "customer_id", "as_of_date", "spend_30d"]].copy()
                rows["prediction"] = np.expm1(prediction)
                rows.to_csv(directory / "validation_predictions.csv", index=False)
                result = {"date": date, "variant": variant["name"], "seed": seed, "RMSLE": score,
                          "inner_date": str(split["inner_date"].date()), "selected_epoch": chosen_epoch,
                          "epochs_searched": selector.epochs_run_, "selection_stop_reason": selector.stop_reason_,
                          "inner_best_rmsle": selector.best_validation_rmsle_,
                          "n_inner_fit": int(fit.sum()), "n_inner_valid": int(inner.sum()),
                          "n_inner_purged": split["n_purged"],
                          "n_refit": int(outer_fit.sum()), "n_valid": int(outer_valid.sum()),
                          "seconds": time.monotonic() - started}
                result["artifact_sha256"] = {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ["checkpoint.pt", "selection_history.csv", "refit_history.csv", "validation_predictions.csv"]}
                atomic_json(directory / "result.json", result)  # Written last: job is now complete.
                results.append(result)
                write_summaries(results, args.out_dir)
                print(f"{job_id}: outer RMSLE={score:.6f}", flush=True)
    summary = write_summaries(results, args.out_dir)
    print(summary.to_string(index=False), flush=True)
    print("Report saved; no variant was automatically selected and no Kaggle upload was performed", flush=True)
