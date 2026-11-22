"""Anchor-free GPU producer for three fixed TabM preprocessing recipes.

Only four official CSVs are required. No pretrained weights or external data.
Returns old-quarter OOF, unscored recent-quarter and test probabilities.
Full forecasts are diagnostic: this producer never uploads or promotes a CSV.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
from tempfile import TemporaryDirectory
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features
from hw4_inspections.tabm_preprocessing import CASES, feature_view, fit_bounded
from hw4_inspections.tabm_training import OPTIMIZER, predict_bundle

QUARTERS = ("2023-07-01", "2023-10-01", "2024-01-01")
RECENT = ("2024-04-01", "2024-07-01")
SEEDS = (42, 17, 2026)
RAW = ("train.csv", "test_student.csv", "inspection_history.csv", "sample_submission.csv")
SOURCES = ("features.py", "extensions.py", "neural.py", "tabm_models.py", "tabm_training.py",
           "classical_features.py", "tabm_preprocessing.py")
RETURN_FILES = ("protocol.json", "completion.json", "old_oof.csv", "test_predictions.csv",
                "fold_metrics.csv", "epoch_selections.json", "recent_predictions.csv")


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def signature(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def run_job(args, sig, name, seed, quarter, phase, train, frame, cats, epochs=None):
    period = train.loc[train.forecast_period.lt(quarter), "forecast_period"].max() if phase == "inner" else quarter
    fit = np.arange(len(train)) if phase == "full" else np.flatnonzero(train.forecast_period.lt(period))
    valid = np.arange(len(train), len(frame)) if phase == "full" else np.flatnonzero(train.forecast_period.eq(period))
    if not len(fit) or not len(valid):
        raise ValueError("Empty neural partition")
    identity = dict(signature=sig, case=name, seed=seed, quarter=quarter, phase=phase,
                    actual_period=period, train_rows=len(fit), predict_rows=len(valid), fixed_epochs=epochs)
    path = args.out / "jobs" / f"{name}__{quarter}__s{seed}__{phase}"
    if path.exists():
        if not args.resume:
            raise ValueError("Existing checkpoint requires --resume")
        marker = json.loads((path / "complete.json").read_text())
        if (marker["identity"] != identity or not marker["reload_exact"]
                or set(marker["sha256"]) != {"model.joblib", "prediction.npy", "curve.json"}
                or any(sha(path / n) != digest for n, digest in marker["sha256"].items())):
            raise ValueError("Changed completed neural job")
        prediction = np.load(path / "prediction.npy", allow_pickle=False)
        if (prediction.shape != (len(valid),) or not np.isfinite(prediction).all()
                or ((prediction < 0) | (prediction > 1)).any()):
            raise ValueError("Changed prediction shape or finite values")
        print(json.dumps(dict(event="job_reused", **identity)), flush=True)
        return prediction, marker["selected_epochs"]
    print(json.dumps(dict(event="job_start", **identity)), flush=True)
    started = time.monotonic()
    kwargs = dict(mode=CASES[name]["mode"], seed=seed, device=args.device, threads=args.threads,
                  k=2 if args.smoke else 32, width=8 if args.smoke else 256,
                  blocks=1 if args.smoke else 3, bins=4 if args.smoke else 16,
                  embedding_dim=2 if args.smoke else 8, batch_size=256,
                  epochs=args.max_epochs if phase == "inner" else epochs, patience=args.patience)
    if phase == "inner":
        kwargs.update(eval_frame=frame.iloc[valid], eval_target=train.target.iloc[valid])
    def log(event, **details):
        if event in {"epoch_complete", "first_batch_complete"}:
            print(json.dumps(dict(event=event, case=name, quarter=quarter, seed=seed, phase=phase, **details)), flush=True)
    bundle = fit_bounded(frame.iloc[fit], train.target.iloc[fit], cats, log=log, **kwargs)
    prediction = predict_bundle(bundle, frame.iloc[valid])
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".pending-gpu-") as temporary:
        temporary = Path(temporary) / "job"
        temporary.mkdir()
        joblib.dump(bundle, temporary / "model.joblib", compress=1)
        if not np.array_equal(prediction, predict_bundle(joblib.load(temporary / "model.joblib"), frame.iloc[valid])):
            raise ValueError("Reloaded GPU model did not reproduce forecasts exactly")
        np.save(temporary / "prediction.npy", prediction, allow_pickle=False)
        write_json(temporary / "curve.json", bundle["curve"])
        write_json(temporary / "complete.json", dict(
            identity=identity, reload_exact=True, selected_epochs=bundle["epochs"],
            trained_epochs=bundle["trained_epochs"], inner_best_ap=bundle["best_ap"],
            sha256={n: sha(temporary / n) for n in ["model.joblib", "prediction.npy", "curve.json"]},
            seconds=time.monotonic() - started, new_2024q2_q3_metrics=False))
        os.rename(temporary, path)
    del bundle
    if args.device == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps(dict(event="job_complete", **identity, seconds=time.monotonic() - started)), flush=True)
    return prediction, epochs if phase != "inner" else json.loads((path / "complete.json").read_text())["selected_epochs"]


def run(args):
    if args.device == "cpu" and not args.smoke:
        raise ValueError("Production producer is for CUDA; use CPU only for synthetic smoke")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing a silent CPU fallback")
    if min(args.max_epochs, args.patience, args.threads) < 1:
        raise ValueError("Positive limits required")
    source_hashes = {"src/hw4_inspections/" + n: sha(ROOT / "src/hw4_inspections" / n) for n in SOURCES}
    source_hashes[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    plan = dict(
        format="hw4-tabm-preprocessing-v100-1", smoke=args.smoke, cases=CASES, seeds=list(SEEDS),
        old_quarters=list(QUARTERS), recent_quarters=list(RECENT), input_sha256={n: sha(args.raw / n) for n in RAW},
        source_sha256=source_hashes,
        versions={n: metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "torch", "tabm", "rtdl-num-embeddings", "joblib"]},
        machine=platform.machine(), python=platform.python_version(), device=args.device,
        gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        threads=args.threads, max_epochs=args.max_epochs, patience=args.patience,
        architecture=dict(k=2 if args.smoke else 32, width=8 if args.smoke else 256,
                          blocks=1 if args.smoke else 3, bins=4 if args.smoke else 16,
                          embedding_dim=2 if args.smoke else 8, dropout=.1, batch_size=256),
        optimizer=OPTIMIZER, epoch_selection="seed42 previous-quarter inner AP; fresh outer refit; same counts17/2026",
        full="All3 diagnostic recipes mean all3 seeds; epochs median3 earlier inner choices; local admission deferred until exact anchors available",
        recent="User authorized one additional temporal diagnostic. Predict two recent quarters with the frozen full recipe/old-inner median epochs; NO recent AP or recipe selection in this producer. Local diagnostic happens once after finalist freeze.",
        variants_deferred=["raw", ".75*context04+.25*neural", ".75*previous07+.25*neural"],
        full_policy_change="Before any real neural fit: user confirmed V100. Anchor-free remote producer computes all three diagnostic forecasts; local exporter still admits at most one family finalist. No model/weight selection uses2024Q2/Q3.",
        external_pretrained_weights=False, new_2024q2_q3_metrics=False, kaggle_uploads=0,
    )
    plan["signature"] = signature(plan)
    protocol = args.out / "protocol.json"
    if protocol.exists():
        if not args.resume or json.loads(protocol.read_text()) != plan:
            raise ValueError("Changed source, data or neural protocol; use a new output folder")
    else:
        if any(p.name not in {".lock", "run.log"} for p in args.out.iterdir()):
            raise ValueError("New GPU run requires an empty output directory")
        write_json(protocol, plan)
    train = pd.read_csv(args.raw / "train.csv")
    test = pd.read_csv(args.raw / "test_student.csv")
    sample = pd.read_csv(args.raw / "sample_submission.csv")
    if ("target" in test or not train.target.isin([0, 1]).all()
            or not train.row_id.is_unique or not test.row_id.is_unique or not sample.row_id.is_unique
            or train.row_id.isna().any() or test.row_id.isna().any() or sample.row_id.isna().any()
            or len(sample) != len(test)
            or set(train.row_id) & set(test.row_id) or set(test.row_id) != set(sample.row_id)):
        raise ValueError("Invalid official train/test/sample")
    if args.smoke:
        if train.groupby("forecast_period").size().max() > 32:
            raise ValueError("Smoke is for small synthetic fixtures only")
    elif len(train) != 21017 or len(test) != 2246:
        raise ValueError("Official data size changed")
    rows = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    history = pd.read_csv(args.raw / "inspection_history.csv")
    frame, cats = build_features(rows, history)
    frame = pd.concat([frame, build_context_features(rows, history)], axis=1)
    if len(frame.columns) != 346 or len(cats) != 10:
        raise ValueError("Expected full346 features")
    old = pd.concat([train.loc[train.forecast_period.eq(q), ["row_id", "site_id", "forecast_period", "target"]] for q in QUARTERS], ignore_index=True)
    recent = pd.concat([train.loc[train.forecast_period.eq(q), ["row_id", "site_id", "forecast_period", "target"]] for q in RECENT], ignore_index=True)
    forecast = test[["row_id", "forecast_period"]].copy()
    metrics, selections = [], {}
    for case, spec in CASES.items():
        view, categories = feature_view(frame, cats, spec["view"])
        counts = []
        seed_predictions = {seed: [] for seed in SEEDS}
        for quarter in QUARTERS:
            _, count = run_job(args, plan["signature"], case, 42, quarter, "inner", train, view, categories)
            counts.append(count)
            y = train.loc[train.forecast_period.eq(quarter), "target"]
            for seed in SEEDS:
                p, _ = run_job(args, plan["signature"], case, seed, quarter, "outer", train, view, categories, count)
                seed_predictions[seed].append(p.astype(np.float64))
                metrics.append(dict(case=case, seed=seed, forecast_period=quarter,
                                    n=len(y), positive=int(y.sum()), ap=float(average_precision_score(y, p)), epochs=count))
        for seed in SEEDS:
            old[f"{case}__s{seed}"] = np.concatenate(seed_predictions[seed])
        old[f"{case}__mean3"] = old[[f"{case}__s{seed}" for seed in SEEDS]].mean(axis=1)
        epochs = int(np.median(counts))
        selections[case] = dict(inner_selected_epochs=counts, full_epochs=epochs)
        full = [run_job(args, plan["signature"], case, seed, "full", "full", train, view, categories, epochs)[0].astype(np.float64)
                for seed in SEEDS]
        for seed, probability in zip(SEEDS, full):
            forecast[f"{case}__s{seed}"] = probability
        forecast[f"{case}__mean3"] = np.mean(full, axis=0)
        for seed in SEEDS:
            recent[f"{case}__s{seed}"] = np.concatenate([
                run_job(args, plan["signature"], case, seed, q, "recent", train, view, categories, epochs)[0].astype(np.float64)
                for q in RECENT])
        recent[f"{case}__mean3"] = recent[[f"{case}__s{seed}" for seed in SEEDS]].mean(axis=1)
    old.to_csv(args.out / "old_oof.csv", index=False)
    forecast.to_csv(args.out / "test_predictions.csv", index=False)
    recent.to_csv(args.out / "recent_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(args.out / "fold_metrics.csv", index=False)
    write_json(args.out / "epoch_selections.json", selections)
    write_json(args.out / "completion.json", dict(
        status="complete", signature=plan["signature"], old_rows=len(old), test_rows=len(test),
        jobs=len(list((args.out / "jobs").glob("*/complete.json"))), expected_jobs=63,
        recent_rows=len(recent), recent_scores_computed=0,
        artifacts={n: sha(args.out / n) for n in RETURN_FILES if n != "completion.json"},
        new_2024q2_q3_metrics=False, kaggle_uploads=0, exports="diagnostic, local gate pending"))
    with tarfile.open(args.out / "return_results.tar.gz", "w:gz") as archive:
        for name in RETURN_FILES:
            archive.add(args.out / name, arcname=name)
    print(json.dumps(dict(event="complete", jobs=63, archive=str(args.out / "return_results.tar.gz"),
                          new_2024q2_q3_metrics=False, kaggle_uploads=0)), flush=True)
    return json.loads((args.out / "completion.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
