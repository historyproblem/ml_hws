"""Post-Public exploratory native TabICLv2 ensemble32; OLD3 plus full test only."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import tarfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

import run_hw4_tabicl_v100 as original
import run_hw4_tabpfn_v100 as common

CASE = original.CASES[0]
SETTINGS = {**original.SETTINGS, "n_estimators": 32}
FILES = ("protocol.json", "completion.json", "modelinfo.json", "old_oof.csv",
         "test_predictions.csv", "fold_metrics.csv")
SOURCES = tuple(dict.fromkeys(["scripts/run_hw4_tabicl_ensemble32.py",
    "scripts/run_hw4_tabicl_ensemble32.sh", "scripts/analyze_hw4_tabicl_ensemble32.py",
    "src/safe_archives.py", *original.SOURCES]))


def pack(out):
    target = out / "return_results.tar.gz"
    with target.with_suffix(".partial").open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for name in FILES:
                    info = archive.gettarinfo(str(out / name), arcname=name)
                    info.uid = info.gid = 0; info.uname = info.gname = ""; info.mtime = 0
                    with (out / name).open("rb") as source: archive.addfile(info, source)
    os.replace(target.with_suffix(".partial"), target)
    return target


def run(args):
    original.verify_env(args)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # The original file is untouched; this process uses its verified native job
    # implementation with an explicitly recorded replacement settings object.
    original.SETTINGS = {**SETTINGS, "n_estimators": 2 if args.smoke else 32}
    attempts = common.offline_runtime(args.out / ".offline-state")
    if args.smoke:
        result = original.probe(args)
        if attempts: raise ValueError("Smoke attempted network access")
        common.write_json(args.out / "smoke.json", {"status": "PASS", **result,
            "native_estimators": 2, "weights_sha256": common.sha(args.checkpoint),
            "real_hw4_inference_jobs": 0})
        print(json.dumps({"event": "smoke_complete", **result}), flush=True)
        return
    if args.device != "cuda": raise ValueError("Registered real run requires CUDA")
    inputs = {n: common.sha(args.raw / n) for n in common.RAW}
    if inputs != common.RAW: raise ValueError("Official input SHA mismatch")
    plan = {"format": "hw4-tabicl-native32-postpublic-1", "case": CASE,
        "purpose": "Exploratory extension after observing Public11 .34642 and12 .34242; no independent confirmation claimed",
        "settings": SETTINGS, "checkpoint_sha256": original.WEIGHT,
        "source_sha256": {n: common.sha(common.ROOT / n) for n in SOURCES},
        "input_sha256": inputs, "old_quarters": list(common.OLD),
        "periods": [*common.OLD, "full"], "recent_periods": [],
        "context": "All strictly earlier official rows; stable (forecast_period,row_id); all21017 for test",
        "view": "Same compact94 as native4 Public11; no feature/preprocessing/weight change",
        "ensemble": "Native32 transformed views, seed42, batch1; independent full execution, no prefix reuse assumed",
        "versions": {n: metadata.version(n) for n in ["tabicl", "scikit-learn", "torch", "numpy", "pandas", "scipy"]},
        "python": platform.python_version(), "device": args.device,
        "gpu": common.gpu_info(args.device), "threads": args.threads,
        "new_2024q2_q3_metrics": False, "kaggle_uploads": 0,
        "score_policy": "Only old3; recent outcomes are not loaded into scoring. No weight search."}
    plan["signature"] = common.digest(plan)
    protocol = args.out / "protocol.json"
    if protocol.exists():
        if not args.resume or common.load(protocol) != plan: raise ValueError("Changed run provenance")
    else:
        if any(p.name not in {".lock", "run.log", ".offline-state"} for p in args.out.iterdir()):
            raise ValueError("New run requires an empty output directory")
        common.write_json(protocol, plan)
    train, test, sample = [pd.read_csv(args.raw / n) for n in
        ["train.csv", "test_student.csv", "sample_submission.csv"]]
    if (len(train), len(test)) != (21017, 2246) or "target" in test:
        raise ValueError("Official shapes/target mismatch")
    if any(not f.row_id.is_unique or f.row_id.isna().any() for f in (train, test, sample)):
        raise ValueError("Invalid official identity")
    if set(train.row_id) & set(test.row_id) or set(test.row_id) != set(sample.row_id):
        raise ValueError("Official identity overlap/mismatch")
    rows = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    history = pd.read_csv(args.raw / "inspection_history.csv")
    features, cats = common.build_features(rows, history)
    features = pd.concat([features, common.build_context_features(rows, history)], axis=1)
    frame, cat_ids = common.views(features, cats)[common.CASES[0]]
    if len(frame.columns) != 94: raise ValueError("Expected compact94")
    keys = ["row_id", "site_id", "forecast_period", "target"]
    old = pd.concat([train.loc[train.forecast_period.eq(q), keys] for q in common.OLD], ignore_index=True)
    forecast = test[["row_id", "forecast_period"]].copy()
    oldp, jobs, metrics = [], [], []
    for period in (*common.OLD, "full"):
        values, record = original.job(args, plan["signature"], CASE, period, train, test, frame, cat_ids)
        if record["n_estimators"] != 32: raise ValueError("Wrong actual native ensemble size")
        jobs.append(record)
        if period == "full": forecast[CASE] = values
        else:
            oldp.append(values)
            y = train.loc[train.forecast_period.eq(period), "target"]
            metrics.append({"forecast_period": period, "case": CASE, "seed": 42,
                "n": len(y), "positive": int(y.sum()), "train_rows": record["identity"]["train_rows"],
                "ap": float(average_precision_score(y, values))})
    if attempts: raise ValueError("Network attempted during inference")
    old[CASE] = np.concatenate(oldp)
    common.write_csv(args.out / "old_oof.csv", old)
    common.write_csv(args.out / "test_predictions.csv", forecast)
    common.write_csv(args.out / "fold_metrics.csv", pd.DataFrame(metrics))
    common.write_json(args.out / "modelinfo.json", {"jobs": jobs, "runtime_network_attempts": 0,
        "checkpoint_sha256": original.WEIGHT, "settings": SETTINGS})
    common.write_json(args.out / "completion.json", {"status": "complete", "signature": plan["signature"],
        "jobs": 4, "old_rows": len(old), "test_rows": len(test), "new_2024q2_q3_metrics": False,
        "kaggle_uploads": 0, "artifacts": {n: common.sha(args.out / n) for n in FILES if n != "completion.json"}})
    archive = pack(args.out)
    print(json.dumps({"event": "complete", "jobs": 4, "archive": str(archive),
        "sha256": common.sha(archive)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "out", "checkpoint"): parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); args.checkpoint = args.checkpoint.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock, threadpool_limits(limits=args.threads):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__": main()
