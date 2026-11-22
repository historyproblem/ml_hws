"""Rebuild the immutable HW4 final-four feature cache from four official CSVs.

From a checkout with the existing scientific Python environment::

    python scripts/build_hw4_final_feature_cache.py --raw data --out analysis/final-features
    python scripts/build_hw4_final_feature_cache.py --raw data --out analysis/final-features --resume

The directory receives features.joblib and features-meta.json. It is never
used to overwrite an existing cache. --resume verifies identical raw inputs,
builder/CLI sources, package versions and the saved cache hash, without rebuild.

The four CSVs suffice ONLY for feature construction without official current-row targets. Model
runners additionally require the previously verified baseline_oof.csv and
submission_04_context_cb.csv via --baseline-cv/--context-test. Those predictions
are not generated here and cannot be inferred from four raw files.

Current Mac runtime/data paths::

    /Users/khlystov/Documents/ChatGPT/ML ДЗ/.venv/bin/python
    /Users/khlystov/Documents/ChatGPT/ML ДЗ/data/hw4/raw
    /Users/khlystov/Documents/ChatGPT/ML ДЗ/analysis/hw4-next-round-20261121/baseline_oof.csv
    /Users/khlystov/Documents/ChatGPT/ML ДЗ/outputs/hw4/final-20261120/submission_04_context_cb.csv

For Mac model runners needing OpenMP, use the existing Torch library directory
(no extra dependency installation) in the runner process, for example::

    export HW4_OPENMP="/Users/khlystov/Documents/ChatGPT/ML ДЗ/.venv/lib/python3.12/site-packages/torch/lib"
    export DYLD_LIBRARY_PATH="$HW4_OPENMP${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

This builder itself neither imports native CatBoost/XGBoost nor trains models.
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
from tempfile import TemporaryDirectory
import time

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features
from hw4_inspections.seasonal_context import COLUMNS as SEASONAL_COLUMNS, build_seasonal_context_features
from hw4_inspections.transition_features import FEATURE_COLUMNS as TRANSITION_COLUMNS, build_transition_features

RAW_FILES = ("train.csv", "test_student.csv", "inspection_history.csv", "sample_submission.csv")
BUILDERS = ("features.py", "extensions.py", "transition_features.py", "seasonal_context.py")


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fingerprints(raw):
    return dict(
        raw_sha256={name: sha(raw / name) for name in RAW_FILES},
        builder_sha256={name: sha(ROOT / "src/hw4_inspections" / name) for name in BUILDERS},
        builder_cli_sha256=sha(Path(__file__)),
        package_versions={name: metadata.version(name) for name in ["numpy", "pandas", "joblib"]},
        python=platform.python_version(),
    )


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build(raw, out, resume=False):
    absent = [name for name in RAW_FILES if not (raw / name).is_file()]
    if absent:
        raise FileNotFoundError(f"Missing official CSVs in {raw.resolve()}: {absent}")
    evidence = fingerprints(raw)
    frozen = signature(evidence)
    cache_path, meta_path = out / "features.joblib", out / "features-meta.json"
    if cache_path.exists() or meta_path.exists():
        if not resume:
            raise FileExistsError("Existing feature cache is immutable; use --resume or a new --out")
        if not cache_path.is_file() or not meta_path.is_file():
            raise ValueError("Incomplete existing cache preserved; choose a new --out directory")
        previous = json.loads(meta_path.read_text())
        if previous.get("build_signature") != frozen:
            raise ValueError("Changed raw data, builder/CLI source or environment; existing cache preserved")
        if previous.get("cache_sha256") != sha(cache_path):
            raise ValueError("Existing feature cache bytes changed; rebuild into a new directory")
        print(json.dumps({"event": "cache_reused", "cache": str(cache_path),
                          "cache_sha256": previous["cache_sha256"], "feature_builds": 0}), flush=True)
        return previous
    if any(path.name != ".lock" for path in out.iterdir()):
        raise ValueError("New cache requires an empty --out directory")
    started = time.monotonic()
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test_student.csv")
    history = pd.read_csv(raw / "inspection_history.csv")
    sample = pd.read_csv(raw / "sample_submission.csv")
    for frame, name in [(train, "train"), (test, "test"), (sample, "sample")]:
        if (not frame.columns.is_unique or "row_id" not in frame
                or not frame.row_id.is_unique or frame.row_id.isna().any()):
            raise ValueError(f"Invalid {name} row IDs/schema")
    if ("target" not in train or "target" in test or not train.target.isin([0, 1]).all()
            or set(train.row_id) & set(test.row_id) or set(sample.row_id) != set(test.row_id)):
        raise ValueError("Invalid official train/test/sample separation")
    # The target is physically absent from every feature-builder input.
    rows = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    if "target" in rows:
        raise AssertionError("Official current-row target entered feature construction")
    print(json.dumps({"event": "feature_build_start", "train_rows": len(train),
                      "test_rows": len(test), "history_rows": len(history)}), flush=True)
    base, cats = build_features(rows, history)
    base = pd.concat([base, build_context_features(rows, history)], axis=1)
    transition = build_transition_features(rows, history)
    seasonal = build_seasonal_context_features(rows, history)
    if len(base.columns) != 346 or len(cats) != 10:
        raise ValueError("Expected the frozen 346-column base and ten categorical columns")
    if list(transition) != list(TRANSITION_COLUMNS) or list(seasonal) != list(SEASONAL_COLUMNS):
        raise ValueError("Transition/seasonal feature schema changed")
    views = dict(base=base, transition=pd.concat([base, transition], axis=1),
                 seasonal=pd.concat([base, seasonal], axis=1))
    for frame in views.values():
        if (not frame.index.equals(pd.RangeIndex(len(rows))) or not frame.columns.is_unique
                or {"row_id", "site_id", "target", "forecast_period", "history_before"} & set(frame)):
            raise ValueError("Cached feature rows/schema/identifier exclusion violated")
    cache = dict(train=train, test=test, sample=sample, views=views, cats=list(cats))
    if fingerprints(raw) != evidence:
        raise ValueError("Raw inputs or sources changed during construction; no cache published")
    with TemporaryDirectory(dir=out, prefix=".pending-feature-cache-") as temp:
        temp = Path(temp)
        built = temp / "features.joblib"
        joblib.dump(cache, built, compress=3)
        record = dict(
            **evidence, build_signature=frozen, builder_cli=Path(__file__).name,
            cache_sha256=sha(built), cache_bytes=built.stat().st_size,
            views={name: len(frame.columns) for name, frame in views.items()},
            rows=len(rows), train_rows=len(train), test_rows=len(test), history_rows=len(history),
            feature_build_seconds=time.monotonic() - started,
            official_current_row_targets_used_in_features=False,
            historical_outcomes_used_with_strict_snapshot_and_cutoff=True,
            new_model_metrics=0, new_closed_quarter_metrics=0, model_fits=0,
            semantics={
                "row_order": "train rows followed by test rows; original RangeIndex/order retained in cached raw frames",
                "snapshots": "exact forecast snapshot and strict per-row history_before; snapshots never pooled",
                "context": "other-site context with focal site removed from groups and global priors",
                "transitions": "adjacent routine pairs with both endpoints past cutoff; unknown outcomes excluded, focal pairs removed from all priors and code adjustments",
                "seasonal": "other-site first routine in admissible completed past same-season quarters; conservative weekly boundary/coverage rules",
                "codes": "opaque recorded tokens; no external code dictionary or target recovery",
            },
        )
        temporary_meta = temp / "features-meta.json"
        temporary_meta.write_text(json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        if fingerprints(raw) != evidence:
            raise ValueError("Raw inputs or sources changed during serialization; no cache published")
        # Hard-link publication fails rather than overwriting any destination.
        # The lock protects cooperating runs; these calls also protect an
        # externally created file between the empty-directory check and commit.
        os.link(built, cache_path)
        os.link(temporary_meta, meta_path)
    print(json.dumps({"event": "feature_cache_complete", "cache": str(cache_path),
                      "cache_sha256": record["cache_sha256"], "views": record["views"],
                      "seconds": record["feature_build_seconds"], "model_fits": 0}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build(args.raw, args.out, args.resume)


if __name__ == "__main__":
    main()
