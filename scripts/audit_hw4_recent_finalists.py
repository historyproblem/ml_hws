"""Freeze finalist recipes and probabilities, then score the recent pair once.

Use separate `freeze` and `score` commands. A finished `score --resume` verifies
existing results without recomputing AP/bootstrap. An interrupted scoring marker
fails closed rather than silently running a second diagnostic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.recent_diagnostic import META, QUARTERS, align_recent, describe_recent

RAW = {
    "train.csv": "30b6acd8c9f2eec4259a775b098017934ba5497cb37689eb00b639433472f0d9",
    "test_student.csv": "1014585cdbdb45e6b1ce7bcb5723700aa025826975a6c5c6910ed5ce39f6f643",
    "inspection_history.csv": "9637ec0279d9cf521795268a0f44099d7d51a795b5155a0d78405a770966afc7",
    "sample_submission.csv": "2d9b5084b510302cd027d46e0c5f9b8e986d48f417d9a582a8b5ec2b3c953b42",
}


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def source_evidence():
    paths = [Path(__file__), ROOT / "src/hw4_inspections/recent_diagnostic.py",
             ROOT / "src/hw4_inspections/classical_analysis.py"]
    return {str(p.resolve()): sha(p) for p in paths}


def signed_files(entries, base):
    if not isinstance(entries, list) or not entries:
        raise ValueError("Nonempty source/evidence file list required for each finalist")
    result = {}
    for item in entries:
        name = item if isinstance(item, str) else item["path"]
        path = Path(name)
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        actual = sha(path)
        if isinstance(item, dict) and item.get("sha256", actual) != actual:
            raise ValueError(f"Changed finalist source/evidence file: {path}")
        if str(path) in result:
            raise ValueError("Duplicate source/evidence path")
        result[str(path)] = actual
    return result


def specification(args):
    recipes = read(args.recipes)
    if (recipes.get("format") != "hw4-recent-finalists-1" or recipes.get("hyperparameter_search") is not False
            or recipes.get("weight_search") is not False or recipes.get("formulas_frozen") is not True):
        raise ValueError("A frozen no-search finalist request is required")
    finalists = recipes.get("finalists")
    if not isinstance(finalists, list) or len(finalists) < 2:
        raise ValueError("Reference and at least one frozen finalist required")
    models = [item["id"] for item in finalists]
    if len(models) != len(set(models)) or recipes.get("reference") not in models:
        raise ValueError("Unique finalist IDs and a listed reference required")
    origins = {}
    for item in finalists:
        if not isinstance(item.get("recipe"), dict) or not item["recipe"] or not isinstance(item.get("formula"), str) or not item["formula"].strip():
            raise ValueError("Every finalist needs an explicit immutable recipe and formula")
        origins[item["id"]] = dict(recipe=item["recipe"], formula=item["formula"],
            sources=signed_files(item.get("sources"), args.recipes.parent),
            old_selection_evidence=signed_files(item.get("old_selection_evidence"), args.recipes.parent))
    raw_hashes = {name: sha(args.raw / name) for name in RAW}
    if not args.smoke and raw_hashes != RAW:
        raise ValueError("Expected the four original official HW4 CSVs")
    if (not args.smoke and args.bootstrap != 2000) or args.bootstrap < 0:
        raise ValueError("Production diagnostic fixes bootstrap=2000; smoke may use fewer")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("A nonnegative integer bootstrap seed must be frozen before scoring")
    train = csv(args.raw / "train.csv")
    if args.smoke:
        if train.groupby("forecast_period").size().max() > 32:
            raise ValueError("Smoke accepts only small synthetic data")
    elif len(train) != 21017:
        raise ValueError("Official training row count differs")
    aligned = align_recent(train, csv(args.predictions), models)
    canonical_digest = hashlib.sha256(aligned.to_csv(index=False).encode("utf-8")).hexdigest()
    payload = dict(format="hw4-one-shot-recent-freeze-1", smoke=args.smoke,
        raw_directory=str(args.raw.resolve()), raw_sha256=raw_hashes,
        input_predictions=str(args.predictions.resolve()), input_predictions_sha256=sha(args.predictions),
        canonical_predictions_sha256=canonical_digest,
        recipes_file=str(args.recipes.resolve()), recipes_sha256=sha(args.recipes),
        finalist_ids=models, reference=recipes["reference"], finalist_definitions=origins,
        selection_explanation=recipes.get("selection_explanation", "Chosen before this diagnostic from old-development evidence"),
        scorer_source_sha256=source_evidence(),
        packages={name: metadata.version(name) for name in ["numpy", "pandas", "scikit-learn"]},
        quarters=list(QUARTERS), rows=len(aligned), unique_sites=int(aligned.site_id.nunique()),
        bootstrap=args.bootstrap, bootstrap_seed=args.seed, fresh_holdout=False,
        parameter_search=False, weight_search=False, model_training=False, kaggle_uploads=0)
    return payload, aligned


def freeze(args):
    payload, aligned = specification(args)
    path = args.out / "freeze.json"
    signature = digest(payload)
    if path.exists():
        frozen = read(path)
        if frozen["request"] != payload or frozen["signature"] != signature:
            raise ValueError("Different recipes, source, evidence or probabilities cannot replace this diagnostic")
        verify_frozen(args.out)
        return dict(status="ALREADY_FROZEN", signature=signature, metrics_computed=0)
    if any(p.name not in {".lock"} for p in args.out.iterdir()):
        raise ValueError("A new diagnostic requires an empty output directory")
    with TemporaryDirectory(dir=args.out, prefix=".pending-freeze-") as temporary:
        temporary = Path(temporary)
        prediction_path = temporary / "predictions.frozen.csv"
        aligned.to_csv(prediction_path, index=False)
        pd.testing.assert_frame_equal(aligned, csv(prediction_path), check_dtype=False, check_exact=True)
        frozen = dict(request=payload, signature=signature,
                      predictions_sha256=sha(prediction_path),
                      frozen_at_utc=datetime.now(timezone.utc).isoformat(), metrics_computed=0)
        write(temporary / "freeze.json", frozen)
        os.link(prediction_path, args.out / "predictions.frozen.csv")
        os.link(temporary / "freeze.json", path)
    return dict(status="FROZEN_UNSCORED", signature=signature, metrics_computed=0, rows=len(aligned))


def verify_frozen(out):
    frozen = read(out / "freeze.json")
    request = frozen["request"]
    if digest(request) != frozen["signature"] or frozen["metrics_computed"] != 0:
        raise ValueError("Broken pre-scoring freeze")
    if (sha(out / "predictions.frozen.csv") != frozen["predictions_sha256"]
            or frozen["predictions_sha256"] != request["canonical_predictions_sha256"]):
        raise ValueError("Frozen probability bytes changed")
    if source_evidence() != request["scorer_source_sha256"]:
        raise ValueError("Scorer source changed after freeze")
    if {n: metadata.version(n) for n in request["packages"]} != request["packages"]:
        raise ValueError("Scoring environment changed after freeze")
    for name, expected in request["raw_sha256"].items():
        if sha(Path(request["raw_directory"]) / name) != expected:
            raise ValueError("Official inputs changed after freeze")
    for file, expected in [(request["recipes_file"], request["recipes_sha256"]),
                           (request["input_predictions"], request["input_predictions_sha256"])]:
        if sha(file) != expected:
            raise ValueError("Recipes or input predictions changed after freeze")
    for item in request["finalist_definitions"].values():
        for group in ["sources", "old_selection_evidence"]:
            if any(sha(path) != expected for path, expected in item[group].items()):
                raise ValueError("Finalist recipe/source/evidence changed after freeze")
    return frozen


def score(args):
    frozen = verify_frozen(args.out)
    request, signature = frozen["request"], frozen["signature"]
    result_directory = args.out / "results"
    if result_directory.exists():
        if not args.resume:
            raise ValueError("This diagnostic is complete; --resume verifies it without rescoring")
        completion = read(result_directory / "complete.json")
        if (completion["signature"] != signature or completion["status"] != "COMPLETE"
                or completion["freeze_sha256"] != sha(args.out / "freeze.json")
                or any(sha(result_directory / n) != h for n, h in completion["artifacts"].items())):
            raise ValueError("Completed diagnostic output changed")
        return dict(status="REUSED_COMPLETE", signature=signature, new_metrics_computed=0,
                    results=str(result_directory))
    started = args.out / "scoring_started.json"
    if started.exists():
        raise ValueError("Earlier scoring started without a completed publication; automatic rescoring refused")
    write(started, dict(signature=signature, freeze_sha256=sha(args.out / "freeze.json"),
                        started_utc=datetime.now(timezone.utc).isoformat(),
                        recovery="Preserve this output and review interrupted diagnostics; do not silently rescore"))
    aligned = csv(args.out / "predictions.frozen.csv")
    result = describe_recent(aligned, request["finalist_ids"], request["reference"],
                             bootstrap=request["bootstrap"], seed=request["bootstrap_seed"])
    with TemporaryDirectory(dir=args.out, prefix=".pending-score-") as temporary:
        temporary = Path(temporary) / "results"
        temporary.mkdir()
        result["quarter_metrics"].to_csv(temporary / "quarter-metrics.csv", index=False)
        result["summary"].to_csv(temporary / "summary.csv", index=False)
        write(temporary / "comparison.json", {**result["protocol"], "freeze_signature": signature,
            "finalists_fixed_before_scores": True, "interpretation": "Descriptive temporal stress diagnostic of already-consumed quarters; no parameter or weight selection"})
        completion = dict(status="COMPLETE", signature=signature, freeze_sha256=sha(args.out / "freeze.json"),
            scoring_invocations=1, new_fits=0, model_selection_performed=False, weight_search=False,
            artifacts={p.name: sha(p) for p in temporary.iterdir()},
            recent_quarters_scored=list(QUARTERS), fresh_holdout=False, kaggle_uploads=0)
        write(temporary / "complete.json", completion)
        os.rename(temporary, result_directory)
    return dict(status="COMPLETE", signature=signature, new_metrics_computed=1, results=str(result_directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("freeze")
    for name in ["raw", "recipes", "predictions", "out"]:
        first.add_argument("--" + name, type=Path, required=True)
    first.add_argument("--bootstrap", type=int, default=2000)
    first.add_argument("--seed", type=int, default=20261122)
    first.add_argument("--smoke", action="store_true")
    second = commands.add_parser("score")
    second.add_argument("--out", type=Path, required=True)
    second.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = freeze(args) if args.command == "freeze" else score(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
