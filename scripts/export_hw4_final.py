"""Freeze two HW4 finalists, score the two reserved quarters once, export test CSV.

No architecture/weight search and no Kaggle uploads. The two-epoch legacy visit
recipe is fixed from the median of earlier development choices5/2/2.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.classical_analysis import prepare_weighted_ap
from hw4_inspections.classical_features import select_feature_columns
from hw4_inspections.classical_models import SPECS, fit_model, predict_model
from hw4_inspections.delivery import choose_primary, make_candidates, validate_submission
from hw4_inspections.delivery_neural import fit_visit, predict_visit
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features
from hw4_inspections.neural import build_sequences

HOLDOUT = ["2024-04-01", "2024-07-01"]
SEEDS = [42, 17, 2026]
FILES = ["train.csv", "test_student.csv", "inspection_history.csv", "sample_submission.csv"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def paired_ci(folds, repeats):
    metadata = pd.concat([part for part, _ in folds], ignore_index=True)
    codes, sites = pd.factorize(metadata.site_id, sort=True)
    masks, scorers, offset = [], [], 0
    for part, candidates in folds:
        mask = np.arange(offset, offset + len(part))
        offset += len(part)
        masks.append(mask)
        scorers.append([prepare_weighted_ap(part.target, candidates[name]) for name in ["cb_rank50", "cb_visit_rank3"]])
    rng = np.random.default_rng(20261120)
    deltas = []
    for _ in range(repeats):
        weight = rng.multinomial(len(sites), np.full(len(sites), 1 / len(sites)))[codes]
        deltas.append(np.mean([score[1](weight[mask]) - score[0](weight[mask]) for mask, score in zip(masks, scorers)]))
    return dict(bootstrap=repeats, seed=20261120, unit="site_id", comparison="B minus A",
                ci95=[float(v) for v in np.quantile(deltas, [.025, .975])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/hw4/final-20261120")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--reuse-public-context", type=Path,
                        help="Optional verified all-train context model directory; used only for final seed42, never holdouts")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.reuse_public_context:
        raise ValueError("Synthetic smoke cannot reuse an all-train public model")
    if args.threads < 1 or args.bootstrap < 1:
        raise ValueError("Positive threads/bootstrap required")
    missing = [f for f in FILES if not (args.raw / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Put four CSV directly into {args.raw.resolve()}; missing {missing}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    started = time.monotonic()

    def log(event, **details):
        print(json.dumps(dict(event=event, elapsed_seconds=round(time.monotonic() - started, 3), **details)), flush=True)

    sources = [Path(__file__), *(ROOT / "src/hw4_inspections" / f for f in
               ["features.py", "extensions.py", "neural.py", "classical_features.py", "classical_models.py",
                "classical_analysis.py", "delivery.py", "delivery_neural.py"])]
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    cb_trees, nn_epochs = (5, 1) if args.smoke else (500, 2)
    recipe = {"A": "rank50(mean_probability3seeds(reference346),mean_probability3seeds(compact96))",
              "B": "equal thirds of the two family ranks and legacyVisit346 rank",
              "seeds": SEEDS, "visit_seed": 42, "visit_epochs": nn_epochs,
              "visit_epoch_source": "Frozen median of prior dev5/2/2; no epoch selection on reserved quarters",
              "cb_iterations": cb_trees, "holdout": HOLDOUT,
              "primary_rule": "A default; B only if delta>=0 in BOTH reserved quarters and mean delta>0",
              "extra_exports": "compact/context family means are component diagnostics, not extra holdout-selected finalists",
              "rank_ties": "average percentile within each forecast quarter",
              "new_hyperparameter_or_weight_search": False}
    provenance = {"config": config, "recipe": recipe,
                  "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
                  "input_sha256": {f: sha(args.raw / f) for f in FILES},
                  "versions": {f: importlib.metadata.version(f) for f in ["numpy", "pandas", "torch", "catboost", "scikit-learn", "joblib"]},
                  "machine": platform.machine(), "neural_device": args.device,
                  "catboost_specs": {k: SPECS[k] for k in ["cb_reference", "cb_compact"]}}
    if args.reuse_public_context:
        provenance["reused_public_context_sha256"] = {f: sha(args.reuse_public_context / f)
                                                    for f in ["manifest.json", "model.cbm", "submission_context.csv"]}
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / "frozen_recipe.json"
    if frozen.exists():
        old = json.loads(frozen.read_text())
        if not args.resume or old["signature"] != signature:
            raise ValueError("Use new --out or unchanged --resume; frozen recipe cannot be overwritten")
    else:
        write_json(frozen, {**provenance, "signature": signature, "frozen_before_holdout_scoring": True})
    log("recipe_frozen", signature=signature, cb_trees=cb_trees, visit_epochs=nn_epochs)
    train = pd.read_csv(args.raw / "train.csv")
    test = pd.read_csv(args.raw / "test_student.csv")
    sample = pd.read_csv(args.raw / "sample_submission.csv")
    if not train.row_id.is_unique or not test.row_id.is_unique or set(train.row_id) & set(test.row_id):
        raise ValueError("Invalid official identities")
    if args.smoke:
        train = train.groupby("forecast_period", group_keys=False).head(32).reset_index(drop=True)
    rows = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    history = pd.read_csv(args.raw / "inspection_history.csv")
    log("features_start", rows=len(rows))
    x, cats = build_features(rows, history)
    x = pd.concat([x, build_context_features(rows, history)], axis=1)
    compact = select_feature_columns(x, "compact_context")
    events, valid, sequence_audit = build_sequences(rows, history)
    del history
    if len(x.columns) != 346 or len(compact) != 96:
        raise ValueError("Frozen feature schemas changed")
    log("features_complete", full=346, compact=96, sequence_audit=sequence_audit)
    reused, fitted = 0, 0

    def fit_predict(stage, model_name, seed, fit_idx, eval_idx):
        nonlocal reused, fitted
        job = args.out / stage / f"{model_name}_s{seed}"
        job.mkdir(parents=True, exist_ok=True)
        marker = job / "complete.json"
        if args.resume and marker.exists():
            record = json.loads(marker.read_text())
            if record["signature"] != signature or any(sha(job / p) != h for p, h in record["artifacts"].items()):
                raise ValueError("Changed completed delivery job")
            p = np.load(job / "prediction.npy", allow_pickle=False)
            reused += 1
            log("job_reused", stage=stage, model=model_name, seed=seed)
            return p
        log("fit_start", stage=stage, model=model_name, seed=seed, train_n=len(fit_idx), predict_n=len(eval_idx))
        t0 = time.monotonic()
        source = "new_fit"
        if (stage == "full_refit" and model_name == "cb_reference" and seed == 42
                and args.reuse_public_context is not None):
            from catboost import CatBoostClassifier
            directory = args.reuse_public_context
            stored = json.loads((directory / "manifest.json").read_text())
            if stored["input_sha256"] != provenance["input_sha256"] or stored["train_rows"] != len(train):
                raise ValueError("Existing public model data/train-count mismatch")
            for source_file in ["src/hw4_inspections/features.py", "src/hw4_inspections/extensions.py"]:
                if stored["source_sha256"].get(source_file) != provenance["source_sha256"][source_file]:
                    raise ValueError("Existing public model feature source mismatch")
            model = CatBoostClassifier()
            model.load_model(str(directory / "model.cbm"))
            params = model.get_all_params()
            if model.tree_count_ != cb_trees or params["depth"] != 5 or params["random_seed"] != 42:
                raise ValueError("Existing public model recipe mismatch")
            if abs(params["learning_rate"] - .04) > 1e-7 or params["l2_leaf_reg"] != 10:
                raise ValueError("Existing public model optimizer mismatch")
            bundle = {"family": "catboost", "features": list(x), "cats": cats,
                      "preprocessor": None, "model": model}
            p = predict_model(bundle, x.iloc[eval_idx])
            previous = pd.read_csv(directory / "submission_context.csv", float_precision="round_trip").set_index("row_id")
            np.testing.assert_allclose(p, previous.loc[test.row_id, "target"].to_numpy(), rtol=0, atol=2e-15)
            joblib.dump(bundle, job / "model.joblib")
            restored = predict_model(joblib.load(job / "model.joblib"), x.iloc[eval_idx])
            source = "verified_existing_public_context_refit"
        elif model_name == "visit":
            def progress(event, **detail):
                log(event, stage=stage, model=model_name, seed=seed, **detail)
            bundle = fit_visit(x.iloc[fit_idx], train.iloc[fit_idx].target, cats, events[fit_idx], valid[fit_idx],
                               epochs=nn_epochs, seed=seed, device=args.device, threads=args.threads, log=progress)
            p = predict_visit(bundle, x.iloc[eval_idx], events[eval_idx], valid[eval_idx])
            joblib.dump(bundle, job / "model.joblib")
            restored = predict_visit(joblib.load(job / "model.joblib"), x.iloc[eval_idx], events[eval_idx], valid[eval_idx])
        else:
            spec = dict(SPECS[model_name], max_iterations=cb_trees)
            columns = compact if model_name == "cb_compact" else list(x)
            model_cats = [c for c in cats if c in columns]
            with threadpool_limits(limits=args.threads):
                bundle = fit_model(spec, x.iloc[fit_idx][columns], train.iloc[fit_idx].target,
                                   model_cats, seed, args.threads, iterations=cb_trees)
                p = predict_model(bundle, x.iloc[eval_idx][columns])
                joblib.dump(bundle, job / "model.joblib")
                restored = predict_model(joblib.load(job / "model.joblib"), x.iloc[eval_idx][columns])
        if not np.array_equal(p, restored):
            raise AssertionError("Delivery model reload forward differs")
        np.save(job / "prediction.npy", p, allow_pickle=False)
        record = {"signature": signature, "stage": stage, "model": model_name, "seed": seed,
                  "train_rows": len(fit_idx), "prediction_rows": len(eval_idx), "seconds": time.monotonic() - t0,
                  "loaded_forward_exact": True, "source": source, "artifacts": {f: sha(job / f) for f in ["model.joblib", "prediction.npy"]}}
        write_json(marker, record)
        if source == "new_fit":
            fitted += 1
        else:
            reused += 1
        log("fit_complete", stage=stage, model=model_name, seed=seed, seconds=round(record["seconds"], 2), loaded_forward_exact=True, source=source)
        return p

    def stage_predictions(stage, fit_idx, eval_idx):
        families = {name: {} for name in ["cb_reference", "cb_compact"]}
        for name in families:
            for seed in SEEDS:
                families[name][seed] = fit_predict(stage, name, seed, fit_idx, eval_idx)
        visit = fit_predict(stage, "visit", 42, fit_idx, eval_idx)
        return make_candidates(families["cb_reference"], families["cb_compact"], visit)

    quarter_scores, holdout_predictions, folds = {}, [], []
    # Both recipes are immutable before this first reserved-label score.
    for quarter in HOLDOUT:
        fit_idx = np.flatnonzero(train.forecast_period.lt(quarter))
        eval_idx = np.flatnonzero(train.forecast_period.eq(quarter))
        candidates = stage_predictions("holdout_" + quarter, fit_idx, eval_idx)
        part = train.iloc[eval_idx][["row_id", "site_id", "forecast_period", "target"]].copy()
        quarter_scores[quarter] = {name: float(average_precision_score(part.target, candidates[name]))
                                   for name in ["cb_rank50", "cb_visit_rank3"]}
        for name in ["cb_rank50", "cb_visit_rank3", "context_cb"]:
            prediction = part.assign(model=name, prediction=candidates[name])
            holdout_predictions.append(prediction)
        folds.append((part, candidates))
        log("reserved_quarter_scored", quarter=quarter, **quarter_scores[quarter])
    holdout = pd.concat(holdout_predictions, ignore_index=True)
    holdout.to_csv(args.out / "reserved_predictions.csv", index=False)
    primary = choose_primary(quarter_scores)
    holdout_report = {"quarter_scores": quarter_scores, "primary": primary, **paired_ci(folds, args.bootstrap),
                      "recipes_changed_after_scoring": False, "reserved_quarters_now_consumed": True}
    write_json(args.out / "reserved_report.json", holdout_report)
    log("reserved_evaluation_complete", primary=primary, report=holdout_report)
    fit_idx = np.arange(len(train))
    eval_idx = np.arange(len(train), len(rows))
    candidates = stage_predictions("full_refit", fit_idx, eval_idx)
    files = {}
    for i, name in enumerate(["cb_rank50", "cb_visit_rank3", "compact_cb", "context_cb"], start=1):
        submission = validate_submission(test, sample, candidates[name], expected_rows=2246)
        file = args.out / f"submission_{i:02d}_{name}.csv"
        submission.to_csv(file, index=False)
        files[name] = {"file": file.name, "sha256": sha(file), "rows": len(submission), "primary": name == primary}
    write_json(args.out / "manifest.json", {"signature": signature, "status": "complete", "submissions": files,
               "primary": primary, "smoke": args.smoke, "quality_claim_from_smoke": False,
               "reserved_evaluation": holdout_report, "new_fit_jobs": fitted, "reused_jobs": reused,
               "full_train_rows": len(train), "test_rows": len(test), "test_refit_complete": True,
               "kaggle_uploads": 0, "elapsed_seconds": time.monotonic() - started})
    log("complete", primary=primary, csv=[v["file"] for v in files.values()], new_fits=fitted, reused=reused, kaggle_uploads=0)


if __name__ == "__main__":
    main()
