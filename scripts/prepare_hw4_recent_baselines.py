"""Reuse historical context/E checkpoints; prepare recent baselines without AP.

Six saved context CatBoost holdout components and two fixed-E teacher models
are reused and replayed. Only the two missing old-GAM C=.1 fits may be trained.
No full-train model is substituted for a holdout forecast; no metric is scored.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.additive_logit import AdditiveLogitTransform, CLASSIFIER_PARAMS, predict_additive
from hw4_inspections.additive_search import predict_recipe, recipe_by_id
from hw4_inspections.classical_models import _transform
from hw4_inspections.recent_diagnostic import META, QUARTERS

SEEDS = (42, 17, 2026)
RAW = ("train.csv", "test_student.csv", "inspection_history.csv", "sample_submission.csv")
PARENT_SIGNATURE = "38eaf418ff4a66e573285bdbf6a51d6caded2416b183be995fcda5e733499b40"
CORE_SPEC = dict(family="catboost", feature_set="context", add_site=False, recency=False,
                 max_iterations=500, tune_iterations=False,
                 params=dict(depth=5, learning_rate=.04, l2_leaf_reg=10.))


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def probabilities(value, n):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n,) or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("Invalid recent probability vector")
    return value


def check_sources(values):
    for name, expected in values.items():
        path = ROOT / name if "/" in name else ROOT / ("scripts" if name.startswith("run_") else "src/hw4_inspections") / name
        if sha(path) != expected:
            raise ValueError(f"Original source changed: {name}")


def provenance(args):
    parent = read(args.delivery_run / "frozen_recipe.json")
    signed = {k: v for k, v in parent.items() if k not in {"signature", "frozen_before_holdout_scoring"}}
    if (digest(signed) != parent["signature"] or not parent["frozen_before_holdout_scoring"]
            or not args.smoke and parent["signature"] != PARENT_SIGNATURE
            or parent["recipe"]["seeds"] != list(SEEDS) or parent["recipe"]["cb_iterations"] != 500
            or parent["recipe"]["holdout"] != list(QUARTERS)
            or parent["catboost_specs"]["cb_reference"] != CORE_SPEC):
        raise ValueError("Unexpected original context recipe/signature")
    check_sources(parent["source_sha256"])
    raw_hashes = {n: sha(args.raw / n) for n in RAW}
    if parent["input_sha256"] != raw_hashes:
        raise ValueError("Original context data differ from current raw files")
    cache_meta = read(args.feature_cache.with_name("features-meta.json"))
    if cache_meta["cache_sha256"] != sha(args.feature_cache) or cache_meta["raw_sha256"] != raw_hashes:
        raise ValueError("Feature cache is not bound to these official inputs")
    check_sources(cache_meta["builder_sha256"])
    teacher = read(args.teacher_run / "freeze.json")
    if (digest({k: v for k, v in teacher.items() if k != "signature"}) != teacher["signature"]
            or teacher["raw_sha256"] != raw_hashes or teacher["cache_sha256"] != sha(args.feature_cache)
            or teacher["pre_results_plan"]["teacher"]["recipe"] != "core_interactions_C0.01"):
        raise ValueError("Expected the fixed same-data E/C=.01 teacher library")
    check_sources(teacher["source_sha256"])
    checkpoints = {}
    for q in QUARTERS:
        for seed in SEEDS:
            path = args.delivery_run / f"holdout_{q}" / f"cb_reference_s{seed}"
            record = read(path / "complete.json")
            expected = dict(signature=parent["signature"], stage=f"holdout_{q}", model="cb_reference", seed=seed)
            if any(record[k] != v for k, v in expected.items()) or not record["loaded_forward_exact"]:
                raise ValueError("Unexpected old context checkpoint identity")
            if set(record["artifacts"]) != {"model.joblib", "prediction.npy"}:
                raise ValueError("Unexpected original checkpoint payload")
            if any(sha(path / n) != h for n, h in record["artifacts"].items()):
                raise ValueError("Original context payload changed")
            checkpoints[f"context_{q}_{seed}"] = dict(directory=str(path.resolve()),
                marker_sha256=sha(path / "complete.json"), artifacts=record["artifacts"])
        path = args.teacher_run / "teacher" / q
        record = read(path / "complete.json")
        if (record["identity"]["signature"] != teacher["signature"]
                or record["identity"]["recipe"] != "core_interactions_C0.01"
                or record["identity"]["quarter"] != q or record["identity"]["kind"] != "teacher"
                or not record["reload_raw_exact"] or record["evaluation_metrics_computed"] is not False):
            raise ValueError("Unexpected fixed-E teacher checkpoint")
        if set(record["sha256"]) != {"model.joblib", "raw.npy", "row-ids.npy"} or any(sha(path / n) != h for n, h in record["sha256"].items()):
            raise ValueError("Fixed-E payload changed")
        checkpoints[f"teacher_{q}"] = dict(directory=str(path.resolve()), marker_sha256=sha(path / "complete.json"), artifacts=record["sha256"])
    sources = [Path(__file__), *(ROOT / "src/hw4_inspections" / n for n in [
        "additive_logit.py", "additive_search.py", "classical_features.py", "classical_models.py", "recent_diagnostic.py"])]
    return dict(format="hw4-unscored-recent-baselines-1", smoke=args.smoke, raw_sha256=raw_hashes,
        cache_sha256=sha(args.feature_cache), cache_meta_sha256=sha(args.feature_cache.with_name("features-meta.json")),
        parent_signature=parent["signature"], parent_freeze_sha256=sha(args.delivery_run / "frozen_recipe.json"),
        teacher_signature=teacher["signature"], teacher_freeze_sha256=sha(args.teacher_run / "freeze.json"),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        packages={n: metadata.version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib"]},
        quarters=list(QUARTERS), context_seeds=list(SEEDS), context_spec=CORE_SPEC,
        old_gam_classifier=CLASSIFIER_PARAMS, old_gam_training_threads=2,
        fixed_gam_recipe="core_interactions_C0.01", original_checkpoints=checkpoints,
        formula_previous07=".75*context_3seed+.25*old_gam_core", metrics_computed=0, kaggle_uploads=0)


def fit_old_core(frame, target):
    """Same fixed original transform/classifier, using two fit threads."""
    transform = AdditiveLogitTransform().fit(frame)
    model = LogisticRegression(**CLASSIFIER_PARAMS)
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model.fit(transform.transform(frame), target)
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected fixed-GAM target class order")
    return dict(transformer=transform, model=model, classifier=dict(CLASSIFIER_PARAMS),
                original_transform=True, threads=2, iterations=int(model.n_iter_.max()))


def old_core_job(args, signature, quarter, train, frame):
    fit = np.flatnonzero(train.forecast_period.lt(quarter))
    valid = np.flatnonzero(train.forecast_period.eq(quarter))
    identity = dict(signature=signature, quarter=quarter, train_rows=len(fit), predict_rows=len(valid),
                    classifier=CLASSIFIER_PARAMS, fit_max_period=train.forecast_period.iloc[fit].max())
    path = args.out / "old-gam" / quarter
    if path.exists():
        if not args.resume:
            raise ValueError("Existing old-GAM job requires --resume")
        record = read(path / "complete.json")
        if record["identity"] != identity or not record["reload_exact"] or any(sha(path / n) != h for n, h in record["sha256"].items()):
            raise ValueError("Old-GAM checkpoint changed")
        value = np.load(path / "prediction.npy", allow_pickle=False)
        return probabilities(value, len(valid)), record
    bundle = fit_old_core(frame.iloc[fit], train.target.iloc[fit])
    with threadpool_limits(limits=2):
        probability = bundle["model"].predict_proba(bundle["transformer"].transform(frame.iloc[valid]))[:, 1]
    probability = probabilities(probability, len(valid))
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".pending-old-gam-") as temporary:
        temporary = Path(temporary) / "job"
        temporary.mkdir()
        joblib.dump(bundle, temporary / "model.joblib", compress=1)
        loaded = joblib.load(temporary / "model.joblib")
        with threadpool_limits(limits=2):
            replay = loaded["model"].predict_proba(loaded["transformer"].transform(frame.iloc[valid]))[:, 1]
        if not np.array_equal(probability, replay):
            raise ValueError("Reloaded old-GAM forward differs")
        np.save(temporary / "prediction.npy", probability, allow_pickle=False)
        np.save(temporary / "row-ids.npy", train.row_id.iloc[valid].to_numpy(dtype=str), allow_pickle=False)
        record = dict(identity=identity, reload_exact=True, new_fit_on_creation=True, metrics_computed=0,
            iterations=bundle["iterations"], sha256={n: sha(temporary / n) for n in ["model.joblib", "prediction.npy", "row-ids.npy"]})
        write(temporary / "complete.json", record)
        os.rename(temporary, path)
    return probability, record


def run(args):
    plan = provenance(args)
    plan["signature"] = digest(plan)
    signature = plan["signature"]
    frozen = args.out / "freeze.json"
    if frozen.exists():
        if not args.resume or read(frozen) != plan:
            raise ValueError("Changed baseline source/raw/parents; use a new output directory")
    else:
        if any(p.name not in {".lock", "run.log"} for p in args.out.iterdir()):
            raise ValueError("New recent baselines require an empty output directory")
        write(frozen, plan)
    completed = args.out / "manifest.json"
    if completed.exists():
        result = read(completed)
        if result["signature"] != signature or any(sha(args.out / n) != h for n, h in result["output_sha256"].items()):
            raise ValueError("Completed unscored baseline artifacts changed")
        for q, evidence in result["old_gam_jobs"].items():
            directory = args.out / "old-gam" / q
            if sha(directory / "complete.json") != evidence["marker_sha256"] or any(sha(directory / n) != h for n, h in evidence["artifacts"].items()):
                raise ValueError("Completed old-GAM checkpoint changed")
        print(json.dumps(dict(event="reused_complete", new_fits=0, new_metrics=0)), flush=True)
        return result
    cache = joblib.load(args.feature_cache)
    train, test, sample, frame, cats = [cache[n] for n in ["train", "test", "sample", "views", "cats"]]
    frame = frame["base"]
    for value, name in [(train, "train.csv"), (test, "test_student.csv"), (sample, "sample_submission.csv")]:
        pd.testing.assert_frame_equal(value, csv(args.raw / name), check_dtype=False)
    if len(frame) != len(train) + len(test) or len(frame.columns) != 346 or len(cats) != 10:
        raise ValueError("Unexpected original base346 schema")
    if args.smoke:
        if train.groupby("forecast_period").size().max() > 32:
            raise ValueError("Smoke is synthetic-only")
    elif len(train) != 21017 or len(test) != 2246:
        raise ValueError("Official dimensions changed")
    parts, proof, own_jobs = [], [], {}
    for quarter in QUARTERS:
        fit = np.flatnonzero(train.forecast_period.lt(quarter))
        valid = np.flatnonzero(train.forecast_period.eq(quarter))
        part = train.iloc[valid][list(META)].reset_index(drop=True).copy()
        components = []
        for seed in SEEDS:
            path = args.delivery_run / f"holdout_{quarter}" / f"cb_reference_s{seed}"
            marker = read(path / "complete.json")
            if marker["train_rows"] != len(fit) or marker["prediction_rows"] != len(valid):
                raise ValueError("Old context split sizes differ")
            bundle = joblib.load(path / "model.joblib")
            if (bundle["features"] != list(frame) or bundle["cats"] != list(cats)
                    or bundle["seed"] != seed or bundle["iterations"] != 500 or bundle["model"].tree_count_ != 500):
                raise ValueError("Old context schema/seed/tree count differs")
            params = bundle["model"].get_all_params()
            native_expected = dict(depth=5, learning_rate=.04, l2_leaf_reg=10., boosting_type="Plain",
                                   bootstrap_type="MVS", subsample=.8, rsm=1., border_count=254,
                                   grow_policy="SymmetricTree", max_ctr_complexity=4, one_hot_max_size=2)
            for name, value in native_expected.items():
                actual = params[name]
                if isinstance(value, (int, float)):
                    if np.float32(actual) != np.float32(value):
                        raise ValueError(f"Old native context parameter differs: {name}")
                elif actual != value:
                    raise ValueError(f"Old native context parameter differs: {name}")
            value = probabilities(np.load(path / "prediction.npy", allow_pickle=False), len(valid))
            replay = bundle["model"].predict_proba(_transform(bundle, frame.iloc[valid]), thread_count=2)[:, 1]
            if not np.array_equal(value, replay):
                raise ValueError("Old context raw forecast failed exact current-cache replay")
            part[f"context_s{seed}"] = value
            components.append(value)
            proof.append(dict(quarter=quarter, model=f"context_s{seed}", source=str(path.resolve()),
                reused=True, forward_exact=True, train_rows=len(fit), predict_rows=len(valid), native_parameters=params))
        part["context_3seed"] = np.mean(components, axis=0)
        old_core, core_record = old_core_job(args, signature, quarter, train, frame)
        part["old_gam_core"] = old_core
        part["previous07"] = .75 * part.context_3seed + .25 * old_core
        own = args.out / "old-gam" / quarter
        own_jobs[quarter] = dict(marker_sha256=sha(own / "complete.json"), artifacts=core_record["sha256"])
        path = args.teacher_run / "teacher" / quarter
        marker = read(path / "complete.json")
        if (marker["identity"]["train_rows"] != len(fit) or marker["identity"]["predict_rows"] != len(valid)
                or marker["train_max_period"] != train.forecast_period.iloc[fit].max()):
            raise ValueError("Fixed-E teacher split differs")
        ids = np.load(path / "row-ids.npy", allow_pickle=False)
        raw = np.load(path / "raw.npy", allow_pickle=False)
        if not np.array_equal(ids, part.row_id.to_numpy(dtype=str)) or raw.dtype != np.float64 or not np.isfinite(raw).all():
            raise ValueError("Fixed-E identities/logits differ")
        bundle = joblib.load(path / "model.joblib")
        if bundle["recipe"] != recipe_by_id("core_interactions_C0.01"):
            raise ValueError("Fixed-E recipe differs")
        with threadpool_limits(limits=2):
            replay = bundle["model"].decision_function(bundle["transformer"].transform(frame.iloc[valid]))
        if not np.array_equal(raw, replay):
            raise ValueError("Fixed-E raw-logit replay differs")
        value = probabilities(expit(raw), len(valid))
        if not np.array_equal(value, predict_recipe(bundle, frame.iloc[valid])):
            raise ValueError("Fixed-E probability conversion differs")
        part["fixed_gam_interactions_c001"] = value
        proof.append(dict(quarter=quarter, model="fixed_gam_interactions_c001", source=str(path.resolve()),
                          reused=True, forward_exact=True, train_rows=len(fit), predict_rows=len(valid)))
        parts.append(part)
    components = pd.concat(parts, ignore_index=True)
    final = components[[*META, "context_3seed", "old_gam_core", "previous07", "fixed_gam_interactions_c001"]]
    for name, data in [("predictions.csv", final), ("components.csv", components)]:
        temporary = args.out / (name + ".tmp")
        data.to_csv(temporary, index=False)
        pd.testing.assert_frame_equal(data, csv(temporary), check_exact=True, check_dtype=False)
        temporary.replace(args.out / name)
    write(args.out / "reuse-proof.json", dict(checkpoints=proof, metrics_computed=0,
        context_fits=0, fixed_E_fits=0, old_gam_recent_fits=2, recent_rows=len(final),
        interpretation="Unscored, exact-component-derived baselines; use only in the later frozen diagnostic"))
    result = dict(status="COMPLETE_UNSCORED", signature=signature, rows=len(final),
                  quarters=list(QUARTERS), columns=list(final), new_context_fits=0, new_fixed_E_fits=0,
                  old_gam_jobs=own_jobs, new_old_gam_fits=2, metrics_computed=0, kaggle_uploads=0,
                  output_sha256={n: sha(args.out / n) for n in ["freeze.json", "predictions.csv", "components.csv", "reuse-proof.json"]})
    write(completed, result)
    print(json.dumps(dict(event="complete_unscored", rows=len(final), new_old_gam_fits=2, metrics=0)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "feature-cache", "delivery-run", "teacher-run", "out"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
