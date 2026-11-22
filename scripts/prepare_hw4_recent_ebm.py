"""Unscored Q2/Q3 forecasts of the frozen final canonical-pairs EBM recipe.

The six recent models fit strictly past official rows. Native random validation
and eight bags remain confined to those past fit pools; no external evaluation
frame, hyperparameter selection, blend or AP calculation is performed here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import prepare_hw4_recent_baselines as common
from hw4_inspections.ebm_canonical import CANONICAL_RECIPES, fit_canonical, predict_canonical
from hw4_inspections.ebm_search import load_runtime
from hw4_inspections.recent_diagnostic import META, QUARTERS

SEEDS = (42, 17, 2026)
RECIPE = "ebm_canonical_pairs10"
sha, digest, read, write, csv = common.sha, common.digest, common.read, common.write, common.csv


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def provenance(args):
    parent = read(args.parent_run / "freeze.json")
    selection = read(args.parent_run / "inner-selection.json")
    manifest = read(args.parent_run / "forecast-manifest.json")
    if (digest({k: v for k, v in parent.items() if k != "signature"}) != parent["signature"]
            or selection["final"]["recipe"] != RECIPE or manifest["final_recipe"] != selection["final"]
            or manifest["signature"] != parent["signature"] or not manifest["complete"]):
        raise ValueError("Expected a completed same-signature fixed final-pairs recipe")
    common.check_sources(parent["source_sha256"])
    expected = normalized(CANONICAL_RECIPES[RECIPE])
    recipe = parent["recipes"][RECIPE]
    if not args.smoke and recipe != expected:
        raise ValueError("Parent final recipe differs from the canonical2000/8bag/ES50 specification")
    if recipe["id"] != RECIPE or recipe["view"] != "compact94":
        raise ValueError("Unexpected final recipe identity")
    raw = {n: sha(args.raw / n) for n in common.RAW}
    if raw != parent["raw_sha256"] or sha(args.feature_cache) != parent["cache_sha256"]:
        raise ValueError("Recent EBM raw/cache differ from the frozen parent")
    cache_meta = read(args.feature_cache.with_name("features-meta.json"))
    if cache_meta["cache_sha256"] != sha(args.feature_cache) or cache_meta["raw_sha256"] != raw:
        raise ValueError("Cache metadata differs")
    common.check_sources(cache_meta["builder_sha256"])
    load_runtime(args.runtime)
    runtime = parent["official_runtime"]
    if metadata.version("interpret-core") != runtime["version"] or runtime["version"] != "0.7.8":
        raise ValueError("Pinned interpret-core0.7.8 is required")
    dist = metadata.distribution("interpret-core")
    record_hash = hashlib.sha256((dist.read_text("RECORD") or "").encode()).hexdigest()
    if record_hash != runtime["distribution_record_sha256"]:
        raise ValueError("Interpret distribution changed")
    for name, value in runtime["native_and_ebm_source_sha256"].items():
        if sha(dist.locate_file(name)) != value:
            raise ValueError("Interpret/native binary source differs from parent")
    parents = {}
    for seed in SEEDS:
        path = args.parent_run / "models" / f"{RECIPE}__full__s{seed}"
        item = read(path / "complete.json")
        identity = item["identity"]
        if (identity["signature"] != parent["signature"] or identity["recipe"] != RECIPE
                or identity["seed"] != seed or identity["period"] != "full" or identity["stage"] != "full"
                or not item["reload_exact"] or set(item["sha256"]) != {"model.joblib", "prediction.npy"}
                or any(sha(path / name) != h for name, h in item["sha256"].items())):
            raise ValueError("Parent full model is incomplete or changed")
        bundle = joblib.load(path / "model.joblib")
        if normalized(bundle["recipe"]) != recipe or bundle["seed"] != seed:
            raise ValueError("Parent fitted model is not the frozen final recipe")
        parents[str(seed)] = dict(directory=str(path.resolve()), marker_sha256=sha(path / "complete.json"),
                                  payload_sha256=item["sha256"], recipe_verified=True)
    sources = [Path(__file__), ROOT / "scripts/prepare_hw4_recent_baselines.py", *(ROOT / "src/hw4_inspections" / n
               for n in ["ebm_canonical.py", "ebm_search.py", "classical_features.py", "recent_diagnostic.py"])]
    return dict(format="hw4-unscored-recent-ebm-1", smoke=args.smoke,
        raw_sha256=raw, cache_sha256=sha(args.feature_cache),
        cache_meta_sha256=sha(args.feature_cache.with_name("features-meta.json")),
        parent_signature=parent["signature"], parent_freeze_sha256=sha(args.parent_run / "freeze.json"),
        parent_selection_sha256=sha(args.parent_run / "inner-selection.json"),
        parent_manifest_sha256=sha(args.parent_run / "forecast-manifest.json"), parent_full_models=parents,
        recipe=recipe, recipe_choice="Previously frozen largest mean OLD INNER AP; no recent choice",
        seeds=list(SEEDS), quarters=list(QUARTERS), runtime=runtime, threads=2,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        versions={n: metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "joblib", "interpret-core"]},
        internal_validation="15% random validation per8 native bags confined to caller's strict PAST fit rows; ES50/max2000, no external q target",
        metrics_computed=0, parameter_search=False, blending=False, kaggle_uploads=0)


def job(args, signature, quarter, seed, recipe, train, frame, cats):
    fit = np.flatnonzero(train.forecast_period.lt(quarter))
    valid = np.flatnonzero(train.forecast_period.eq(quarter))
    identity = dict(signature=signature, quarter=quarter, seed=seed, recipe=RECIPE,
                    train_rows=len(fit), predict_rows=len(valid), train_max_period=train.forecast_period.iloc[fit].max(),
                    training_ids_sha256=digest(train.row_id.iloc[fit].tolist()),
                    prediction_ids_sha256=digest(train.row_id.iloc[valid].tolist()))
    if not len(fit) or not len(valid) or identity["train_max_period"] >= quarter:
        raise ValueError("Noncausal or empty recent split")
    path = args.out / "models" / f"{quarter}__s{seed}"
    if path.exists():
        if not args.resume:
            raise ValueError("Existing recent model requires --resume")
        marker = read(path / "complete.json")
        if marker["identity"] != identity or not marker["reload_exact"] or any(sha(path / n) != h for n, h in marker["sha256"].items()):
            raise ValueError("Changed recent EBM checkpoint")
        return common.probabilities(np.load(path / "prediction.npy", allow_pickle=False), len(valid)), marker
    print(json.dumps(dict(event="recent_fit_start", **identity)), flush=True)
    started = time.monotonic()
    bundle = fit_canonical(deepcopy(recipe), frame.iloc[fit], train.target.iloc[fit], cats,
                           runtime=args.runtime, seed=seed, threads=2)
    value = predict_canonical(bundle, frame.iloc[valid])
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".pending-recent-ebm-") as temporary:
        temporary = Path(temporary) / "model"
        temporary.mkdir()
        joblib.dump(bundle, temporary / "model.joblib", compress=1)
        replay = predict_canonical(joblib.load(temporary / "model.joblib"), frame.iloc[valid])
        if not np.array_equal(value, replay):
            raise ValueError("Recent native EBM reload forward differs")
        np.save(temporary / "prediction.npy", value, allow_pickle=False)
        np.save(temporary / "row-ids.npy", train.row_id.iloc[valid].to_numpy(dtype=str), allow_pickle=False)
        marker = dict(identity=identity, reload_exact=True, metrics_computed=0, new_fit_on_creation=True,
            outer_labels_seen=False, internal_validation_policy=bundle["internal_validation_policy"],
            native_library_sha256=bundle["native_library_sha256"], stage_update_counts=bundle["stage_update_counts"],
            seconds=time.monotonic() - started,
            sha256={n: sha(temporary / n) for n in ["model.joblib", "prediction.npy", "row-ids.npy"]})
        write(temporary / "complete.json", marker)
        os.rename(temporary, path)
    print(json.dumps(dict(event="recent_fit_complete", quarter=quarter, seed=seed,
                          seconds=marker["seconds"], metrics_computed=0)), flush=True)
    return value, marker


def run(args):
    plan = provenance(args)
    plan["signature"] = digest(plan)
    signature = plan["signature"]
    frozen = args.out / "freeze.json"
    if frozen.exists():
        if not args.resume or read(frozen) != plan:
            raise ValueError("Changed recent EBM source, recipe, parents or data")
    else:
        if any(p.name not in {".lock", "run.log"} for p in args.out.iterdir()):
            raise ValueError("A new recent EBM forecast needs an empty directory")
        write(frozen, plan)
    completed = args.out / "manifest.json"
    if completed.exists():
        result = read(completed)
        if result["signature"] != signature or any(sha(args.out / n) != h for n, h in result["output_sha256"].items()):
            raise ValueError("Completed recent forecast changed")
        for name, item in result["models"].items():
            path = args.out / "models" / name
            if sha(path / "complete.json") != item["marker_sha256"] or any(sha(path / n) != h for n, h in item["artifacts"].items()):
                raise ValueError("Completed recent model changed")
        print(json.dumps(dict(event="reused_complete", new_fits=0, metrics_computed=0)), flush=True)
        return result
    cache = joblib.load(args.feature_cache)
    train, test, sample, frame, cats = cache["train"], cache["test"], cache["sample"], cache["views"]["base"], cache["cats"]
    for data, name in [(train, "train.csv"), (test, "test_student.csv"), (sample, "sample_submission.csv")]:
        pd.testing.assert_frame_equal(data, csv(args.raw / name), check_dtype=False)
    if len(frame) != len(train) + len(test) or len(frame.columns) != 346 or len(cats) != 10:
        raise ValueError("Expected frozen base346 cache")
    if args.smoke:
        if train.groupby("forecast_period").size().max() > 32:
            raise ValueError("Smoke is synthetic-only")
    elif len(train) != 21017 or len(test) != 2246:
        raise ValueError("Official data dimensions differ")
    parts, markers = [], {}
    for quarter in QUARTERS:
        valid = np.flatnonzero(train.forecast_period.eq(quarter))
        part = train.iloc[valid][list(META)].reset_index(drop=True).copy()
        vectors = []
        for seed in SEEDS:
            values, marker = job(args, signature, quarter, seed, plan["recipe"], train, frame, cats)
            part[f"ebm_s{seed}"] = values
            vectors.append(values)
            name = f"{quarter}__s{seed}"
            path = args.out / "models" / name
            markers[name] = dict(marker_sha256=sha(path / "complete.json"), artifacts=marker["sha256"])
        part["ebm_fixed_pairs_mean3"] = np.mean(vectors, axis=0)
        parts.append(part)
    components = pd.concat(parts, ignore_index=True)
    prediction = components[[*META, "ebm_fixed_pairs_mean3"]]
    for name, frame in [("components.csv", components), ("predictions.csv", prediction)]:
        temporary = args.out / (name + ".tmp")
        frame.to_csv(temporary, index=False)
        pd.testing.assert_frame_equal(frame, csv(temporary), check_exact=True, check_dtype=False)
        temporary.replace(args.out / name)
    result = dict(status="COMPLETE_UNSCORED", signature=signature, rows=len(prediction),
        new_model_fits=6, models=markers, metrics_computed=0, parameter_search=False, blending=False,
        kaggle_uploads=0, output_sha256={n: sha(args.out / n) for n in ["freeze.json", "components.csv", "predictions.csv"]})
    write(completed, result)
    print(json.dumps(dict(event="complete_unscored", rows=len(prediction), new_fits=6, metrics_computed=0)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "feature-cache", "parent-run", "runtime", "out"]:
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
