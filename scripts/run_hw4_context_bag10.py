"""Test one prespecified ten-seed mean of the existing HW4 context classifier."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import run_hw4_final_research as primary
from hw4_inspections.classical_models import predict_model

OLD = (42, 17, 2026)
ADDED = (7, 23, 101, 123, 456, 777, 1000)
NAME = "context_bag10"
PARENT_SIGNATURE = "38eaf418ff4a66e573285bdbf6a51d6caded2416b183be995fcda5e733499b40"
OOF_AUDIT_SHA = "6e302488907e7e06b348f8142fa9c0ee815db59003a8f23e3be9585ffabd8749"
PARAMS = {"depth": 5, "learning_rate": .04, "l2_leaf_reg": 10., "boosting_type": "Plain",
          "bootstrap_type": "MVS", "subsample": .8, "rsm": 1., "border_count": 254,
          "grow_policy": "SymmetricTree"}
PARENT_SOURCES = ("scripts/export_hw4_final.py", *("src/hw4_inspections/" + f for f in
    ["features.py", "extensions.py", "neural.py", "classical_features.py", "classical_models.py",
     "classical_analysis.py", "delivery.py", "delivery_neural.py"]))


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def validate_oof_audit(args, raw_hashes, n_old):
    path = args.baseline_cv.with_name("baseline-audit.json")
    if not args.smoke and primary.sha(path) != OOF_AUDIT_SHA:
        raise ValueError("Unexpected incumbent OOF proof")
    audit = primary.read_json(path)
    if (audit["status"] != "PASS" or audit["seeds"] != list(OLD) or audit["quarters"] != list(primary.QUARTERS)
            or audit["current_official_input_sha256"] != raw_hashes or audit["validation_rows"] != n_old
            or audit["baseline_oof_sha256"] != primary.sha(args.baseline_cv) or not audit["baseline_csv_roundtrip_exact"]
            or not audit["all_effective_parameters_match_except_float_serialization_precision"]):
        raise ValueError("Invalid incumbent OOF proof")
    checks = audit["checkpoint_checks"]
    if (len(checks) != 9 or {(row["seed"], row["quarter"]) for row in checks} !=
            {(seed, q) for seed in OLD for q in primary.QUARTERS} or
            any(not all(row[key] for key in ["saved_oof_forward_exact", "current_prediction_helper_forward_exact",
                                              "feature_schema_exact", "categorical_indices_exact",
                                              "all_parameter_differences_are_identical_float32_values"]) for row in checks)):
        raise ValueError("Missing or unverified individual incumbent OOF checkpoint")
    for file, expected in audit["source_sha256"].items():
        if primary.sha(primary.ROOT / file) != expected:
            raise ValueError(f"Original OOF evidence source changed: {file}")
    return audit, primary.sha(path)


def parent_arrays(args, input_hashes, train, test, frame, cats, anchor_test, n_old):
    root = args.delivery_run
    path = root / "frozen_recipe.json"
    original = primary.read_json(path)
    sig = original["signature"]
    signed = {k:v for k,v in original.items() if k not in {"signature", "frozen_before_holdout_scoring"}}
    count = 3 if args.smoke else 500
    if (digest(signed) != sig or (not args.smoke and sig != PARENT_SIGNATURE) or
            original["input_sha256"] != input_hashes or original["recipe"]["seeds"] != list(OLD)
            or original["recipe"]["cb_iterations"] != count or not original["frozen_before_holdout_scoring"]):
        raise ValueError("Original delivery signature, source data, seeds or fixed tree count changed")
    if set(original["source_sha256"]) != set(PARENT_SOURCES):
        raise ValueError("Missing original delivery sources")
    for name, expected in original["source_sha256"].items():
        if primary.sha(primary.ROOT / name) != expected:
            raise ValueError(f"Original delivery source changed: {name}")
    manifest = primary.read_json(root/"manifest.json")
    if (manifest["signature"] != sig or manifest["status"] != "complete" or
            manifest["submissions"]["context_cb"]["sha256"] != primary.sha(args.context_test) or
            manifest["full_train_rows"] != len(train) or manifest["test_rows"] != len(test)):
        raise ValueError("Original delivery manifest or context04 anchor changed")
    audit, audit_hash = validate_oof_audit(args, input_hashes, n_old)
    legacy = {rec["seed"]:rec for rec in audit["current_full_refit_compatibility"]}
    evid = {"signature":sig, "frozen_recipe_sha256":primary.sha(path), "manifest_sha256":primary.sha(root/"manifest.json"),
            "nine_checkpoint_oof_audit_sha256":audit_hash, "seeds":{}}
    values = []
    for seed in OLD:
        job = root / "full_refit" / f"cb_reference_s{seed}"
        rec = primary.read_json(job/"complete.json")
        expected = {"signature":sig, "stage":"full_refit", "model":"cb_reference", "seed":seed,
                    "train_rows":len(train), "prediction_rows":len(test), "loaded_forward_exact":True}
        if any(rec.get(k) != v for k,v in expected.items()) or set(rec["artifacts"]) != {"model.joblib", "prediction.npy"}:
            raise ValueError(f"Wrong original full-refit identity for seed {seed}")
        for file, recorded in rec["artifacts"].items():
            if primary.sha(job/file) != recorded:
                raise ValueError(f"Original full artifact hash changed: {seed}/{file}")
        if (legacy[seed]["model_sha256"] != rec["artifacts"]["model.joblib"] or
                not legacy[seed]["feature_names_exact"] or not legacy[seed]["categorical_names_exact"]):
            raise ValueError("Incumbent full model differs from incumbent OOF evidence")
        bundle = joblib.load(job/"model.joblib")
        model = bundle["model"]
        if bundle["features"] != list(frame) or bundle["cats"] != cats or bundle["family"] != "catboost":
            raise ValueError("Original full model feature names or categories changed")
        params = model.get_all_params()
        for key, wanted in {**PARAMS, "task_type":"CPU", "loss_function":"Logloss",
                            "random_seed":seed, "iterations":count}.items():
            actual = params.get(key)
            if (not np.isclose(actual, wanted, rtol=0, atol=1e-7) if isinstance(wanted, float) else actual != wanted):
                raise ValueError(f"Original full model parameter differs: {key}")
        if model.tree_count_ != count:
            raise ValueError("Original full model actual fitted tree count differs")
        stored = np.load(job/"prediction.npy", allow_pickle=False)
        replay = predict_model(bundle, frame.iloc[len(train):])
        if stored.dtype != np.float64 or not np.array_equal(stored, replay):
            raise ValueError("Original full model prediction does not replay on official test order")
        values.append(primary.probabilities(stored, len(test)))
        evid["seeds"][str(seed)] = {"full_marker_sha256":primary.sha(job/"complete.json"),
                                   **rec["artifacts"], "replay_exact":True, "params_match":True}
    mean = np.mean(values, axis=0, dtype=np.float64)
    if not np.array_equal(mean, anchor_test):
        raise ValueError("Mean of original three full arrays differs from context04 CSV")
    return mean, evid


def bag_mean(old_mean, additions):
    vectors = [primary.probabilities(v, len(old_mean)) for v in additions]
    if len(vectors) != len(ADDED):
        raise ValueError("All seven prespecified seed forecasts are required")
    return (3.0 * old_mean + np.sum(vectors, axis=0, dtype=np.float64)) / 10.0


def run(args):
    if args.threads < 1 or args.bootstrap < (0 if args.smoke else 1):
        raise ValueError("Invalid thread count or bootstrap repeats")
    count = 3 if args.smoke else 500
    case = {"family":NAME, "spec":{"family":"catboost", "params":PARAMS, "max_iterations":count},
            "view":"base", "inner":False}
    fingerprint = primary.provenance(args, {NAME:case})
    train,test,sample,views,cats = primary.validate_cache(joblib.load(args.feature_cache), args)
    frame = views["base"]
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(primary.REQUIRED)] for q in primary.QUARTERS],
                     ignore_index=True)
    anchor, anchor_test = primary.align_anchors(meta, test, args)
    old_full, parent = parent_arrays(args, fingerprint["raw_sha256"], train, test, frame, cats, anchor_test, len(meta))
    plan = {key:fingerprint[key] for key in ["raw_sha256","source_sha256","cache_sha256","cache_metadata_sha256",
                                           "anchors_sha256","packages","python","system","threads","bootstrap","smoke"]}
    plan.update({"runner_sha256":primary.sha(Path(__file__)), "model":case, "old_seeds":list(OLD),
                 "additional_seeds":list(ADDED), "old_folds":list(primary.QUARTERS), "reference":primary.ANCHOR,
                 "old_full_and_oof_evidence":parent, "old_full_mean_matches_context04":True,
                 "prediction":"(3*verified_mean3 + sum(7 additional probabilities, fixed seed order))/10",
                 "gate":{"mean_delta_exclusive":0, "nonnegative_quarters_at_least":2,
                         "worst_quarter_delta_at_least":primary.WORST_FLOOR},
                 "bootstrap_seed":20261122, "raw_only":True, "seed_subset_search":False,
                 "new_q2_q3_metrics":False, "full_fit_rule":"exactly seven new fits iff old-development gate passes",
                 "interpretation":"Last prespecified fallback on reused development data; site-bootstrap CI is nominal."})
    sig = digest(plan)
    plan["signature"] = sig
    freeze = args.out/"freeze.json"
    if freeze.exists():
        if not args.resume or primary.read_json(freeze) != plan:
            raise ValueError("Changed ensemble freeze or missing --resume")
    else:
        if any(p.name != ".lock" for p in args.out.iterdir()):
            raise ValueError("A new ensemble experiment requires an empty output directory")
        primary.write_json(freeze,plan)
    manifest_path = args.out/"forecast-manifest.json"
    if manifest_path.exists():
        previous = primary.read_json(manifest_path)
        if previous["signature"] != sig or any(primary.sha(args.out/file) != h for file,h in previous["sha256"].items()):
            raise ValueError("Changed completed ensemble outputs")
    def fixed(seed,q):
        values, actual = primary.run_job(args.out,sig,NAME,case,seed,q,count,train,frame,cats,args.threads,args.resume)
        if actual != count:
            raise ValueError("Unexpected saved tree count")
        return values
    added = [np.concatenate([fixed(seed,q) for q in primary.QUARTERS]) for seed in ADDED]
    mean = bag_mean(anchor,added)
    result = primary.evaluate_candidates(meta,{primary.ANCHOR:anchor,NAME:mean},reference=primary.ANCHOR,
                                         bootstrap=args.bootstrap,seed=20261122)
    summary = result["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(primary.WORST_FLOOR)
    primary.write_csv(args.out/"quarter-metrics.csv",result["quarter_scores"])
    primary.write_csv(args.out/"summary.csv",summary)
    primary.write_csv(args.out/"old-oof.csv",meta.assign(context_3seed=anchor,context_bag10=mean))
    accepted = bool(summary.set_index("model").at[NAME,"eligible"])
    files = []
    if accepted:
        final = bag_mean(old_full,[fixed(seed,"full") for seed in ADDED])
        submission = primary.validate_submission(test,sample,final,len(test) if args.smoke else 2246)
        path = args.out/"submission_context_bag10.csv"
        primary.write_csv(path,submission)
        pd.testing.assert_frame_equal(submission,primary.read_csv(path),check_exact=True)
        files.append({"file":path.name,"sha256":primary.sha(path),"rows":len(submission)})
    outputs = [p for p in args.out.iterdir() if p.suffix in {".csv",".json"} and p != manifest_path]
    manifest = {"signature":sig,"status":"complete","eligible":accepted,"files":files,"new_q2_q3_metrics":False,
                "uploads":0,"smoke":args.smoke,"planned_completed_old_fold_jobs":21,"completed_full_jobs":7 if accepted else 0,
                "additional_seeds":list(ADDED),"reused_incumbent_seeds":list(OLD),"sha256":{p.name:primary.sha(p) for p in outputs}}
    primary.write_json(manifest_path,manifest)
    print(json.dumps({"event":"bag10_complete","eligible":accepted,"files":files}),flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw","out","feature-cache","baseline-cv","context-test","delivery-run"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--smoke",action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
