"""Validate local V100 foundation-model evidence without unpickling models."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_hw4_tabpfn_v100 as pfn
import run_hw4_tabicl_v100 as icl
from analyze_hw4_tabm_preprocessing import ANCHORS, scores
from hw4_inspections.delivery import validate_submission
from hw4_inspections.next_round import evaluate_candidates
from safe_archives import read_known_tar


def read_csv(source):
    return pd.read_csv(BytesIO(source) if isinstance(source, bytes) else source, float_precision="round_trip")


def read_archive(path, expected):
    return read_known_tar(path, expected, max_decompressed_bytes=32_000_000,
                          max_member_bytes=10_000_000, max_total_payload_bytes=30_000_000)


def npy_digest(values):
    output = BytesIO()
    np.save(output, np.asarray(values, dtype=np.float64), allow_pickle=False)
    return hashlib.sha256(output.getvalue()).hexdigest()


def verify(args):
    module = pfn if args.family == "tabpfn" else icl
    cases = module.CASES
    checkpoint = pfn.CHECKPOINT if args.family == "tabpfn" else icl.WEIGHT
    files = read_archive(args.archive, pfn.RETURN_FILES)
    plan, completion, info = [json.loads(files[n]) for n in ["protocol.json", "completion.json", "modelinfo.json"]]
    body = {k: v for k, v in plan.items() if k != "signature"}
    expected_format = "hw4-tabpfn-v2-offline-1" if args.family == "tabpfn" else "hw4-tabicl-v2-offline-1"
    if (pfn.digest(body) != plan["signature"] or completion["signature"] != plan["signature"]
            or plan["format"] != expected_format or plan["cases"] != list(cases) or plan["seeds"] != [42]
            or plan["old_quarters"] != list(pfn.OLD) or plan["recent_prediction_only_quarters"] != list(pfn.RECENT)
            or plan["smoke"] != args.smoke or not args.smoke and plan["device"] != "cuda"
            or plan["new_2024q2_q3_metrics"] or completion["new_2024q2_q3_metrics"]
            or plan["kaggle_uploads"] or completion["kaggle_uploads"]
            or not plan["external_pretrained_weights"] or plan["checkpoint_sha256"] != checkpoint
            or completion["status"] != "complete" or completion["jobs"] != 12 or completion["expected_jobs"] != 12):
        raise ValueError("Wrong foundation protocol, weights, quarters or completion")
    if args.family == "tabpfn":
        if (plan["context_cap"] != (64 if args.smoke else 10000) or plan["ensemble_estimators"] != 4
                or plan["random_state"] != 42 or not plan["memory_saving_mode"]
                or plan["fit_mode"] != "fit_preprocessors" or plan["inference_precision"] != "torch.float32"
                or plan["versions"]["tabpfn"] != "2.2.1" or plan["versions"]["scikit-learn"] != "1.6.1"
                or plan["tabpfn_wheel_sha256"] != pfn.WHEEL):
            raise ValueError("Unexpected V2 inference settings")
    elif (plan["settings"] != icl.SETTINGS or plan["versions"]["tabicl"] != "2.2.0"
            or plan["versions"]["scikit-learn"] != "1.7.2" or plan["tabicl_wheel_sha256"] != icl.WHEEL):
        raise ValueError("Unexpected TabICL settings")
    if set(plan["source_sha256"]) != set(module.SOURCES):
        raise ValueError("Unexpected foundation source set")
    for name, digest in plan["source_sha256"].items():
        if pfn.sha(ROOT / name) != digest:
            raise ValueError(f"Different executed source: {name}")
    if set(plan["input_sha256"]) != set(pfn.RAW):
        raise ValueError("Missing official data fingerprint")
    for name, digest in plan["input_sha256"].items():
        if pfn.sha(args.raw / name) != digest:
            raise ValueError("Different local/server raw inputs")
    if set(completion["artifacts"]) != set(pfn.RETURN_FILES) - {"completion.json"}:
        raise ValueError("Wrong return artifact set")
    for name, digest in completion["artifacts"].items():
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError("Changed return artifact")
    if info["checkpoint_sha256"] != checkpoint or info["runtime_network_attempts"] != 0:
        raise ValueError("Unexpected model/network metadata")
    if not args.smoke and any(pfn.sha(getattr(args, name)) != digest for name, digest in ANCHORS.items()):
        raise ValueError("Changed local reference predictions")
    train, test, sample = [read_csv(args.raw / n) for n in ["train.csv", "test_student.csv", "sample_submission.csv"]]
    identity = ["row_id", "site_id", "forecast_period", "target"]
    meta = pd.concat([train.loc[train.forecast_period.eq(q), identity] for q in pfn.OLD], ignore_index=True)
    recent_meta = pd.concat([train.loc[train.forecast_period.eq(q), identity] for q in pfn.RECENT], ignore_index=True)
    old, forecast, recent = [read_csv(files[n]) for n in ["old_oof.csv", "test_predictions.csv", "recent_predictions.csv"]]
    for frame, expected in [(old, meta), (forecast, test[["row_id", "forecast_period"]]), (recent, recent_meta)]:
        if (not frame.row_id.is_unique or frame.row_id.isna().any()
                or list(frame) != [*expected.columns, *cases]):
            raise ValueError("Unexpected forecast schema or duplicate IDs")
        pd.testing.assert_frame_equal(frame[list(expected)], expected, check_dtype=False, check_exact=True)
        for case in cases:
            scores(frame[case], len(frame))
    if (completion["old_rows"] != len(old) or completion["test_rows"] != len(test)
            or completion["recent_prediction_rows"] != len(recent)):
        raise ValueError("Wrong forecast sizes")
    registry = info["jobs"]
    expected_jobs = {(c, q) for c in cases for q in (*pfn.OLD, "full", *pfn.RECENT)}
    if len(registry) != 12 or {(r["identity"]["case"], r["identity"]["period"]) for r in registry} != expected_jobs:
        raise ValueError("Missing or duplicate contexts")
    causal_rows = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    history = read_csv(args.raw / "inspection_history.csv")
    base, cats = pfn.build_features(causal_rows, history)
    base = pd.concat([base, pfn.build_context_features(causal_rows, history)], axis=1)
    canonical = pfn.views(base, cats)
    schemas = {case: (list(canonical[pfn_case][0]), canonical[pfn_case][1])
               for case, pfn_case in zip(cases, pfn.CASES)}
    for job in registry:
        item = job["identity"]
        period, case = item["period"], item["case"]
        prior = train if period == "full" else train.loc[train.forecast_period.lt(period)]
        if args.family == "tabpfn":
            if (item["seed"] != 42 or item["n_estimators"] != 4
                    or item["available_train_rows"] != len(prior)
                    or job["actual_ensemble_estimators"] != 4
                    or job["external_base_checkpoint_sha256"] != checkpoint
                    or job["fit_mode"] != "fit_preprocessors"
                    or job["stored_fitted_weights"] is not False):
                raise ValueError("Per-job TabPFN execution settings/checkpoint differ")
        elif job["n_estimators"] != 4 or job["checkpoint_sha256"] != checkpoint:
            raise ValueError("Per-job TabICL execution settings/checkpoint differ")
        chosen = prior.sort_values(["forecast_period", "row_id"], kind="stable")
        if args.family == "tabpfn":
            chosen = chosen.tail(64 if args.smoke else 10000)
        observed = forecast if period == "full" else old.loc[old.forecast_period.eq(period)] if period in pfn.OLD else recent.loc[recent.forecast_period.eq(period)]
        if (item["signature"] != plan["signature"] or item["train_rows"] != len(chosen)
                or item["predict_rows"] != len(observed)
                or item["training_ids_sha256"] != pfn.digest(chosen.row_id.tolist())
                or item["prediction_ids_sha256"] != pfn.digest(observed.row_id.tolist())
                or item["training_quarter_counts"] != {q: int(n) for q, n in chosen.forecast_period.value_counts().sort_index().items()}
                or job["prediction_sha256"] != npy_digest(observed[case])):
            raise ValueError("Context or persisted prediction identity differs")
        columns = item["feature_columns"] if args.family == "tabpfn" else item["features"]
        categories = item["categorical_indices"] if args.family == "tabpfn" else item["categorical_indices_for_dtype_confirmation"]
        if columns != schemas[case][0] or categories != schemas[case][1] or set(columns) & set(identity + ["history_before"]):
            raise ValueError("Feature schema includes identifiers/labels")
        if case == cases[0] and period == pfn.OLD[0] and not job["repeated_forward_exact"]:
            raise ValueError("First real old-fold replay was not exact")
    metrics = read_csv(files["fold_metrics.csv"])
    keys = {(r.case, r.forecast_period) for r in metrics.itertuples()}
    if len(metrics) != 6 or keys != {(c, q) for c in cases for q in pfn.OLD} or not metrics.seed.eq(42).all():
        raise ValueError("Wrong reported score scope")
    for item in metrics.to_dict("records"):
        mask = old.forecast_period.eq(item["forecast_period"])
        y = old.loc[mask, "target"]
        reported_ap = float(item["ap"])
        expected_train_rows = next(job["identity"]["train_rows"] for job in registry
                                  if job["identity"]["case"] == item["case"]
                                  and job["identity"]["period"] == item["forecast_period"])
        if (not np.isfinite(reported_ap) or not 0 <= reported_ap <= 1
                or item["n"] != len(y) or item["positive"] != int(y.sum())
                or item["train_rows"] != expected_train_rows
                or abs(average_precision_score(y, old.loc[mask, item["case"]]) - reported_ap) > 1e-12):
            raise ValueError("Old AP/counts differ from official rows")
    return meta, old, forecast, recent, test, sample, plan


def run(args):
    meta, old, forecast, recent, test, sample, plan = verify(args)
    extra = read_csv(args.additive_oof).set_index("row_id").loc[meta.row_id].reset_index()
    baseline = read_csv(args.baseline_cv).set_index("row_id").loc[meta.row_id].reset_index()
    for frame in [extra, baseline]:
        pd.testing.assert_frame_equal(meta, frame[list(meta)], check_dtype=False)
    anchor, context = scores(extra.additive_logit__blend75, len(meta)), scores(baseline.prediction, len(meta))
    values, lookup = {"previous07": anchor, "context04": context}, {}
    for case in plan["cases"]:
        for variant, prediction in [("raw", old[case]), ("context75", .75 * context + .25 * old[case]),
                                    ("additive75", .75 * anchor + .25 * old[case])]:
            name = f"{case}__{variant}"
            values[name], lookup[name] = prediction, (case, variant)
    result = evaluate_candidates(meta, values, reference="previous07", bootstrap=args.bootstrap, seed=20261122)
    summary = result["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(-.005)
    summary = summary.sort_values(["mean_ap", "model"], ascending=[False, True], kind="stable")
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    result["quarter_scores"].to_csv(args.out / "quarter-metrics.csv", index=False)
    pd.concat([meta, pd.DataFrame(values)], axis=1).to_csv(args.out / "old-oof.csv", index=False)
    recent.to_csv(args.out / "recent-unscored.csv", index=False)
    accepted = summary.loc[summary.eligible & summary.model.isin(lookup)]
    files = []
    if len(accepted):
        chosen = accepted.iloc[0]
        case, variant = lookup[chosen.model]
        prediction = forecast[case].to_numpy()
        if variant != "raw":
            parent = read_csv(args.context_test if variant == "context75" else args.additive_test).set_index("row_id").loc[test.row_id]
            prediction = .75 * scores(parent.target, len(test)) + .25 * prediction
        path = args.out / f"submission_{case}_{variant}.csv"
        output = validate_submission(test, sample, prediction, len(test) if args.smoke else 2246)
        output.to_csv(path, index=False)
        pd.testing.assert_frame_equal(output, read_csv(path), check_exact=True)
        files = [dict(file=path.name, sha256=pfn.sha(path), rows=len(test), mean_ap=float(chosen.mean_ap),
                      delta_vs_previous07=float(chosen.delta_vs_reference), model=chosen.model,
                      public_ap=None, status="READY_NOT_SUBMITTED")]
    proof = dict(status="PASS", family=args.family, smoke=args.smoke, archive_sha256=pfn.sha(args.archive),
                 signature=plan["signature"], checked_contexts=12, old_fold_metrics_recomputed=6,
                 recent_predictions_verified_but_unscored=True, new_2024q2_q3_metrics=False,
                 files=files, kaggle_uploads=0, reference="previous07",
                 statistical_note="Reused old development; nominal uncertainty after adaptive model/variant selection. No recent scores or post-recent tuning.")
    pfn.write_json(args.out / "verification.json", proof)
    print(json.dumps(proof, ensure_ascii=False, indent=2))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["tabpfn", "tabicl"], required=True)
    for name in ["archive", "out", "raw", "baseline-cv", "context-test", "additive-oof", "additive-test"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
