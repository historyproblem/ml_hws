"""Bounded safe validation of the six-file native32 archive; only OLD3 scores."""
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
import run_hw4_tabicl_ensemble32 as producer
import run_hw4_tabpfn_v100 as common
from safe_archives import read_known_tar


def read(value):
    return pd.read_csv(BytesIO(value) if isinstance(value, bytes) else value, float_precision="round_trip")


def run(args):
    files = read_known_tar(args.archive, producer.FILES)
    protocol, completion, info = [json.loads(files[n]) for n in ["protocol.json", "completion.json", "modelinfo.json"]]
    if (common.digest({k:v for k,v in protocol.items() if k != "signature"}) != protocol["signature"]
        or completion["signature"] != protocol["signature"] or protocol["format"] != "hw4-tabicl-native32-postpublic-1"
        or protocol["settings"] != producer.SETTINGS or protocol["periods"] != [*common.OLD, "full"]
        or protocol["recent_periods"] != [] or protocol["device"] != "cuda"
        or protocol["checkpoint_sha256"] != producer.original.WEIGHT
        or protocol["versions"]["tabicl"] != "2.2.0" or protocol["versions"]["scikit-learn"] != "1.7.2"
        or completion["status"] != "complete" or completion["jobs"] != 4
        or info["runtime_network_attempts"] != 0 or info["settings"] != producer.SETTINGS
        or protocol["new_2024q2_q3_metrics"] or completion["new_2024q2_q3_metrics"]
        or protocol["kaggle_uploads"] or completion["kaggle_uploads"]): raise ValueError("Wrong protocol/completion")
    if set(protocol["source_sha256"]) != set(producer.SOURCES): raise ValueError("Wrong source set")
    for name, digest in protocol["source_sha256"].items():
        if common.sha(ROOT / name) != digest: raise ValueError("Changed executed source: " + name)
    if protocol["input_sha256"] != common.RAW: raise ValueError("Wrong official data fingerprints")
    for name, digest in common.RAW.items():
        if common.sha(args.raw / name) != digest: raise ValueError("Changed local raw inputs")
    if set(completion["artifacts"]) != set(producer.FILES) - {"completion.json"}: raise ValueError("Wrong members")
    for name, digest in completion["artifacts"].items():
        if hashlib.sha256(files[name]).hexdigest() != digest: raise ValueError("Changed artifact")
    train, test, sample = [read(args.raw / n) for n in ["train.csv", "test_student.csv", "sample_submission.csv"]]
    keys = ["row_id", "site_id", "forecast_period", "target"]
    meta = pd.concat([train.loc[train.forecast_period.eq(q), keys] for q in common.OLD], ignore_index=True)
    causal_rows = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    history = read(args.raw / "inspection_history.csv")
    base, cats = common.build_features(causal_rows, history)
    base = pd.concat([base, common.build_context_features(causal_rows, history)], axis=1)
    canonical, canonical_categories = common.views(base, cats)[common.CASES[0]]
    canonical_columns = list(canonical)
    old, forecast, metrics = [read(files[n]) for n in ["old_oof.csv", "test_predictions.csv", "fold_metrics.csv"]]
    for frame, expected in [(old, meta), (forecast, test[["row_id", "forecast_period"]])]:
        if list(frame) != [*expected, producer.CASE] or not frame.row_id.is_unique: raise ValueError("Bad schema/ID")
        pd.testing.assert_frame_equal(frame[list(expected)], expected, check_dtype=False, check_exact=True)
        p = frame[producer.CASE].to_numpy()
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any(): raise ValueError("Invalid probabilities")
    if completion["old_rows"] != len(old) or completion["test_rows"] != len(test): raise ValueError("Wrong rows")
    registry = info["jobs"]
    if len(registry) != 4 or {r["identity"]["period"] for r in registry} != {*common.OLD, "full"}: raise ValueError("Wrong jobs")
    score_rows = []
    if len(metrics) != 3 or set(metrics.forecast_period) != set(common.OLD): raise ValueError("Wrong score scope")
    for job in registry:
        identity = job["identity"]; period = identity["period"]
        prior = train if period == "full" else train.loc[train.forecast_period.lt(period)]
        prior = prior.sort_values(["forecast_period", "row_id"], kind="stable")
        predicted = forecast if period == "full" else old.loc[old.forecast_period.eq(period)]
        if (identity["signature"] != protocol["signature"] or identity["case"] != producer.CASE
            or identity["train_rows"] != len(prior) or identity["predict_rows"] != len(predicted)
            or identity["training_ids_sha256"] != common.digest(prior.row_id.tolist())
            or identity["prediction_ids_sha256"] != common.digest(predicted.row_id.tolist())
            or identity["training_quarter_counts"] != {q:int(n) for q,n in prior.forecast_period.value_counts().sort_index().items()}
            or identity["features"] != canonical_columns
            or identity["categorical_indices_for_dtype_confirmation"] != canonical_categories
            or set(identity["features"]) & {"row_id", "site_id", "target", "forecast_period", "history_before"}
            or job["n_estimators"] != 32 or job["checkpoint_sha256"] != producer.original.WEIGHT): raise ValueError("Wrong causal job")
        buf = BytesIO(); np.save(buf, predicted[producer.CASE].to_numpy(dtype=np.float64), allow_pickle=False)
        if hashlib.sha256(buf.getvalue()).hexdigest() != job["prediction_sha256"]: raise ValueError("Changed predictions")
        if period == common.OLD[0] and not job["repeated_forward_exact"]: raise ValueError("Missing exact replay")
        if period != "full":
            ap = average_precision_score(predicted.target, predicted[producer.CASE])
            reported = metrics.loc[metrics.forecast_period.eq(period)].iloc[0]
            if (not np.isfinite(reported.ap) or abs(ap - reported.ap) > 1e-12
                or reported.case != producer.CASE or reported.seed != 42
                or reported.n != len(predicted) or reported.positive != int(predicted.target.sum())
                or reported.train_rows != len(prior)): raise ValueError("Wrong OLD AP/counts")
            score_rows.append({"forecast_period": period, "native32_ap": ap})
    # All validation precedes output writes. No archive extraction or pickle loads.
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in files.items(): (args.out / name).write_bytes(value)
    forecast.set_index("row_id").loc[sample.row_id, [producer.CASE]].rename(columns={producer.CASE:"target"}).reset_index().to_csv(args.out / "submission_tabicl_native32.csv", index=False)
    common.write_csv(args.out / "old-metrics-verified.csv", pd.DataFrame(score_rows))
    receipt = {"status":"PASS", "archive_sha256":common.sha(args.archive), "signature":protocol["signature"],
        "old_mean_ap":float(np.mean([r["native32_ap"] for r in score_rows])), "new_recent_scores":0,
        "rows":len(test), "kaggle_uploads":0}
    common.write_json(args.out / "verification.json", receipt); print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "raw", "out"): parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
