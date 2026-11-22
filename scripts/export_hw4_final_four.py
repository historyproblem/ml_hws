"""Package verified HW4 finalists without fitting models or searching weights.

The additive runner selected its one raw CSV. This separate delivery decision
also exports its already evaluated fixed75/25 blend; the original frozen runner
and manifest remain unchanged. The two boosting files are weak reserves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTEXT_SHA = "636f8cbe87367b6df0bae6d9a1d79a57de863bc78a21bc2296f5c2ff85970634"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def immutable_write(path, value):
    if path.exists() and path.read_bytes() != value:
        raise ValueError(f"Refusing to replace different delivery artifact: {path}")
    path.write_bytes(value)


def probabilities(frame, sample):
    if list(frame.columns) != ["row_id", "target"]:
        raise ValueError("Unexpected submission schema")
    if len(frame) != 2246 or not frame.row_id.equals(sample.row_id) or frame.row_id.duplicated().any():
        raise ValueError("Submission differs from official unique sample order")
    values = frame.target.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid probability")
    return values


def parent_run(directory, sample):
    manifest = read_json(directory / "forecast-manifest.json")
    if manifest.get("smoke") or not manifest.get("complete") or manifest.get("new_q2_q3_metrics") is not False:
        raise ValueError("Expected completed real old-development-only parent")
    for name, digest in manifest["output_sha256"].items():
        if sha(directory / name) != digest:
            raise ValueError(f"Changed parent artifact: {directory / name}")
    if len(manifest["files"]) != 1:
        raise ValueError("Expected one qualified original finalist")
    item = manifest["files"][0]
    path = directory / item["csv"]
    if sha(path) != item["sha256"]:
        raise ValueError("Changed parent CSV")
    probabilities(read_csv(path), sample)
    return path, manifest


def eligible_summary(path, model):
    row = read_csv(path).set_index("model").loc[model]
    if not (bool(row.eligible) and row.delta_vs_reference > 0
            and row.nonnegative_quarters >= 2 and row.worst_quarter_delta >= -.005):
        raise ValueError("Candidate fails the existing historical admission rule")
    return {key: (value.item() if hasattr(value, "item") else value)
            for key, value in row.items() if key in {
                "mean_ap", "delta_vs_reference", "ci95_low", "ci95_high",
                "nonnegative_quarters", "worst_quarter_delta", "eligible"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["research", "raw", "context", "out"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    sample, test = read_csv(args.raw / "sample_submission.csv"), read_csv(args.raw / "test_student.csv")
    if len(test) != 2246 or test.row_id.duplicated().any() or set(test.row_id) != set(sample.row_id):
        raise ValueError("Unexpected official test IDs")
    freeze = read_json(args.research / "additive" / "freeze.json")
    if any(sha(args.raw / name) != digest for name, digest in freeze["raw_sha256"].items()):
        raise ValueError("Changed official data")
    if sha(args.context) != CONTEXT_SHA:
        raise ValueError("Context anchor must be the exact submitted Public .34646 file")
    context = probabilities(read_csv(args.context), sample)
    additive, additive_manifest = parent_run(args.research / "additive", sample)
    d6, _ = parent_run(args.research / "d6-ap", sample)
    xgb, _ = parent_run(args.research / "search", sample)
    additive_values = probabilities(read_csv(additive), sample)
    blended = sample.copy()
    blended["target"] = .75 * context + .25 * additive_values
    specs = [
        ("submission_07_context_additive_probability75.csv", "first", blended.to_csv(index=False).encode(),
         "additive", "screen-summary.csv", "additive_logit__blend75", ".75*context04+.25*additive08"),
        ("submission_08_additive_logit.csv", "second_after_public_review", additive.read_bytes(),
         "additive", "screen-summary.csv", "additive_logit__raw", "additive08"),
        ("reserve/submission_09_context_cb_d6_probability75.csv", "weak_reserve", d6.read_bytes(),
         "d6-ap", "confirmed-summary.csv", "cb_plain_d6_inner_ap__blend75", ".75*context04+.25*mean3_CB_d6"),
        ("reserve/submission_10_context_xgb_probability75.csv", "very_weak_reserve", xgb.read_bytes(),
         "search", "confirmed-summary.csv", "xgb_d4__blend75", ".75*context04+.25*mean3_XGB_d4"),
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, priority, content, directory, summary, model, formula in specs:
        stats = eligible_summary(args.research / directory / summary, model)
        path = args.out / name
        path.parent.mkdir(exist_ok=True)
        immutable_write(path, content)
        values = probabilities(read_csv(path), sample)
        if priority == "first" and not np.array_equal(values, .75 * context + .25 * additive_values):
            raise ValueError("Supplemental fixed blend did not round trip exactly")
        entries.append(dict(file=name, priority=priority, sha256=sha(path), rows=2246,
                            model=model, formula=formula, historical_cv=stats,
                            public_ap=None, private_ap=None, status="READY_NOT_SUBMITTED"))
    manifest = dict(
        date="2026-11-22", incumbent_public_ap=.34646, target_public_baseline_last_observed=.35212,
        remaining_attempts_user_reported=4, agent_uploads=0, new_fits=0, new_q2_q3_metrics=False,
        supplemental_delivery_decision="The frozen additive runner selected raw only. After reviewing its already reported raw and fixed75/25 old-dev results, this separate export also delivers the fixed75/25 blend. Both passed the original gate; no new weights or fits. This supplemental output is not original one-CSV-policy compliance or independent confirmation.",
        interpretation="Reused old development quarters; all intervals nominal and unadjusted for model selection. Boosting improvements are tiny, reserves are not four equally strong recommendations. Send07 first and review Public before spending the other attempts.",
        parents={"context04": CONTEXT_SHA, "additive_raw": sha(additive),
                 "additive_manifest": sha(args.research / "additive" / "forecast-manifest.json"),
                 "additive_signature": additive_manifest["signature"], "cb_d6_blend": sha(d6), "xgb_d4_blend": sha(xgb)},
        official_sha256=freeze["raw_sha256"], files=entries,
    )
    immutable_write(args.out / "manifest.json", (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
