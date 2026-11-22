"""Finite old-dev probability/odds blends; no training or recent-quarter scores.

G0 is the already selected fixed E/C.01 GAM. Its old CV is explicitly a
postselection diagnostic, distinct from the earlier prequential tuner CV.
Mixture formulas are supplied by the preregistered finite ensemble plan.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.delivery import validate_submission
from hw4_inspections.next_round import QUARTERS, REQUIRED, evaluate_candidates


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def probability(value, n):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n,) or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("Invalid probability vector")
    return value


def aligned(path, meta, column):
    frame = read(path)
    if not frame.row_id.is_unique or set(frame.row_id) != set(meta.row_id):
        raise ValueError("Old forecast IDs differ from official development")
    frame = frame.set_index("row_id").loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(frame[list(REQUIRED)], meta, check_dtype=False, check_exact=True)
    return probability(frame[column], len(meta))


def formulas(gam, context, provider, name):
    # No ranks, per-quarter choices or continuous weight fitting.
    return {
        "fixed_GAM_E": gam, "context04": context,
        "context75_GAM25": .75 * context + .25 * gam,
        "context50_GAM50": .5 * context + .5 * gam,
        f"{name}__raw": provider,
        f"{name}__GAM75_probability": .75 * gam + .25 * provider,
        f"{name}__GAM75_odds": expit(.75 * logit(np.clip(gam, 1e-6, 1 - 1e-6))
                                     + .25 * logit(np.clip(provider, 1e-6, 1 - 1e-6))),
        f"{name}__context50_GAM25_probability": .5 * context + .25 * gam + .25 * provider,
    }


def run(args):
    plan = json.loads(args.plan.read_text())
    if plan["logit_clip"] != [1e-6, 1 - 1e-6]:
        raise ValueError("Changed planned logit bounds")
    if not args.provider_name.isidentifier():
        raise ValueError("Use a simple unique provider identifier")
    inputs = {name: sha(getattr(args, name)) for name in ["baseline_cv", "gam_oof", "gam_test", "context_test", "provider_oof", "plan", "provider_recipe"]}
    if args.provider_test:
        inputs["provider_test"] = sha(args.provider_test)
    protocol = dict(format="hw4-finite-old-ensembles-1", smoke=args.smoke,
                    raw_sha256={n: sha(args.raw / n) for n in ["train.csv", "test_student.csv", "inspection_history.csv", "sample_submission.csv"]},
                    input_sha256=inputs, source_sha256={"runner": sha(Path(__file__)), **{
                        n: sha(ROOT / "src/hw4_inspections" / n) for n in ["delivery.py", "next_round.py", "classical_analysis.py"]}},
                    versions={n: metadata.version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn"]},
                    provider=args.provider_name, provider_column=args.provider_column,
                    provider_test_column=args.provider_test_column, gam_column=args.gam_column,
                    reference="fixed_GAM_E", bootstrap=args.bootstrap,
                    admission="positive old mean delta, >=2 nonnegative quarters, worst>=-.005 vs fixed_GAM_E",
                    covariance_note="Provider OOF can describe a prequential algorithm while its full recipe was selected by the frozen old-inner rule; G0 is a postselected fixed recipe. None is an independent new holdout.",
                    new_2024q2_q3_metrics=False, new_fits=0, kaggle_uploads=0)
    protocol["signature"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    freeze = args.out / "freeze.json"
    if freeze.exists():
        if not args.resume or json.loads(freeze.read_text()) != protocol:
            raise ValueError("Changed ensemble sources, predictions or settings")
    else:
        if any(args.out.iterdir()):
            raise ValueError("New ensemble analysis requires an empty directory")
        write_json(freeze, protocol)
    train, test, sample = [read(args.raw / n) for n in ["train.csv", "test_student.csv", "sample_submission.csv"]]
    if not args.smoke and (len(train), len(test)) != (21017, 2246):
        raise ValueError("Unexpected official data sizes")
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(REQUIRED)] for q in QUARTERS], ignore_index=True)
    context = aligned(args.baseline_cv, meta, "prediction")
    gam = aligned(args.gam_oof, meta, args.gam_column)
    provider = aligned(args.provider_oof, meta, args.provider_column)
    predictions = formulas(gam, context, provider, args.provider_name)
    result = evaluate_candidates(meta, predictions, reference="fixed_GAM_E", bootstrap=args.bootstrap, seed=20261122)
    summary = result["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(-.005)
    summary = summary.sort_values(["mean_ap", "model"], ascending=[False, True], kind="stable")
    summary.to_csv(args.out / "summary.csv", index=False)
    result["quarter_scores"].to_csv(args.out / "quarter-metrics.csv", index=False)
    pd.concat([meta, pd.DataFrame(predictions)], axis=1).to_csv(args.out / "old-oof.csv", index=False)
    accepted = summary.loc[summary.eligible & summary.model.str.startswith(args.provider_name + "__")]
    files, conditional = [], None
    if len(accepted):
        choice = accepted.iloc[0]
        conditional = dict(provider=args.provider_name, model=choice.model,
                           mean_ap=float(choice.mean_ap), delta_vs_fixed_E=float(choice.delta_vs_reference),
                           provider_recipe_sha256=sha(args.provider_recipe))
        if args.provider_test:
            values = []
            for path, column in [(args.gam_test, "target"), (args.context_test, "target"),
                                 (args.provider_test, args.provider_test_column)]:
                frame = read(path)
                if not frame.row_id.is_unique or set(frame.row_id) != set(test.row_id):
                    raise ValueError("Full forecast IDs differ")
                values.append(probability(frame.set_index("row_id").loc[test.row_id, column], len(test)))
            full = formulas(*values, args.provider_name)[choice.model]
            path = args.out / f"submission_{choice.model}.csv"
            output = validate_submission(test, sample, full, len(test) if args.smoke else 2246)
            output.to_csv(path, index=False)
            pd.testing.assert_frame_equal(output, read(path), check_exact=True)
            files = [dict(file=path.name, sha256=sha(path), rows=len(test), public_ap=None,
                          model=choice.model, mean_ap=float(choice.mean_ap), status="READY_NOT_SUBMITTED")]
    manifest = dict(complete=True, signature=protocol["signature"], files=files,
                    conditional_full_forecast_request=conditional if not args.provider_test else None,
                    no_new_fits=True, new_2024q2_q3_metrics=False, kaggle_uploads=0,
                    statistical_note="Finite predeclared formulas on reused old development; nominal intervals and adaptive provider selection, no recent/Public weight tuning.")
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "out", "plan", "baseline-cv", "gam-oof", "gam-test", "context-test", "provider-oof", "provider-recipe"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--provider-name", required=True)
    parser.add_argument("--provider-column", required=True)
    parser.add_argument("--gam-column", default="fixed_teacher")
    parser.add_argument("--provider-test", type=Path)
    parser.add_argument("--provider-test-column", default="prediction")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
