"""One preregistered mean of two fixed GAM75 probability mixtures."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from evaluate_hw4_fixed_ensembles import aligned, probability, read, sha, write_json
from hw4_inspections.next_round import QUARTERS, REQUIRED, evaluate_candidates
from hw4_inspections.delivery import validate_submission


def run(args):
    planned = json.loads(args.plan.read_text())
    if planned["weights"] != dict(fixed_GAM_E=.75, ebm_canonical_mean3=.125, tabicl_own22=.125):
        raise ValueError("Different predefined three-source mixture")
    files = ["plan", "first_oof", "second_oof", "reference_oof", "first_test", "second_test"]
    protocol = dict(input_sha256={n: sha(getattr(args, n)) for n in files},
                    raw_sha256={n: sha(args.raw / n) for n in ["train.csv", "test_student.csv", "sample_submission.csv", "inspection_history.csv"]},
                    source_sha256=sha(Path(__file__)), bootstrap=args.bootstrap,
                    columns=dict(first=args.first_column, second=args.second_column, reference=args.reference_column),
                    formula=".5*(.75G0+.25EBM)+.5*(.75G0+.25TabICL)=.75G0+.125EBM+.125TabICL",
                    new_fits=0, new_recent_AP=0, kaggle_uploads=0)
    protocol["signature"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Use a fresh directory for the fixed composite comparison")
    write_json(args.out / "freeze.json", protocol)
    train, test, sample = [read(args.raw / n) for n in ["train.csv", "test_student.csv", "sample_submission.csv"]]
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(REQUIRED)] for q in QUARTERS], ignore_index=True)
    reference = aligned(args.reference_oof, meta, args.reference_column)
    first, second = aligned(args.first_oof, meta, args.first_column), aligned(args.second_oof, meta, args.second_column)
    predictions = dict(fixed_GAM_E=reference, GAM75_EBM25=first, GAM75_TabICL25=second,
                       GAM75_EBM125_TabICL125=.5 * first + .5 * second)
    results = evaluate_candidates(meta, predictions, reference="fixed_GAM_E", bootstrap=args.bootstrap, seed=20261122)
    summary = results["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(-.005)
    summary.to_csv(args.out / "summary.csv", index=False)
    results["quarter_scores"].to_csv(args.out / "quarter-metrics.csv", index=False)
    pd.concat([meta, pd.DataFrame(predictions)], axis=1).to_csv(args.out / "old-oof.csv", index=False)
    files = []
    row = summary.set_index("model").loc["GAM75_EBM125_TabICL125"]
    if row.eligible:
        test_values = []
        for path in [args.first_test, args.second_test]:
            frame = read(path)
            if not frame.row_id.is_unique or set(frame.row_id) != set(test.row_id):
                raise ValueError("Different component full-forecast IDs")
            test_values.append(probability(frame.set_index("row_id").loc[test.row_id].target, len(test)))
        output = validate_submission(test, sample, .5 * test_values[0] + .5 * test_values[1], 2246)
        path = args.out / "submission_GAM75_EBM125_TabICL125.csv"
        output.to_csv(path, index=False)
        pd.testing.assert_frame_equal(output, read(path), check_exact=True)
        files = [dict(file=path.name, sha256=sha(path), rows=2246, mean_ap=float(row.mean_ap),
                      delta_vs_fixed_GAM_E=float(row.delta_vs_reference), public_ap=None, status="READY_NOT_SUBMITTED")]
    manifest = dict(complete=True, signature=protocol["signature"], files=files,
                    new_fits=0, new_recent_AP=0, kaggle_uploads=0,
                    uncertainty="Nominal reused-development intervals; no recent/Public tuning")
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ["raw", "out", "plan", "first-oof", "second-oof", "reference-oof", "first-test", "second-test"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--first-column", required=True)
    parser.add_argument("--second-column", required=True)
    parser.add_argument("--reference-column", default="fixed_teacher")
    parser.add_argument("--bootstrap", type=int, default=2000)
    run(parser.parse_args())
