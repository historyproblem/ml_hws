"""One fixed75/25 mixture substitutes the final code-augmented GAM for G0."""
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
from hw4_inspections.delivery import validate_submission
from hw4_inspections.next_round import QUARTERS, REQUIRED, evaluate_candidates


def run(a):
    sources = [Path(__file__), ROOT / "scripts/evaluate_hw4_fixed_ensembles.py",
               *[ROOT / "src/hw4_inspections" / p for p in ["next_round.py", "classical_analysis.py", "delivery.py"]]]
    plan = json.loads(a.plan.read_text())
    if plan["formula"] != ".75*coded_GAM+.25*canonical_EBM":
        raise ValueError("Changed fixed substitution")
    protocol = dict(input_sha256={p: sha(getattr(a, p)) for p in ["plan", "gam_oof", "coded_oof", "ebm_oof", "coded_test", "ebm_test"]},
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    raw_sha256={p: sha(a.raw / p) for p in ["train.csv", "test_student.csv", "sample_submission.csv", "inspection_history.csv"]},
                    formula=plan["formula"], bootstrap=a.bootstrap, new_fits=0, recent_AP=0, kaggle_uploads=0)
    protocol["signature"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise ValueError("Fresh fixed-substitution output required")
    write_json(a.out / "freeze.json", protocol)
    train, test, sample = [read(a.raw / p) for p in ["train.csv", "test_student.csv", "sample_submission.csv"]]
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(REQUIRED)] for q in QUARTERS], ignore_index=True)
    gam, coded, ebm = aligned(a.gam_oof, meta, "fixed_teacher"), aligned(a.coded_oof, meta, "gam_g0_plus_code_score"), aligned(a.ebm_oof, meta, "ebm_raw")
    predictions = dict(GAM75_EBM25=.75 * gam + .25 * ebm,
                       CODED_GAM75_EBM25=.75 * coded + .25 * ebm)
    result = evaluate_candidates(meta, predictions, reference="GAM75_EBM25", bootstrap=a.bootstrap, seed=20261122)
    summary = result["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(-.005)
    summary.to_csv(a.out / "summary.csv", index=False)
    result["quarter_scores"].to_csv(a.out / "quarter-metrics.csv", index=False)
    pd.concat([meta, pd.DataFrame(predictions)], axis=1).to_csv(a.out / "old-oof.csv", index=False)
    row = summary.set_index("model").loc["CODED_GAM75_EBM25"]
    files = []
    if row.eligible:
        values = []
        for p, c in [(a.coded_test, "gam_g0_plus_code_score"), (a.ebm_test, "prediction")]:
            data = read(p)
            if not data.row_id.is_unique or set(data.row_id) != set(test.row_id):
                raise ValueError("Different full prediction IDs")
            values.append(probability(data.set_index("row_id").loc[test.row_id, c], len(test)))
        output = validate_submission(test, sample, .75 * values[0] + .25 * values[1], 2246)
        path = a.out / "submission_CODED_GAM75_EBM25.csv"
        output.to_csv(path, index=False)
        pd.testing.assert_frame_equal(output, read(path), check_exact=True)
        files = [dict(file=path.name, sha256=sha(path), rows=2246, mean_ap=float(row.mean_ap),
                      delta_vs_prior_mixture=float(row.delta_vs_reference), public_ap=None, status="READY_NOT_SUBMITTED")]
    manifest = dict(complete=True, signature=protocol["signature"], files=files, new_fits=0,
                    new_recent_AP=0, kaggle_uploads=0, nominal_reused_development=True)
    write_json(a.out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ["raw", "out", "plan", "gam-oof", "coded-oof", "ebm-oof", "coded-test", "ebm-test"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    run(parser.parse_args())
