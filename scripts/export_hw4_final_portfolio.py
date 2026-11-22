"""Export only named, frozen finalists after the one-shot diagnostic exists."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import audit_hw4_recent_finalists as diagnostic
from hw4_inspections.delivery import validate_submission


def run(a):
    frozen = diagnostic.verify_frozen(a.diagnostic)
    request = frozen["request"]
    signed_inputs = {}
    for definition in request["finalist_definitions"].values():
        for group in ("sources", "old_selection_evidence"):
            for name, digest in definition[group].items():
                name = str(Path(name).resolve())
                if name in signed_inputs and signed_inputs[name] != digest:
                    raise ValueError("Conflicting frozen source hashes")
                signed_inputs[name] = digest
    for path in (a.predictions, a.source_proof):
        name = str(path.resolve())
        if name not in signed_inputs or diagnostic.sha(path) != signed_inputs[name]:
            raise ValueError("Portfolio inputs must be exact pre-diagnostic frozen sources")
    # This verifies existing results and is incapable of re-running a finished
    # diagnostic. Refuse export while results are absent.
    if not (a.diagnostic / "results/complete.json").is_file():
        raise ValueError("Complete the single diagnostic before final portfolio export")
    existing = diagnostic.score(Namespace(out=a.diagnostic, resume=True))
    if existing["status"] != "REUSED_COMPLETE" or existing["new_metrics_computed"] != 0:
        raise ValueError("Portfolio export may only reuse completed statistics")
    proof = diagnostic.read(a.source_proof)
    if (proof["output_sha256"]["test-predictions.csv"] != diagnostic.sha(a.predictions)
            or proof["reference"] != request["reference"] or proof["metrics_computed"] != 0):
        raise ValueError("Full-test matrix differs from its pre-diagnostic preparation proof")
    if not 1 <= len(a.models) <= 4 or len(set(a.models)) != len(a.models):
        raise ValueError("Choose one to four distinct existing frozen IDs")
    if not set(a.models) <= set(request["finalist_ids"]):
        raise ValueError("Model/formula was not frozen before the diagnostic")
    raw = Path(request["raw_directory"])
    test, sample = [diagnostic.csv(raw / name) for name in ["test_student.csv", "sample_submission.csv"]]
    full = diagnostic.csv(a.predictions)
    if not full.row_id.is_unique or set(full.row_id) != set(test.row_id):
        raise ValueError("Full-test IDs are not the official2246")
    if list(full) != ["row_id", *request["finalist_ids"]]:
        raise ValueError("Unexpected final full-test matrix schema")
    full = full.set_index("row_id").loc[test.row_id]
    summary = diagnostic.csv(a.diagnostic / "results/summary.csv").set_index("model")
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise ValueError("Use an empty final portfolio directory")
    entries = []
    for priority, model in enumerate(a.models, start=1):
        probability = full[model].to_numpy(dtype=np.float64)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Invalid final probabilities")
        submission = validate_submission(test, sample, probability, 2246)
        path = a.out / f"submission_{a.start_number + priority - 1:02d}_{model}.csv"
        submission.to_csv(path, index=False)
        pd.testing.assert_frame_equal(submission, diagnostic.csv(path), check_exact=True)
        item = summary.loc[model]
        entries.append(dict(priority=priority, model=model, file=path.name, sha256=diagnostic.sha(path), rows=2246,
                            recipe=request["finalist_definitions"][model],
                            recent_mean_ap=float(item.mean_ap), recent_delta_vs_context=float(item.delta_vs_reference),
                            recent_nominal_ci95=[float(item.ci95_low), float(item.ci95_high)],
                            public_ap=None, private_ap=None, status="READY_NOT_SUBMITTED"))
    result = dict(date="2026-11-22", reference=request["reference"],
                  recent_freeze_signature=frozen["signature"],
                  frozen_full_matrix_sha256=diagnostic.sha(a.predictions),
                  pre_diagnostic_source_proof_sha256=diagnostic.sha(a.source_proof),
                  diagnostic_completion_sha256=diagnostic.sha(a.diagnostic / "results/complete.json"),
                  export_source_sha256=diagnostic.sha(Path(__file__)),
                  new_fits=0, new_metrics_computed=0, new_formulas=0, kaggle_uploads=0,
                  interpretation="Selection among previously frozen candidates after one descriptive temporal diagnostic. Recent quarters were already consumed earlier; intervals nominal. Public/Private unknown.",
                  files=entries)
    diagnostic.write(a.out / "manifest.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False, indent=2))
    for item in entries:
        print(json.dumps({k: v for k, v in item.items() if k != "recipe"}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-proof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--start-number", type=int, default=11)
    run(parser.parse_args())
