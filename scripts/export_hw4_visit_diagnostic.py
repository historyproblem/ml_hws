"""Export the already-fitted visit component; no training or holdout scoring."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.delivery import validate_submission


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args()
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frozen = json.loads((args.run / "frozen_recipe.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("smoke"):
        raise ValueError("A completed real delivery is required")
    raw = args.raw or Path(frozen["config"]["raw"])
    for name in ["test_student.csv", "sample_submission.csv"]:
        if sha(raw / name) != frozen["input_sha256"][name]:
            raise ValueError("Official test/sample SHA changed")
    job = args.run / "full_refit/visit_s42"
    record = json.loads((job / "complete.json").read_text())
    prediction_path = job / "prediction.npy"
    if (record["signature"] != manifest["signature"] or not record["loaded_forward_exact"]
            or record["artifacts"]["prediction.npy"] != sha(prediction_path)):
        raise ValueError("Visit prediction artifact verification failed")
    test = pd.read_csv(raw / "test_student.csv")
    sample = pd.read_csv(raw / "sample_submission.csv")
    prediction = np.load(prediction_path, allow_pickle=False)
    submission = validate_submission(test, sample, prediction, expected_rows=2246)
    output = args.run / "submission_05_visit_transformer.csv"
    submission.to_csv(output, index=False)
    proof = {"purpose": "standalone component diagnostic", "selected_finalist": False,
             "parent_manifest_sha256": sha(manifest_path), "prediction_npy_sha256": sha(prediction_path),
             "file": output.name, "sha256": sha(output), "rows": len(submission),
             "new_fits": 0, "new_holdout_scores": 0, "kaggle_uploads": 0,
             "frozen_recipe_changed": False}
    (args.run / "diagnostic_manifest.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
