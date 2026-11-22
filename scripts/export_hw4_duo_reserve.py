"""Replay the one explicitly admitted post-Public native4/CatBoost odds reserve."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["freeze", "sample", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    if (freeze["previous_strict_gate"] != "FAIL, all3OLD below native4; recordnotreclassified"
            or freeze["native32_substitution_allowed"] or freeze["coefficient_search"]
            or len(freeze["components"]) != 2):
        raise ValueError("Wrong frozen exploratory admission")
    sample = pd.read_csv(args.sample)
    if not sample.row_id.is_unique or len(sample) != 2246:
        raise ValueError("Unexpected official sample identity")
    vectors = []
    for item in freeze["components"]:
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError("Component bytes changed")
        frame = pd.read_csv(path, float_precision="round_trip")
        if (list(frame) != ["row_id", "target"] or not frame.row_id.is_unique
                or set(frame.row_id) != set(sample.row_id)):
            raise ValueError("Component schema or IDs changed")
        value = frame.set_index("row_id").loc[sample.row_id, "target"].to_numpy()
        if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError("Invalid component probabilities")
        vectors.append(logit(np.clip(value, 1e-6, 1-1e-6)))
    result = sample[["row_id"]].copy()
    result["target"] = expit(.5*vectors[0] + .5*vectors[1])
    output_bytes = result.to_csv(index=False).encode()
    if args.out.exists():
        if args.out.read_bytes() != output_bytes:
            raise ValueError("Refusing to replace an existing different artifact")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(output_bytes)
    print(json.dumps({"csv": str(args.out), "sha256": sha(args.out), "rows": len(result),
        "status": "EXPLORATORY_RESERVE_NOT_UPLOADED", "new_fits_or_AP": 0,
        "prior_strict_gate_remains": "FAIL"}))


if __name__ == "__main__":
    main()
