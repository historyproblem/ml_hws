#!/usr/bin/env python3
"""Export completed final-check forecasts and reports, without cache or model weights."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_final_checks import read_completed
from spend_baselines.suite_data import digest


def export(directory, destination):
    directory, destination = Path(directory), Path(destination)
    paths = []
    for name in ["checks/run_manifest.json", "feature_schema.json", "status.json"]:
        path = directory / name
        if path.is_file():
            paths.append(path)
    paths += sorted((directory / "reports").glob("*.csv"))
    completed, unfinished, invalid = 0, [], []
    for folder in sorted((directory / "checks/jobs").glob("*")):
        marker = folder / "result.json"
        if not marker.exists():
            unfinished.append(folder.name)
            if (folder / "error.json").exists():
                paths.append(folder / "error.json")
            continue
        try:
            record = json.loads(marker.read_text())
            read_completed(folder, record["job"])
        except Exception as error:
            invalid.append({"job": folder.name, "reason": str(error)})
            continue
        completed += 1
        paths.append(marker)
        for name in ["predictions.npz", "selection.csv", "refit.csv"]:
            if (folder / name).is_file():
                paths.append(folder / name)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "completed_jobs": completed, "unfinished_jobs": unfinished, "invalid_jobs": invalid,
                "weights_exported": False, "feature_cache_exported": False,
                "sha256": {str(p.relative_to(directory)): digest(p) for p in paths}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(directory)))
        archive.writestr("export_manifest.json", json.dumps(metadata, indent=2))
    temporary.replace(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final-checks-20261115"))
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    destination = args.destination or args.out_dir / "final-results.zip"
    metadata = export(args.out_dir, destination)
    print(json.dumps({"archive": str(destination), "completed_jobs": metadata["completed_jobs"],
                      "invalid_jobs": len(metadata["invalid_jobs"]), "bytes": destination.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
