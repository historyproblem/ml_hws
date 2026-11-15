"""Snapshot completed validation predictions without weights or cache.

Safe to run while training: result.json is the commit marker, and only
checksum-verified, already committed artifacts are copied into the archive.
Live summary CSVs are advisory; analysis can rebuild them from job records.
"""

import argparse
import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

JOB_FILES = (
    "predictions.npz",
    "selection.csv",
    "refit.csv",
    "segments.csv",
    "calibration.csv",
    "diagnostics.npz",
    "model.json",
)
SUMMARY_FILES = (
    "run_manifest.json",
    "plan.json",
    "status.json",
    "scores.csv",
    "pooled_scores.csv",
    "derived_scores.csv",
    "derived_pooled.csv",
    "parallel_execution.json",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def export_snapshot(root, output):
    root, output = Path(root), Path(output)
    if not (root / "run_manifest.json").is_file():
        raise FileNotFoundError(f"No run manifest in {root}")
    if output.exists():
        raise FileExistsError(f"Choose a new archive path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    receipt = {
        "format_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "completed_jobs": [],
        "unfinished_jobs": [],
        "invalid_jobs": [],
        "sha256": {},
        "verification": "exported committed artifacts only; weights are intentionally omitted",
        "summary_csvs": "live advisory copies; recompute final metrics from predictions and official targets",
    }
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:

        def add(name, data):
            archive.writestr(name, data)
            receipt["sha256"][name] = sha256(data)

        for name in SUMMARY_FILES:
            path = root / name
            if path.is_file():
                add(name, path.read_bytes())
        for directory in sorted((root / "jobs").iterdir()):
            if not directory.is_dir():
                continue
            marker = directory / "result.json"
            if not marker.is_file():
                receipt["unfinished_jobs"].append(directory.name)
                continue
            try:
                raw = marker.read_bytes()
                result = json.loads(raw)
                job = result["job"]
                if job.get("test"):
                    continue
                expected_key = sha256(json.dumps(job, sort_keys=True).encode())[:20]
                if directory.name != expected_key:
                    raise ValueError("Job identifier does not match its specification")
                checks = result["sha256"]
                if "predictions.npz" not in checks:
                    raise ValueError("No committed prediction checksum")
                files = {"result.json": raw}
                for name in JOB_FILES:
                    if name in checks:
                        content = (directory / name).read_bytes()
                        if sha256(content) != checks[name]:
                            raise ValueError(f"Checksum mismatch: {name}")
                        files[name] = content
            except (OSError, ValueError, KeyError, TypeError) as error:
                receipt["invalid_jobs"].append(
                    {"id": directory.name, "error": str(error)}
                )
                continue
            for name, content in files.items():
                add(f"jobs/{directory.name}/{name}", content)
            receipt["completed_jobs"].append(directory.name)
        archive.writestr("export_manifest.json", json.dumps(receipt, indent=2))
    if not receipt["completed_jobs"]:
        temporary.unlink()
        raise ValueError("No verified completed jobs were found")
    temporary.replace(output)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/v100-focused"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        args.output = Path("outputs") / f"intermediate-results-{stamp}.zip"
    receipt = export_snapshot(args.run_dir, args.output)
    print(f"Archive: {args.output.resolve()}")
    print(
        f"Verified jobs: {len(receipt['completed_jobs'])}; unfinished: {len(receipt['unfinished_jobs'])}; invalid: {len(receipt['invalid_jobs'])}"
    )
    print(
        f"Size: {args.output.stat().st_size / 2**20:.2f} MiB; no model weights or cache included"
    )
    return 1 if receipt["invalid_jobs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
