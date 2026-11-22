"""Verify the small known GPU return bundle without extracting arbitrary paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from safe_archives import read_known_tar


def _existing_matches(directory, name, content):
    """Check existing artifacts without following links or reading special files."""
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError("Refusing a linked or unreadable GPU return artifact") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("Refusing a nonregular GPU return artifact")
    with os.fdopen(descriptor, "rb") as existing:
        info = os.fstat(existing.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != len(content) or existing.read(len(content) + 1) != content:
            raise ValueError("Refusing to replace a different or nonregular GPU return artifact")
    return True


def _write_values(out, values):
    if out.is_symlink():
        raise ValueError("Refusing a linked GPU output directory")
    try:
        out.mkdir(parents=True, exist_ok=True)
        directory = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("Refusing a linked or invalid GPU output directory") from error
    try:
        # Validate every existing destination before creating any new artifact.
        missing = [name for name, content in values.items() if not _existing_matches(directory, name, content)]
        for name in missing:
            try:
                descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o600, dir_fd=directory)
            except FileExistsError:
                # A concurrent unpack may have created the same immutable artifact.
                if not _existing_matches(directory, name, values[name]):
                    raise ValueError("GPU return destination changed during extraction")
                continue
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(values[name])
    finally:
        os.close(directory)


def unpack(archive_path, out):
    allowed = {"bundle-manifest.json", "tabm_results.tar.gz", "tabpfn_results.tar.gz", "tabicl_results.tar.gz"}
    values = read_known_tar(archive_path, allowed, require_exact=False,
                            max_decompressed_bytes=52_000_000, max_member_bytes=20_000_000,
                            max_total_payload_bytes=50_000_000)
    if "bundle-manifest.json" not in values:
        raise ValueError("Missing GPU bundle manifest")
    manifest = json.loads(values["bundle-manifest.json"])
    if (not isinstance(manifest, dict) or manifest.get("format") != "hw4-extended-v100-bundle-1"
            or type(manifest.get("kaggle_uploads")) is not int or manifest["kaggle_uploads"] != 0
            or type(manifest.get("recent_quarter_metrics_computed")) is not int
            or manifest["recent_quarter_metrics_computed"] != 0
            or not isinstance(manifest.get("files"), dict)
            or set(manifest["files"]) != set(values) - {"bundle-manifest.json"}):
        raise ValueError("Unexpected GPU bundle manifest")
    for family in ("tabm", "tabpfn", "tabicl"):
        status = manifest.get(f"{family}_status")
        if status not in ("complete", "failed") or status == "complete" and f"{family}_results.tar.gz" not in values:
            raise ValueError("Missing completed GPU return or invalid family status")
    for name, digest in manifest["files"].items():
        if hashlib.sha256(values[name]).hexdigest() != digest:
            raise ValueError("Corrupt nested archive")
    _write_values(Path(out), values)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(unpack(args.archive, args.out), indent=2))
