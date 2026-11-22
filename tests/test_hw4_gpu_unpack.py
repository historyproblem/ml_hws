"""Tiny adversarial GPU bundle checks; no model execution or score calculations."""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import unpack_hw4_gpu_bundle as consumer


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected bundle rejection")


def bundle(path, *, families=("tabm", "tabpfn", "tabicl"), manifest_changes=None, extra=()):
    files = {f"{family}_results.tar.gz": f"synthetic {family}".encode() for family in families}
    manifest = dict(format="hw4-extended-v100-bundle-1", kaggle_uploads=0,
                    recent_quarter_metrics_computed=0,
                    files={name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
                    **{f"{family}_status": "complete" if family in families else "failed"
                       for family in ("tabm", "tabpfn", "tabicl")})
    manifest.update(manifest_changes or {})
    values = {"bundle-manifest.json": json.dumps(manifest).encode(), **files}
    with tarfile.open(path, "w:gz") as archive:
        for name, content in [*values.items(), *extra]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, BytesIO(content))
    return manifest, values


def test_gpu_unpack_complete_partial_and_identical_outputs():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        for families in [("tabm", "tabpfn", "tabicl"), ("tabpfn",), ()]:
            archive, out = root / "return.tar.gz", root / f"out-{len(families)}"
            manifest, values = bundle(archive, families=families)
            assert consumer.unpack(archive, out) == manifest
            assert {path.name: path.read_bytes() for path in out.iterdir()} == values
            fixed_time = 1_000_000_000
            for path in out.iterdir():
                os.utime(path, ns=(fixed_time, fixed_time))
            before = {path.name: (path.stat().st_ino, path.stat().st_mtime_ns) for path in out.iterdir()}
            consumer.unpack(archive, out)
            assert {path.name: (path.stat().st_ino, path.stat().st_mtime_ns) for path in out.iterdir()} == before
        # The producer may preserve an existing return from a failed child.
        bundle(archive, manifest_changes={"tabpfn_status": "failed"})
        consumer.unpack(archive, root / "failed-with-return")


def test_gpu_unpack_rejects_missing_or_invalid_status_and_digest():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive, out = root / "return.tar.gz", root / "out"
        for changes in [{"tabm_status": "complete"}, {"tabm_status": None},
                        {"tabm_status": "unknown"}, {"kaggle_uploads": 1},
                        {"recent_quarter_metrics_computed": 1}, {"files": []}]:
            bundle(archive, families=(), manifest_changes=changes)
            rejects(lambda: consumer.unpack(archive, out))
            assert not out.exists()
        bundle(archive, families=("tabpfn",),
               manifest_changes={"files": {"tabpfn_results.tar.gz": "0" * 64}})
        rejects(lambda: consumer.unpack(archive, out))
        bundle(archive, families=(), manifest_changes={"tabm_status": "failed"})
        # Omission is different from a declared failed child.
        with tarfile.open(archive, "r:gz") as source:
            manifest = json.loads(source.extractfile("bundle-manifest.json").read())
        del manifest["tabm_status"]
        raw = json.dumps(manifest).encode()
        with tarfile.open(archive, "w:gz") as target:
            member = tarfile.TarInfo("bundle-manifest.json"); member.size = len(raw)
            target.addfile(member, BytesIO(raw))
        rejects(lambda: consumer.unpack(archive, out))


def test_gpu_unpack_rejects_duplicate_traversal_link_and_oversized_header():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive, out = root / "return.tar.gz", root / "out"
        for extra in [("tabm_results.tar.gz", b"duplicate"), ("../outside", b"escape")]:
            bundle(archive, extra=[extra])
            rejects(lambda: consumer.unpack(archive, out))
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            with tarfile.open(archive, "w:gz") as target:
                member = tarfile.TarInfo("bundle-manifest.json")
                member.type, member.linkname = kind, "outside"
                target.addfile(member)
            rejects(lambda: consumer.unpack(archive, out))
        # Header alone declares a huge member; a reader must reject before reading it.
        with tarfile.open(archive, "w:gz") as target:
            member = tarfile.TarInfo("bundle-manifest.json")
            member.size = 20_000_001
            target.addfile(member)
        rejects(lambda: consumer.unpack(archive, out))
        assert not out.exists() and not (root / "outside").exists()


def test_gpu_unpack_rejects_destination_links_and_preflights_conflicts():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive = root / "return.tar.gz"
        _, values = bundle(archive)
        target = root / "target"; target.mkdir()
        linked_out = root / "linked-out"; linked_out.symlink_to(target, target_is_directory=True)
        rejects(lambda: consumer.unpack(archive, linked_out))
        assert not list(target.iterdir())
        for existing in (False, True):
            out = root / f"out-{existing}"; out.mkdir()
            outside = root / f"outside-{existing}"
            if existing:
                outside.write_bytes(values["bundle-manifest.json"])
            (out / "bundle-manifest.json").symlink_to(outside)
            rejects(lambda: consumer.unpack(archive, out))
            assert outside.exists() == existing
            assert not (out / "tabm_results.tar.gz").exists()
        conflict = root / "conflict"; conflict.mkdir()
        (conflict / "tabicl_results.tar.gz").write_bytes(b"previous immutable return")
        rejects(lambda: consumer.unpack(archive, conflict))
        assert not (conflict / "bundle-manifest.json").exists()
        special = root / "nonregular"; special.mkdir()
        (special / "bundle-manifest.json").mkdir()
        rejects(lambda: consumer.unpack(archive, special))
        assert not (special / "tabm_results.tar.gz").exists()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS", flush=True)
