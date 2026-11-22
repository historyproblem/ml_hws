"""Packaging survives a failed second GPU experiment without dropping results."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


def test_extended_gpu_bundle_success_and_partial_failure():
    for fail in [None, "tabpfn", "tabicl", "tabm_preprocessing"]:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copyfile(ROOT / "scripts/run_hw4_extended_v100.sh", scripts / "run_hw4_extended_v100.sh")
            for name in ["tabm_preprocessing", "tabpfn", "tabicl"]:
                contents = "#!/usr/bin/env bash\nset -euo pipefail\n"
                if fail == name:
                    contents += "exit 23\n"
                else:
                    contents += 'mkdir -p "$1"\nprintf "synthetic archive fixture" > "$1/return_results.tar.gz"\n'
                (scripts / f"run_hw4_{name}_v100.sh").write_text(contents)
            env = {**os.environ, "HW4_PYTHON": sys.executable}
            result = subprocess.run(["bash", str(scripts / "run_hw4_extended_v100.sh"), "analysis/run", "data"],
                                    cwd=root, env=env, capture_output=True, text=True)
            assert result.returncode == (1 if fail else 0), result.stderr
            archive = root / "analysis/run/return_results.tar.gz"
            with tarfile.open(archive, "r:gz") as bundle:
                names = set(bundle.getnames())
                expected = {"bundle-manifest.json"}
                if fail != "tabm_preprocessing":
                    expected.add("tabm_results.tar.gz")
                if fail != "tabpfn":
                    expected.add("tabpfn_results.tar.gz")
                if fail != "tabicl":
                    expected.add("tabicl_results.tar.gz")
                assert names == expected
                manifest = json.loads(bundle.extractfile("bundle-manifest.json").read())
                assert manifest["tabpfn_status"] == ("failed" if fail == "tabpfn" else "complete")
                assert manifest["tabicl_status"] == ("failed" if fail == "tabicl" else "complete")
                assert manifest["tabm_status"] == ("failed" if fail == "tabm_preprocessing" else "complete")
                assert manifest["recent_quarter_metrics_computed"] == 0
                for name, digest in manifest["files"].items():
                    assert hashlib.sha256(bundle.extractfile(name).read()).hexdigest() == digest
