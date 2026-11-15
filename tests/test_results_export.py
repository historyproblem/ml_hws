"""A live export includes only committed, verified prediction artifacts."""

import hashlib
import json
import runpy
import tempfile
import unittest
import zipfile
from pathlib import Path


class ExportTests(unittest.TestCase):
    def test_committed_snapshot_excludes_weights_unfinished_and_corrupt_jobs(self):
        exporter = runpy.run_path(
            "scripts/export_suite_results.py", run_name="export_helpers"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            (root / "run_manifest.json").write_text("{}")
            jobs = root / "jobs"
            jobs.mkdir()
            keys = []
            for seed in [1, 2, 3]:
                job = {"date": "2018-10-01", "seed": seed}
                key = hashlib.sha256(
                    json.dumps(job, sort_keys=True).encode()
                ).hexdigest()[:20]
                keys.append(key)
                directory = jobs / key
                directory.mkdir()
                data = b"prediction fixture"
                (directory / "predictions.npz").write_bytes(data)
                (directory / "model.pt").write_bytes(b"private weights")
                result = {
                    "job": job,
                    "sha256": {
                        "predictions.npz": hashlib.sha256(data).hexdigest(),
                        "model.pt": "unused",
                    },
                }
                if seed != 2:
                    (directory / "result.json").write_text(json.dumps(result))
                if seed == 3:
                    (directory / "predictions.npz").write_bytes(b"corrupt")
            output = Path(tmp) / "snapshot.zip"
            receipt = exporter["export_snapshot"](root, output)
            self.assertEqual(receipt["completed_jobs"], keys[:1])
            self.assertEqual(receipt["unfinished_jobs"], keys[1:2])
            self.assertEqual(receipt["invalid_jobs"][0]["id"], keys[2])
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                self.assertTrue(
                    all(not name.endswith(".pt") for name in archive.namelist())
                )
                self.assertIn(f"jobs/{keys[0]}/predictions.npz", archive.namelist())
                exported = json.loads(archive.read("export_manifest.json"))
                for name, expected in exported["sha256"].items():
                    self.assertEqual(
                        hashlib.sha256(archive.read(name)).hexdigest(), expected
                    )


if __name__ == "__main__":
    unittest.main()
