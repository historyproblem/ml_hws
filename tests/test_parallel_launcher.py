"""Real spawned-worker tests against the serial numerical implementation."""

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from test_suite import synthetic_dataset


class ParallelLauncherTests(unittest.TestCase):
    def test_parallel_resume_preserves_results_and_matches_serial_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            synthetic_dataset(data)
            resumed, reference = root / "resumed", root / "reference"
            env = {
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }

            def serial(out, limit):
                return [
                    sys.executable,
                    "-m",
                    "spend_baselines.suite",
                    "run",
                    "--smoke",
                    "--device",
                    "cpu",
                    "--threads",
                    "1",
                    "--data-dir",
                    str(data),
                    "--out-dir",
                    str(out),
                    "--resume",
                    "--kinds",
                    "neural",
                    "--limit",
                    str(limit),
                ]

            def parallel(threads=1):
                return [
                    sys.executable,
                    "scripts/run_suite_parallel.py",
                    "--smoke",
                    "--device",
                    "cpu",
                    "--threads",
                    str(threads),
                    "--data-dir",
                    str(data),
                    "--out-dir",
                    str(resumed),
                    "--workers",
                    "2",
                    "--kinds",
                    "neural",
                    "--limit",
                    "4",
                ]

            def run(command):
                result = subprocess.run(
                    command,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout[-6000:])
                return result.stdout

            run(serial(resumed, 2))
            manifest = (resumed / "run_manifest.json").read_bytes()
            # Use parent path in the key: every job's artifact is named model.pt.
            before = {
                str(p): p.stat().st_mtime_ns
                for p in (resumed / "jobs").glob("*/model.pt")
            }
            run(parallel())
            self.assertEqual(manifest, (resumed / "run_manifest.json").read_bytes())
            self.assertTrue(
                all(Path(p).stat().st_mtime_ns == stamp for p, stamp in before.items())
            )
            events = json.loads((resumed / "parallel_execution.json").read_text())
            self.assertEqual(events["started_pending"], 2)
            self.assertEqual(len({event["pid"] for event in events["events"]}), 2)
            run(serial(reference, 4))
            for path in (resumed / "jobs").glob("*/predictions.npz"):
                with (
                    np.load(path) as a,
                    np.load(reference / "jobs" / path.parent.name / path.name) as b,
                ):
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
            stamps = {
                str(p): p.stat().st_mtime_ns
                for p in (resumed / "jobs").glob("*/model.pt")
            }
            output = run(parallel())
            self.assertIn("scheduling 0 jobs", output)
            self.assertTrue(
                all(Path(p).stat().st_mtime_ns == stamp for p, stamp in stamps.items())
            )

            incompatible = subprocess.run(
                parallel(threads=2),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(incompatible.returncode, 0)
            self.assertIn("refusing to reuse", incompatible.stdout)
            with (resumed / "run.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                blocked = subprocess.run(
                    parallel(),
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                    check=False,
                )
                self.assertNotEqual(blocked.returncode, 0)
                self.assertIn("already running", blocked.stdout)
            self.assertTrue(
                all(Path(p).stat().st_mtime_ns == stamp for p, stamp in stamps.items())
            )


if __name__ == "__main__":
    unittest.main()
