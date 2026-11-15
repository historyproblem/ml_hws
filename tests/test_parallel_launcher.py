"""Real spawned-worker tests against the serial numerical implementation."""

import fcntl
import json
import os
import runpy
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from test_suite import synthetic_dataset

from spend_baselines.suite import make_plan, read_config
from spend_baselines.suite_models import Neural


class ParallelLauncherTests(unittest.TestCase):
    def test_coverage_has_one_seed_and_only_small_real_bootstrap_ensembles(self):
        launcher = runpy.run_path(
            "scripts/run_suite_parallel.py", run_name="parallel_helpers"
        )
        config = read_config("configs/v100_suite.json")
        plan = make_plan(config)
        selected = launcher["coverage_jobs"](plan, config, ["2018-10-01"])
        self.assertEqual(len(selected), 14)
        self.assertEqual({j["seed"] for j in selected if j["kind"] == "neural"}, {42})
        self.assertFalse(any(j["family"] == "tcn" for j in selected))
        self.assertEqual(
            {j["mode"] for j in selected if j["kind"] == "bagging"}, {"customer"}
        )
        self.assertEqual(
            {j["member"] for j in selected if j["kind"] == "bagging"}, {0, 1, 2, 3}
        )
        self.assertEqual(
            len(launcher["coverage_jobs"](plan, config, ["2018-10-01", "2018-09-01"])),
            28,
        )
        self.assertEqual(
            len(launcher["coverage_jobs"](plan, config, ["2018-10-01"], 0)), 6
        )

    def test_progress_reads_epoch_and_rows_without_tensor_operations(self):
        launcher = runpy.run_path(
            "scripts/run_suite_parallel.py", run_name="parallel_helpers"
        )
        model = Neural("gru", "direct", {})

        def fit(self):
            epoch = 3
            mass = 128.0
            tx = [None] * 256
            validation = ([], [])
            assert self is model and epoch == 3 and mass > 0 and tx and validation
            return launcher["training_progress"](threading.get_ident())

        progress = fit(model)
        self.assertEqual(
            progress,
            {
                "family": "gru",
                "phase": "selection",
                "epoch": 4,
                "processed_rows": 128,
                "rows": 256,
            },
        )

    def test_short_route_finishes_without_claiming_the_full_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            synthetic_dataset(data)
            config = read_config("configs/v100_suite.json")
            config.update(
                validation_dates=["2018-10-01"],
                smoke_customers=64,
                continue_on_error=False,
            )
            config["neural"].update(max_epochs=2, patience=1, batch_size=64)
            for profile in config["profiles"]:
                profile["hidden_sizes"] = [8, 4]
            config["tabular"]["forest"].update(
                n_estimators=3, max_depth=3, min_samples_leaf=2
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            output = root / "output"
            command = [
                sys.executable,
                "scripts/run_suite_parallel.py",
                "--coverage-only",
                "--config",
                str(config_path),
                "--data-dir",
                str(data),
                "--out-dir",
                str(output),
                "--device",
                "cpu",
                "--threads",
                "1",
                "--workers",
                "1",
                "--kinds",
                "neural",
                "bagging",
                "--progress-seconds",
                "0.01",
            ]
            env = {
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
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
            self.assertIn("Coverage subset: 12/12 completed", result.stdout)
            self.assertIn("PROGRESS worker=", result.stdout)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["selected_completed"], 12)
            self.assertEqual(status["completed"], 12)
            self.assertGreater(status["remaining"], 0)

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
                    "--progress-seconds",
                    "0.01",
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
