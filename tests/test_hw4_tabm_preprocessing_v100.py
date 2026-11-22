"""Synthetic native CPU QA of the CUDA producer's complete 63-job protocol."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hw4_tabm_preprocessing_v100 as runner
from test_hw4_additive_runner import fixture


def test_tabm_gpu_producer_native63jobs_archive_resume_and_tamper():
    with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
        args = fixture(Path(temporary))
        args.device = "cpu"
        args.max_epochs = args.patience = args.threads = 1
        result = runner.run(args)
        assert result["status"] == "complete" and result["jobs"] == result["expected_jobs"] == 63
        assert result["old_rows"] == 48 and result["test_rows"] == 16
        assert result["recent_rows"] == 32 and result["recent_scores_computed"] == 0
        with tarfile.open(args.out / "return_results.tar.gz", "r:gz") as archive:
            assert set(archive.getnames()) == set(runner.RETURN_FILES)
            assert all(item.isfile() for item in archive.getmembers())
        args.resume = True
        with patch.object(runner, "fit_bounded", side_effect=AssertionError("Unexpected resume fit")):
            assert runner.run(args) == result
        file = next((args.out / "jobs").glob("*/prediction.npy"))
        file.write_bytes(file.read_bytes() + b"tamper")
        with patch.object(runner, "fit_bounded", side_effect=AssertionError("Unexpected fit")):
            try:
                runner.run(args)
                raise AssertionError("Changed checkpoint accepted")
            except ValueError:
                pass


def test_tabm_gpu_producer_refuses_real_cpu_and_cuda_fallback():
    with TemporaryDirectory() as temporary:
        args = fixture(Path(temporary))
        args.device = "cpu"
        args.smoke = False
        args.max_epochs = args.patience = args.threads = 1
        try:
            runner.run(args)
            raise AssertionError("Real silent CPU use accepted")
        except ValueError:
            pass
        args.device = "cuda"
        with patch.object(runner.torch.cuda, "is_available", return_value=False):
            try:
                runner.run(args)
                raise AssertionError("CUDA fallback accepted")
            except RuntimeError:
                pass
