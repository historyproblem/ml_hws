"""Synthetic causal fixtures exercise gated native TabM execution and resume."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hw4_tabm_preprocessing as runner
from test_hw4_additive_runner import fixture as causal_fixture


def fixture(root):
    args = causal_fixture(root, perfect=True)
    args.max_epochs = args.patience = args.threads = 1
    args.additive_oof = root / "previous-additive.csv"
    args.additive_test = root / "anchor-test.csv"
    data = runner.read_csv(args.baseline_cv)
    data["additive_logit__blend75"] = data.prediction
    data.to_csv(args.additive_oof, index=False)
    return args


def test_tabm_preprocessing_failed_gate_resume_and_changed_recipe():
    with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
        args = fixture(Path(temporary))
        result = runner.run(args)
        assert result["complete"] and result["source_jobs"] == 18 and not result["files"]
        assert not list((args.out / "jobs").glob("*__full"))
        args.resume = True
        with patch.object(runner, "fit_bounded", side_effect=AssertionError("Unexpected resume fit")):
            assert runner.run(args) == result
        args.max_epochs = 2
        try:
            runner.run(args)
            raise AssertionError("Changed source/config should not resume")
        except ValueError:
            pass


def test_tabm_preprocessing_native_full_path_and_sample_alignment():
    with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
        args = fixture(Path(temporary))
        # Synthetic QA forces only admission, then runs all actual native fits.
        selected = dict(case="tabm_clip_full", variant="additive75", mean_ap=.5, delta_vs_reference=.01)
        with patch.object(runner, "assess", return_value=selected):
            result = runner.run(args)
        assert result["smoke"] and result["source_jobs"] == 27 and len(result["files"]) == 1
        frame = runner.read_csv(args.out / result["files"][0]["csv"])
        sample = runner.read_csv(args.raw / "sample_submission.csv")
        assert frame.row_id.equals(sample.row_id)
        values = np.mean([np.load(args.out / "jobs" / f"tabm_clip_full__full__s{seed}__full" / "prediction.npy")
                          .astype(np.float64) for seed in [42, 17, 2026]], axis=0)
        test = runner.read_csv(args.raw / "test_student.csv")
        expected = runner.primary.validate_submission(test, sample, .75 * .5 + .25 * values, len(test))
        assert np.array_equal(frame.target, expected.target)
        args.resume = True
        with patch.object(runner, "assess", return_value=selected), patch.object(
                runner, "fit_bounded", side_effect=AssertionError("Unexpected resume fit")):
            assert runner.run(args) == result
