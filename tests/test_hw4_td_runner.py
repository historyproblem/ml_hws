"""Synthetic-only full paths for the two-preset nested TD continuation."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hw4_td_search as runner
from test_hw4_final_research import fixture, expect_error


def td_fixture(root, perfect=False):
    args = fixture(root)
    old = runner.primary.read_csv(args.baseline_cv)
    old["additive_logit__blend75"] = old.target if perfect else .5
    args.additive_oof = root / "previous07-oof.csv"
    old.to_csv(args.additive_oof, index=False)
    args.additive_test = root / "previous07-test.csv"
    runner.primary.read_csv(args.context_test).to_csv(args.additive_test, index=False)
    args.threads = 2
    return args


def test_td_tiny_all_inner_before_outer_gate_full_exact_resume_and_tamper():
    with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
        args = td_fixture(Path(temporary))
        inner, fixed = runner.fit_path, runner.fit_fixed
        counts = {"inner": 0, "fixed": 0}

        def tracked_inner(*a, **kw):
            assert (args.out / "freeze.json").is_file()
            counts["inner"] += 1
            return inner(*a, **kw)

        def tracked_fixed(*a, **kw):
            assert counts["inner"] == 6
            counts["fixed"] += 1
            return fixed(*a, **kw)

        with patch.object(runner, "fit_path", side_effect=tracked_inner), patch.object(
                runner, "fit_fixed", side_effect=tracked_fixed):
            result = runner.run(args)
        assert counts["inner"] == result["inner_fits_total"] == 6
        assert result["screen_eligible"] and result["confirmed_eligible"]
        assert counts["fixed"] == result["outer_full_fits_total"] == 12
        assert result["actual_model_fits_total"] == 18
        assert len(result["files"]) == 1 and result["new_q2_q3_metrics"] is False
        args.resume = True
        with patch.object(runner, "fit_path", side_effect=AssertionError("Resume inner fit")), patch.object(
                runner, "fit_fixed", side_effect=AssertionError("Resume outer fit")):
            resumed = runner.run(args)
        assert resumed["new_inner_fits_this_invocation"] == resumed["new_outer_full_fits_this_invocation"] == 0
        assert resumed["files"] == result["files"]
        curve = next((args.out / "inner").glob("*/curve.json"))
        with curve.open("a") as target:
            target.write(" ")
        expect_error(lambda: runner.run(args))


def test_td_failed07_gate_stops_after_three_outer_fits_without_full():
    with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
        args = td_fixture(Path(temporary), perfect=True)
        result = runner.run(args)
        assert result["inner_fits_total"] == 6 and result["outer_full_fits_total"] == 3
        assert result["actual_model_fits_total"] == 9
        assert not result["screen_eligible"] and not result["confirmed_eligible"]
        assert result["files"] == []


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS", flush=True)
