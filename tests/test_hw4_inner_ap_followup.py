"""Synthetic-only checks for reusing inner curves and gated fixed outer fits."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hw4_inner_ap_followup as followup
import run_hw4_final_research as primary
from test_hw4_final_research import fixture, expect_error


def curves(case):
    result = {}
    for q, selected, original in zip(primary.QUARTERS, followup.EXPECTED_COUNTS[case], followup.ORIGINAL_COUNTS[case]):
        result[q] = {"inner_quarter": (primary.pd.Period(q, freq="Q") - 1).start_time.strftime("%Y-%m-%d"),
                     "iterations": original,
                     "points": [{"iterations": n, "ap": .3 if n == selected else .2, "logloss": .5}
                                for n in followup.GRID] + [{"iterations": 50, "ap": .99, "logloss": .4}]}
    return result


def synthetic_primary(root, case, perfect=False):
    """Tiny source artifacts stand in for an already completed primary run.

    Their deliberately artificial inner curves are QA data, never evidence of
    model quality. Native outer models still prove serialization and replay.
    """
    args = fixture(root, perfect=perfect)
    pool = primary.cases(False)
    plan = primary.provenance(args, pool)
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["signature"] = signature
    primary.write_json(args.out / "freeze.json", plan)
    train, _, _, views, cats = primary.validate_cache(primary.joblib.load(args.feature_cache), args)
    for q, count in zip(primary.QUARTERS, followup.ORIGINAL_COUNTS[case]):
        primary.run_job(args.out, signature, case, pool[case], 42, q, count, train,
                        views["base"], cats, args.threads, False)
        job = args.out / "jobs" / f"{case}__{q}__s42"
        record = primary.read_json(job / "complete.json")
        record["identity"]["fixed_count"] = None
        record["inner_quarter"] = curves(case)[q]["inner_quarter"]
        primary.write_json(job / "curve.json", curves(case)[q])
        record["sha256"]["curve.json"] = primary.sha(job / "curve.json")
        primary.write_json(job / "complete.json", record)
    args.primary_run = args.out
    args.out = root / f"followup-{case}"
    args.out.mkdir()
    args.case = case
    return args


def test_inner_ap_coarse_only_counts_earliest_tie_and_expected_guard():
    for case in followup.EXPECTED_COUNTS:
        source = curves(case)
        before = deepcopy(source)
        counts, evidence = followup.derive_counts(case, source)
        assert list(counts.values()) == list(followup.EXPECTED_COUNTS[case])
        assert source == before
        assert all(len(row["coarse_inner_ap_points"]) == 4 for row in evidence.values())
        # The higher AP at50 was never a candidate. No prefix hunt is allowed.
        assert 50 not in counts.values()
        q = primary.QUARTERS[0]
        source[q]["points"].append({"iterations": 100, "ap": .9})
        expect_error(lambda: followup.derive_counts(case, source))
        source = curves(case)
        for point in source[q]["points"]:
            if point["iterations"] == 800:
                point["ap"] = .9
        expect_error(lambda: followup.derive_counts(case, source))
    tied = curves("cb_plain_d6")
    # The Q4 expected count100 wins an equal AP tie against the later200.
    q = primary.QUARTERS[1]
    for point in tied[q]["points"]:
        if point["iterations"] == 200:
            point["ap"] = .3
    assert followup.derive_counts("cb_plain_d6", tied)[0][q] == 100


def test_inner_ap_native_synthetic_both_cases_counts_freeze_gate_replay_resume():
    for case, expected_jobs in [("cb_plain_d6", 11), ("xgb_d3_reg", 12)]:
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            args = synthetic_primary(Path(directory), case)
            fit = primary.fit_booster
            seen = []

            def tracking_fit(*fit_args, **fit_kwargs):
                assert (args.out / "freeze.json").is_file(), "Plan must exist before the first new fit"
                seen.append(fit_kwargs["iterations"])
                return fit(*fit_args, **fit_kwargs)

            with patch.object(primary, "fit_booster", side_effect=tracking_fit), patch.object(
                    primary, "choose_tree_count", side_effect=AssertionError("No new inner fits allowed")):
                result = followup.run(args)
            assert len(seen) == expected_jobs
            assert result["new_jobs_this_invocation"] == expected_jobs
            assert set(seen).issubset({100, 200})
            assert result["new_inner_fits"] == result["new_q2_q3_metrics"] == result["uploads"] == 0
            assert result["screen_eligible"] and result["confirmed_eligible"]
            assert result["primary_middle_reused_exact"] == (case == "cb_plain_d6")
            assert len(result["files"]) == 1 and result["files"][0]["full_trees"] == 200
            assert len(list((args.out / "jobs").glob("*/complete.json"))) == expected_jobs
            args.resume = True
            with patch.object(primary, "fit_booster", side_effect=AssertionError("Resume refit")), patch.object(
                    primary, "choose_tree_count", side_effect=AssertionError("Resume inner fit")):
                resumed = followup.run(args)
            assert resumed["new_jobs_this_invocation"] == 0
            assert resumed["files"] == result["files"]
            source_curve = args.primary_run / "jobs" / f"{case}__{primary.QUARTERS[0]}__s42" / "curve.json"
            with source_curve.open("a") as output:
                output.write(" ")
            expect_error(lambda: followup.run(args))


def test_inner_ap_failed_screen_stops_without_confirmation_or_full_fit():
    with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
        args = synthetic_primary(Path(directory), "cb_plain_d6", perfect=True)
        with patch.object(primary, "choose_tree_count", side_effect=AssertionError("New inner fit")):
            result = followup.run(args)
        assert result["new_jobs_this_invocation"] == 2
        assert not result["screen_eligible"] and not result["confirmed_eligible"]
        assert result["files"] == [] and result["primary"] is None
        assert not list(args.out.glob("submission*.csv"))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS", flush=True)
