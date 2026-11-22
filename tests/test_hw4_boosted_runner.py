"""Synthetic-only nested-selection, reuse, gates and resume checks."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hw4_boosted_search as runner
from test_hw4_final_research import fixture, expect_error


def test_nested_recipe_selection_uses_inner_and_full_count_of_one_recipe():
    recipes = deepcopy(runner.RECIPES)
    records = {}
    for name in recipes:
        records[name] = {q: {"best_count": 250, "inner_quarter": "earlier",
                             "curve": [{"count": 250, "ap": .05, "logloss": .6}]}
                         for q in runner.primary.QUARTERS}
    first, later = "gbdt_full_01", "gbdt_full_02"
    for q, n, a, b in zip(runner.primary.QUARTERS, [100, 500, 1000], [.9, .1, .1], [.5, .6, .6]):
        records[first][q]["curve"][0]["ap"] = a
        records[later][q] = {"best_count": n, "inner_quarter": "earlier",
                              "curve": [{"count": n, "ap": b, "logloss": .5}]}
    selected = runner.select_inner(records, recipes)
    assert selected["outer"]["lgb_gbdt_tuned"][runner.primary.QUARTERS[0]]["recipe"] == first
    assert selected["outer"]["lgb_gbdt_tuned"][runner.primary.QUARTERS[1]]["recipe"] == later
    assert selected["final"]["lgb_gbdt_tuned"]["recipe"] == later
    assert selected["final"]["lgb_gbdt_tuned"]["count"] == 500
    assert selected["final"]["lgb_gbdt_tuned"]["old_inner_counts"] == [100, 500, 1000]
    assert selected["final"]["pooled_tuned"]["recipe"] == later


def test_nested_confirmation_avoids_duplicate_predictors_and_caps_at_two():
    choices = [{"algorithm": name} for name in ["best", "same", "different", "fourth"]]
    predictions = {"best": np.array([.1, .9]), "same": np.array([.1, .9]),
                   "different": np.array([.2, .8]), "fourth": np.array([.3, .7])}
    selected, rejected = runner.choose_confirmation(choices, predictions)
    assert [row["algorithm"] for row in selected] == ["best", "different"]
    assert len(rejected) == 2


def test_nested_native_tiny_pipeline_all_inner_before_outer_resume_and_tamper():
    with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
        args = fixture(Path(temporary))
        args.threads = 2
        path = runner.fit_path
        fixed = runner.fit_fixed
        observed = {"inner": 0, "fixed": 0}

        def counted_path(*a, **kw):
            assert (args.out / "freeze.json").is_file()
            observed["inner"] += 1
            return path(*a, **kw)

        def counted_fixed(*a, **kw):
            assert observed["inner"] == 72, "No new outer fit may start before all72 inner paths"
            observed["fixed"] += 1
            return fixed(*a, **kw)

        with patch.object(runner, "fit_path", side_effect=counted_path), patch.object(
                runner, "fit_fixed", side_effect=counted_fixed):
            manifest = runner.run(args)
        assert observed["inner"] == manifest["inner_max_fits_total"] == 72
        assert observed["fixed"] == manifest["outer_full_fits_total"]
        assert manifest["complete"] and manifest["new_q2_q3_metrics"] is False
        assert len(manifest["confirmed_algorithms"]) <= 2
        assert manifest["actual_model_fits_total"] == 72 + observed["fixed"]
        assert observed["fixed"] <= 27
        assert manifest["final_files"], "The synthetic monotone fixture should exercise gated full fitting"
        selection = runner.primary.read_json(args.out / "inner-selection.json")
        for row in manifest["final_files"]:
            assert row["final_inner_selected_recipe"] == selection["final"][row["algorithm"]]
        args.resume = True
        with patch.object(runner, "fit_path", side_effect=AssertionError("Resume inner fit")), patch.object(
                runner, "fit_fixed", side_effect=AssertionError("Resume outer fit")):
            resumed = runner.run(args)
        assert resumed["new_inner_fits_this_invocation"] == resumed["new_outer_full_fits_this_invocation"] == 0
        assert resumed["final_files"] == manifest["final_files"]
        file = next((args.out / "inner").glob("*/curve.json"))
        with file.open("a") as output:
            output.write(" ")
        expect_error(lambda: runner.run(args))


def test_nested_failed_gate_does_not_start_confirmation_or_full_fit():
    with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
        args = fixture(Path(temporary), perfect=True)
        result = runner.run(args)
        assert result["inner_max_fits_total"] == 72
        assert result["confirmed_algorithms"] == [] and result["final_files"] == []
        assert result["outer_full_fits_total"] <= 9


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS", flush=True)
