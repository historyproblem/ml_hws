"""Synthetic TD-preset identity, temporal selection and native-fit checks."""
from copy import deepcopy
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from hw4_inspections.boosted_search import fit_fixed, fit_path, predict
from hw4_inspections.td_presets import TD_GRID, TD_RECIPES, choose_td
from hw4_inspections.next_round import QUARTERS


def expect_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def records():
    result = {name: {} for name in TD_RECIPES}
    # Per-quarter winners differ; final selection must aggregate recipe-specific
    # INNER points, not take the median of per-quarter winning-recipe counts.
    values = {"td_original": [(.41, 100), (.3, 250), (.44, 500)],
              "td_l2_1": [(.4, 250), (.5, 500), (.4, 1000)]}
    for name, points in values.items():
        for q, (ap, count) in zip(QUARTERS, points):
            result[name][q] = {
                "inner_quarter": (pd.Period(q, freq="Q") - 1).start_time.strftime("%Y-%m-%d"),
                "best_count": count,
                "curve": [dict(count=count, ap=ap, logloss=.5)],
                "outer_ap": .999 if name == "td_original" else .01,
            }
    return result


def test_td_recipes_source_constants_and_only_L2_difference():
    assert set(TD_RECIPES) == {"td_original", "td_l2_1"}
    assert TD_GRID == (100, 250, 500, 1000)
    a, b = [TD_RECIPES[name] for name in ["td_original", "td_l2_1"]]
    assert a["family"] == b["family"] == "catboost"
    assert a["view"] == b["view"] == "full" and a["cap"] == b["cap"] == 1000
    assert {name for name in a["params"] if a["params"][name] != b["params"][name]} == {"l2_leaf_reg"}
    p = a["params"]
    assert p["l2_leaf_reg"] == 1e-5 and b["params"]["l2_leaf_reg"] == 1
    assert p["depth"] == 7 and p["learning_rate"] == .08 and p["leaf_estimation_iterations"] == 1
    assert p["one_hot_max_size"] == 15 and p["border_count"] == 254
    assert p["bootstrap_type"] == "Bernoulli" and p["subsample"] == .9
    assert p["boosting_type"] == "Plain" and p["rsm"] == 1 and p["random_strength"] == .8


def test_td_selector_uses_only_own_previous_inner_and_correct_final_recipe_counts():
    source = records()
    before = deepcopy(source)
    selected = choose_td(source)
    assert source == before
    assert [selected["outer"][q]["recipe"] for q in QUARTERS] == ["td_original", "td_l2_1", "td_original"]
    assert [selected["outer"][q]["count"] for q in QUARTERS] == [100, 500, 500]
    assert selected["final"]["recipe"] == "td_l2_1"
    assert selected["final"]["old_inner_counts"] == [250, 500, 1000]
    assert selected["final"]["count"] == 500
    np.testing.assert_allclose(selected["final"]["mean_old_inner_best_ap"], (.4+.5+.4)/3)
    source["td_original"][QUARTERS[-1]]["outer_ap"] = 0
    source["td_l2_1"][QUARTERS[-1]]["outer_ap"] = 1
    assert choose_td(source) == selected


def test_td_selector_ties_and_temporal_schema_score_guards():
    source = records()
    q = QUARTERS[0]
    for name in source:
        source[name][q]["curve"] = [dict(count=100, ap=.4, logloss=.5), dict(count=250, ap=.4, logloss=.4)]
        source[name][q]["best_count"] = 100
    # Within a recipe backend's earliest count wins AP ties, not lowerlogloss.
    assert choose_td(source)["outer"][q]["recipe"] == "td_l2_1"
    changed = deepcopy(source)
    changed["td_original"][q]["inner_quarter"] = q
    expect_error(lambda: choose_td(changed))
    changed = deepcopy(source)
    changed["td_original"][q]["best_count"] = 250
    expect_error(lambda: choose_td(changed))
    changed = records()
    del changed["td_original"][QUARTERS[-1]]
    expect_error(lambda: choose_td(changed))
    for value in [np.nan, np.inf, -.1, 1.1, True]:
        changed = records()
        changed["td_original"][q]["curve"][0]["ap"] = value
        expect_error(lambda changed=changed: choose_td(changed))
    changed_recipes = deepcopy(TD_RECIPES)
    changed_recipes["td_original"]["params"]["depth"] = 8
    expect_error(lambda: choose_td(records(), changed_recipes))


def test_td_two_native_tiny_fits_path_reload_and_thread_budget():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(dict(value=rng.normal(size=80), category=["a", "b", None, "c"] * 20,
                          empty=np.nan))
    target = (x.value.to_numpy() > 0).astype(int)
    valid = pd.DataFrame(dict(value=[-1., 0., .5, np.nan], category=["unseen", "a", None, "b"], empty=np.nan))
    inner_y = np.array([0, 0, 1, 1])
    for name, recipe in TD_RECIPES.items():
        bundle = fit_fixed(recipe, x, target, ["category"], count=5, threads=2)
        expected = predict(bundle, valid)
        stream = BytesIO()
        joblib.dump(bundle, stream)
        stream.seek(0)
        np.testing.assert_array_equal(expected, predict(joblib.load(stream), valid))
        assert np.isfinite(expected).all()
        actual = bundle["effective_params"]
        assert actual["depth"] == 7 and actual["leaf_estimation_iterations"] == 1
        assert actual["one_hot_max_size"] == 15 and actual["bootstrap_type"] == "Bernoulli"
        path = fit_path(recipe, x, target, valid, inner_y, ["category"], cap=5, grid=[2, 5], threads=1)
        assert [point["count"] for point in path["curve"]] == [2, 5]
        assert path["best_count"] in [2, 5]
        expect_error(lambda recipe=recipe: fit_fixed(recipe, x, target, ["category"], count=2, threads=3))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS", flush=True)
