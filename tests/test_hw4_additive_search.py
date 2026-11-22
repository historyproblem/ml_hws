"""Tiny-only representation, weighting and earlier-quarter selection checks."""
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from unittest.mock import patch

from hw4_inspections.additive_logit import AdditiveLogitTransform
from hw4_inspections.additive_search import (
    RECIPES, SearchTransform, choose_final, choose_inner,
    fit_recipe, predict_recipe, recipe_by_id,
)
from test_hw4_additive_logit import _frame


def test_additive_search_has24_fixed_recipes_and_original_core_design_exact():
    assert len(RECIPES) == len({r["id"] for r in RECIPES}) == 24
    frame = _frame()
    old = AdditiveLogitTransform().fit(frame)
    for view in ["core", "core_recency4"]:
        new = SearchTransform(view).fit(frame)
        np.testing.assert_array_equal(old.transform(frame), new.transform(frame))
        assert list(old.feature_names_) == new.feature_names_


def test_additive_search_wider_views_fit_only_bounded_and_interactions_unknown_safe():
    frame = _frame()
    for view in ["compact_all_splines", "own_history", "full_context", "core_interactions"]:
        transform = SearchTransform(view).fit(frame)
        query = frame.iloc[:3].copy()
        for column in ["history_violation_mean", "last_routine_violation_count"]:
            query[column] = [-1e6, 1e6, np.nan]
        query["risk_level"] = "never-seen"
        query["area_id"] = "never-seen"
        matrix = transform.transform(query)
        assert np.isfinite(matrix).all() and matrix.shape[1] == len(transform.feature_names_)
        if view == "own_history":
            assert not any(c.startswith("ctx_") for c in transform.base_.selected_columns_)
        assert not {"forecast_year", "cutoff_gap_days"} & set(transform.base_.selected_columns_)
        if view == "core_interactions":
            assert np.all(matrix[:, len(transform.base_.feature_names_):] == 0)


def test_additive_search_l2_and_recency_are_explicit_and_within2threads():
    frame = _frame()
    target = np.array([0, 1] * 20)
    periods = pd.Series(["2022-01-01", "2023-01-01"] * 20)
    for name in ["core_C0.1", "core_recency4_C0.1", "compact_all_splines_C0.01"]:
        bundle = fit_recipe(frame, target, recipe_by_id(name), periods=periods, reference="2023-07-01")
        assert bundle["threads"] == 2 and bundle["model"].get_params()["C"] == recipe_by_id(name)["C"]
        assert np.isfinite(predict_recipe(bundle, frame)).all()
        if "recency" in name:
            assert bundle["sample_weights"] == "mean-one quarter half-life4"


def test_additive_search_inner_ties_invalid_and_final_majority():
    rows = [dict(recipe=r["id"], status="complete", ap=.2) for r in RECIPES]
    assert choose_inner(rows) == "core_C0.01"
    rows[0]["status"] = "invalid"
    assert choose_inner(rows) == "core_recency4_C0.01"
    try:
        choose_inner([dict(recipe=r["id"], status="invalid", ap=None) for r in RECIPES])
        raise AssertionError("All-invalid inner quarter silently dropped")
    except RuntimeError:
        pass
    complete = [dict(recipe=r["id"], status="complete", ap=.2 + .001 * q)
                for q in range(3) for r in RECIPES]
    assert choose_final(["core_C0.1", "core_interactions_C1", "core_C0.1"], complete) == "core_C0.1"
