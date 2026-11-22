"""Fit-only bounded splines and one fixed additive logistic hypothesis."""
from copy import deepcopy
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from hw4_inspections.additive_logit import (
    AdditiveLogitTransform, CATEGORICAL_COLUMNS, SPLINE_COLUMNS,
    fit_additive, predict_additive,
)
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features
from test_hw4_extensions import fixture


def _frame():
    rows, history = fixture()
    base, _ = build_features(rows, history)
    template = pd.concat([base, build_context_features(rows, history)], axis=1).iloc[[0]]
    frame = pd.concat([template] * 40, ignore_index=True)
    frame["facility_type"] = ["restaurant", "school"] * 20
    frame["risk_level"] = ["high", "low"] * 20
    frame["area_id"] = ["a1", "a2", "a3", "a4"] * 10
    frame["forecast_quarter"] = [1, 2, 3, 4] * 10
    frame["history_violation_mean"] = np.linspace(0, 12, len(frame))
    frame.loc[2, "history_violation_mean"] = np.nan
    frame["last_routine_violation_count"] = np.linspace(1, 9, len(frame))
    frame["lag1_violation_count"] = [0.] * 39 + [2.]
    frame["last3_violation_mean"] = 7.
    frame["violation_change_last_vs_mean"] = np.linspace(-4, 4, len(frame))
    frame["ctx_area_all_visits_n"] = np.arange(len(frame), dtype=float)
    frame["recent90_violation_total"] = np.arange(len(frame), dtype=float) * 2
    frame["recent90_violation_mean"] = np.nan
    assert frame.shape == (40, 346)
    return frame


def _raises(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_additive_fit_only_quantiles_constant_tails_fallbacks_and_missing():
    frame = _frame()
    before = frame.copy(deep=True)
    transform = AdditiveLogitTransform().fit(frame)
    assert transform.numeric_stats_["lag1_violation_count"]["mode"] == "linear_fallback"
    assert transform.numeric_stats_["last3_violation_mean"]["mode"] == "constant_zero"
    assert transform.numeric_stats_["recent90_violation_mean"]["all_missing"]
    observed = frame.history_violation_mean.dropna().to_numpy()
    np.testing.assert_array_equal(transform.spline_knots_["history_violation_mean"], np.unique(np.quantile(observed, [0, .25, .5, .75, 1])))
    knots = {k: v.copy() for k, v in transform.spline_knots_.items()}
    stats = deepcopy(transform.numeric_stats_)
    means = transform.scaler_.mean_.copy()
    query = frame.iloc[:3].copy()
    query.loc[query.index[0], "history_violation_mean"] = -1e6
    query.loc[query.index[1], "history_violation_mean"] = 1e6
    query["ctx_area_all_visits_n"] = [-100, 1e9, np.nan]
    clipped = query.copy()
    for name, state in transform.numeric_stats_.items():
        clipped[name] = clipped[name].clip(state["minimum"], state["maximum"])
    np.testing.assert_array_equal(transform.transform(query), transform.transform(clipped))
    assert stats == transform.numeric_stats_
    np.testing.assert_array_equal(means, transform.scaler_.mean_)
    for name, expected in knots.items():
        np.testing.assert_array_equal(expected, transform.spline_knots_[name])
    matrix = transform.transform(frame)
    names = transform.feature_names_.tolist()
    assert np.all(matrix[:, names.index("last3_violation_mean")] == 0)
    assert np.all(matrix[:, names.index("recent90_violation_mean")] == 0)
    assert np.all(matrix[:, names.index("recent90_violation_mean__missing")] == 1)
    pd.testing.assert_frame_equal(before, frame)


def test_additive_negative_change_is_linear_and_log_rule_is_explicit():
    frame = _frame()
    transform = AdditiveLogitTransform().fit(frame)
    assert not transform.numeric_stats_["violation_change_last_vs_mean"]["log1p"]
    assert transform.numeric_stats_["ctx_area_all_visits_n"]["log1p"]
    assert transform.numeric_stats_["recent90_violation_total"]["log1p"]
    assert not transform.numeric_stats_["lag1_violation_count"]["log1p"]
    matrix = transform.transform(frame)
    names = transform.feature_names_.tolist()
    change_index = names.index("violation_change_last_vs_mean")
    raw = frame.violation_change_last_vs_mean.to_numpy()
    expected = (raw - raw.mean()) / raw.std()
    np.testing.assert_allclose(matrix[:, change_index], expected, atol=1e-14)
    count_index = names.index("ctx_area_all_visits_n")
    logged = np.log1p(frame.ctx_area_all_visits_n.to_numpy())
    np.testing.assert_allclose(matrix[:, count_index], (logged - logged.mean()) / logged.std(), atol=1e-14)


def test_additive_absolute_calendars_omitted_and_quarter_unknown_ohe_is_safe():
    frame = _frame()
    transform = AdditiveLogitTransform().fit(frame)
    changed = frame.copy()
    changed["forecast_year"] = 1e100
    changed["cutoff_gap_days"] = -1e100
    np.testing.assert_array_equal(transform.transform(frame), transform.transform(changed))
    assert not {"forecast_year", "cutoff_gap_days"} & set(transform.selected_columns_)
    assert "forecast_quarter" not in transform.numeric_columns_
    assert transform.categorical_columns_ == list(CATEGORICAL_COLUMNS)
    query = frame.iloc[:3].copy()
    for name in CATEGORICAL_COLUMNS[:-1]:
        query[name] = "new-unseen"
    query["forecast_quarter"] = query.forecast_quarter.astype(float)
    matrix = transform.transform(query)
    assert np.isfinite(matrix).all()
    offset = len(transform.continuous_feature_names_) + len(transform.missing_feature_names_)
    # Only the known quarter indicator remains active in the OHE block.
    np.testing.assert_array_equal(matrix[:, offset:].sum(axis=1), np.ones(len(query)))
    numeric_quarter = frame.copy()
    numeric_quarter["forecast_quarter"] = numeric_quarter.forecast_quarter.astype(float)
    np.testing.assert_array_equal(transform.transform(frame), transform.transform(numeric_quarter))


def test_additive_schema_forbidden_columns_and_no_duplicate_spline_inputs():
    frame = _frame()
    transform = AdditiveLogitTransform().fit(frame)
    for name in ["target", "row_id", "site_id", "forecast_period", "history_before"]:
        _raises(lambda name=name: AdditiveLogitTransform().fit(frame.assign(**{name: 0})))
        _raises(lambda name=name: transform.transform(frame.assign(**{name: 0})))
    _raises(lambda: transform.transform(frame.iloc[:, ::-1]))
    _raises(lambda: transform.transform(frame.drop(columns="history_violation_mean")))
    _raises(lambda: AdditiveLogitTransform().fit(frame.drop(columns="ctx_area_all_visits_n")))
    names = transform.feature_names_.tolist()
    assert len(transform.numeric_columns_) == 90
    assert len(transform.missing_feature_names_) == 90
    assert len(transform.selected_columns_) == 94
    assert len(names) == len(set(names)) == transform.feature_budget_["total"] < 300
    for name in SPLINE_COLUMNS:
        if transform.numeric_stats_[name]["mode"] == "spline":
            assert name not in names
            assert sum(s.startswith(name + "__spline") for s in names) <= 5
        else:
            assert names.count(name) == 1


def test_additive_fixed_logit_deterministic_joblib_reload_and_probabilities():
    frame = _frame()
    target = np.array([0, 1] * 20)
    first = fit_additive(frame, target)
    p = predict_additive(first, frame)
    assert p.shape == (len(frame),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert first["recipe"]["classifier"]["C"] == .1
    assert first["recipe"]["classifier"]["max_iter"] == 1500
    assert first["recipe"]["monotonicity_constraints"] is False
    assert first["converged"]
    stream = BytesIO()
    joblib.dump(first, stream)
    stream.seek(0)
    restored = joblib.load(stream)
    np.testing.assert_array_equal(p, predict_additive(restored, frame))
    second = fit_additive(frame, target)
    np.testing.assert_array_equal(p, predict_additive(second, frame))
    np.testing.assert_array_equal(first["model"].coef_, second["model"].coef_)
    assert predict_additive(first, frame.iloc[:0]).shape == (0,)
    _raises(lambda: fit_additive(frame, np.ones(len(frame))))
    _raises(lambda: fit_additive(frame, target[:-1]))
