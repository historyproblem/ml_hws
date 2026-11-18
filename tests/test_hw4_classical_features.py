"""Feature-contract checks for the fixed HW4 classical experiment."""
import numpy as np
import pandas as pd
import pandas.testing as pdt

from hw4_inspections.classical_features import (
    COMPACT_BASE_COLUMNS, build_relative_features, recency_weights,
    select_feature_columns,
)
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features
from test_hw4_extensions import fixture


def assert_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_compact_selector_exact_prior_ablation_and_order():
    rows, history = fixture()
    base, _ = build_features(rows, history)
    context = build_context_features(rows, history)
    frame = pd.concat([base, context], axis=1)
    # Independent reproduction of the previously measured ablation rule.
    static_calendar = {"facility_type", "risk_level", "area_id",
                       "forecast_year", "forecast_quarter", "cutoff_gap_days"}
    prior = [c for c in base if c in static_calendar or "violation" in c]
    assert len(prior) == 24
    assert prior == list(COMPACT_BASE_COLUMNS)
    expected = prior + list(context)
    assert len(expected) == 96
    assert select_feature_columns(frame, "compact_context") == expected
    assert select_feature_columns(frame, "context") == list(frame)
    shuffled = frame.loc[:, frame.columns[::-1]]
    assert select_feature_columns(shuffled, "compact_context") == expected[::-1]
    assert select_feature_columns(frame.assign(new_violation_feature=1), "compact_context") == expected


def test_selector_rejects_raw_identifiers_targets_and_incomplete_inputs():
    rows, history = fixture()
    base, _ = build_features(rows, history)
    for name in ["target", "row_id", "site_id", "history_before", "forecast_period"]:
        assert_value_error(lambda name=name: select_feature_columns(base.assign(**{name: 1}), "context"))
        assert_value_error(lambda name=name: select_feature_columns(base.assign(**{name: 1}), "compact_context"))
    assert_value_error(lambda: select_feature_columns(base, "unknown"))
    assert_value_error(lambda: select_feature_columns(base.drop(columns="history_violation_mean"), "compact_context"))
    assert_value_error(lambda: select_feature_columns(pd.concat([base, base.iloc[:, :1]], axis=1), "context"))


def test_relative_features_exact_values_missing_and_row_order():
    frame = pd.DataFrame({
        "history_violation_mean": [6, np.nan, 0],
        "last_routine_violation_count": [3, 5, np.nan],
        "ctx_area_365d_routine_load_smooth": [2, 0, np.nan],
        "ctx_area_risk_365d_routine_load_smooth": [1, np.nan, 0],
        "target": [1, 0, 1], "site_id": ["a", "b", "c"],
    }, index=[8, 2, 8])
    before = frame.copy(deep=True)
    out = build_relative_features(frame)
    names = [f"rel_{own}_vs_{context}_{metric}"
             for own in ["history", "last_routine"]
             for context in ["area", "area_risk"]
             for metric in ["difference", "ratio"]]
    assert list(out) == names
    assert out.index.equals(frame.index)
    np.testing.assert_allclose(out.iloc[0], [4, 2, 5, 3, 1, 1, 2, 1.5])
    assert out.iloc[1].isna().tolist() == [True, True, True, True, False, False, True, True]
    assert out.iloc[2].isna().tolist() == [True, True, False, False, True, True, True, True]
    assert out.iloc[1]["rel_last_routine_vs_area_ratio"] == 5
    assert out.iloc[2]["rel_history_vs_area_risk_ratio"] == 0
    assert all(dtype.kind == "f" for dtype in out.dtypes)
    pdt.assert_frame_equal(frame, before)
    pdt.assert_frame_equal(build_relative_features(frame.iloc[::-1]), out.iloc[::-1])
    changed = frame.assign(target=0, site_id="different")
    pdt.assert_frame_equal(build_relative_features(changed), out)
    assert not np.isinf(out.to_numpy()).any()


def test_relative_features_require_inputs_and_guard_zero_denominator():
    frame = pd.DataFrame({
        "history_violation_mean": [2.0], "last_routine_violation_count": [1.0],
        "ctx_area_365d_routine_load_smooth": [-1.0],
        "ctx_area_risk_365d_routine_load_smooth": [1.0],
    })
    out = build_relative_features(frame)
    assert out.rel_history_vs_area_ratio.isna().all()
    assert out.rel_last_routine_vs_area_ratio.isna().all()
    assert_value_error(lambda: build_relative_features(frame.drop(columns="history_violation_mean")))


def test_recency_weights_quarter_half_life_normalization_and_order():
    periods = ["2022-01-01", "2023-01-01", "2023-04-01", "2023-06-30"]
    weights = recency_weights(periods, "2023-07-01", half_life_quarters=4)
    assert weights.dtype == np.float64
    assert np.isfinite(weights).all() and (weights > 0).all()
    np.testing.assert_allclose(weights.mean(), 1, atol=1e-15)
    np.testing.assert_allclose(weights[1] / weights[0], 2, atol=1e-15)
    assert weights[2] == weights[3]
    np.testing.assert_array_equal(recency_weights(periods[::-1], "2023-07-01"), weights[::-1])
    np.testing.assert_array_equal(recency_weights(periods, "2023-09-30"), weights)
    # A common shift in age cancels when normalized; only relative age matters.
    np.testing.assert_allclose(recency_weights(periods, "2024-07-01"), weights, atol=1e-15)


def test_recency_weights_reject_noncausal_or_invalid_inputs():
    for periods in [["2023-07-01"], ["2023-08-01"], ["2024-01-01"], [None], []]:
        assert_value_error(lambda periods=periods: recency_weights(periods, "2023-07-01"))
    for half_life in [0, -1, np.inf, np.nan]:
        assert_value_error(lambda half_life=half_life: recency_weights(["2023-01-01"], "2023-07-01", half_life))
    assert_value_error(lambda: recency_weights(["2023-01-01"], None))
