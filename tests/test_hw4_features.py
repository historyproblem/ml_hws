"""Causal invariance checks for the isolated HW4 feature builder."""

import pandas as pd
import pandas.testing as pdt

from hw4_inspections.features import build_features


def fixture():
    rows = pd.DataFrame({
        "row_id": ["r1", "r2"], "site_id": ["s1", "cold"],
        "forecast_period": ["2024-10-01"] * 2,
        "history_before": ["2024-09-23"] * 2,
        "facility_type": ["restaurant", "school"],
        "risk_level": ["high", "low"], "area_id": ["a1", None],
        "target": [0, 1],
    }, index=[8, 2])
    history = pd.DataFrame({
        "site_id": ["s1", "s1"], "forecast_period": ["2024-10-01"] * 2,
        "inspection_week": ["2024-01-01", "2024-08-05"],
        "inspection_type": ["routine", "complaint"],
        "result": ["pass", "fail"], "violation_codes": ["1 1 44", "2 44"],
    })
    return rows, history


def test_snapshot_future_and_target_invariance():
    rows, history = fixture()
    expected, cats = build_features(rows, history)
    future = history.copy()
    future["inspection_week"] = "2024-09-30"
    cutoff_equal = history.copy()
    cutoff_equal["inspection_week"] = "2024-09-23"
    other_snapshot = history.copy()
    other_snapshot["forecast_period"] = "2025-01-01"
    other_snapshot["result"] = "conditional"
    dirty_history = pd.concat([history, future, cutoff_equal, other_snapshot], ignore_index=True)
    changed_rows = rows.copy()
    changed_rows["target"] = [1, 0]
    actual, actual_cats = build_features(changed_rows, dirty_history)
    pdt.assert_frame_equal(expected, actual)
    assert cats == actual_cats
    assert expected.index.equals(rows.index)
    assert expected.loc[8, "history_n_visits"] == 2
    assert expected.loc[8, "code01_count"] == 1
    assert expected.loc[8, "code44_count"] == 2
    assert expected.loc[8, "code44_last"] == 1
    assert expected.loc[8, "lag1_result"] == "fail"
    assert expected.loc[8, "lag2_result"] == "pass"
    assert expected.loc[8, "lag3_result"] == "<none>"
    assert expected.loc[2, "history_n_visits"] == 0
    assert pd.isna(expected.loc[2, "history_fail_rate"])
    assert not {"row_id", "site_id", "target"} & set(expected)


def test_history_order_and_no_history_invariance():
    rows, history = fixture()
    a, cats = build_features(rows, history)
    b, _ = build_features(rows, history.iloc[::-1])
    pdt.assert_frame_equal(a, b)
    empty, empty_cats = build_features(rows, history.iloc[:0])
    assert list(empty) == list(a)
    assert empty_cats == cats
    assert (empty.history_n_visits == 0).all()


def test_each_row_has_individual_cutoff():
    rows, history = fixture()
    rows = pd.concat([rows.iloc[:1], rows.iloc[:1]], ignore_index=True)
    rows.loc[1, "history_before"] = "2024-07-01"
    x, _ = build_features(rows, history)
    assert x.history_n_visits.tolist() == [2, 1]
    assert x.lag1_result.tolist() == ["fail", "pass"]
