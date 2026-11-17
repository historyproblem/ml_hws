"""Causal, leave-own-site-out checks for the optional feature hypotheses."""
import numpy as np
import pandas as pd
import pandas.testing as pdt

from hw4_inspections.extensions import build_context_features, build_dynamics_features


def fixture():
    rows = pd.DataFrame({
        "site_id": ["s1", "cold"], "forecast_period": ["2024-10-01"] * 2,
        "history_before": ["2024-09-23"] * 2, "area_id": ["a1", "a2"],
        "facility_type": ["restaurant", "school"], "risk_level": ["high", "low"],
        "target": [0, 1]}, index=[8, 2])
    history = pd.DataFrame({
        "site_id": ["s1", "s1", "s2", "s3"], "forecast_period": ["2024-10-01"] * 4,
        "inspection_week": ["2024-01-01", "2024-08-05", "2024-07-01", "2024-07-08"],
        "inspection_type": ["routine", "complaint", "routine", "routine"],
        "result": ["fail", "pass", "pass", "fail"], "violation_codes": ["1 2", "2 3", "4", "5 6 7"],
        "area_id": ["a1", "a1", "a1", "a2"], "facility_type": ["restaurant", "restaurant", "restaurant", "school"],
        "risk_level": ["high", "high", "high", "low"]})
    return rows, history


def test_extensions_snapshot_cutoff_target_and_order_invariance():
    rows, history = fixture()
    future = history.copy()
    future["inspection_week"] = "2024-09-23"
    other = history.copy()
    other["forecast_period"] = "2025-01-01"
    dirty = pd.concat([history, future, other], ignore_index=True).sample(frac=1, random_state=2)
    changed = rows.copy()
    changed["target"] = [1, 0]
    for builder in [build_context_features, build_dynamics_features]:
        a = builder(rows, history)
        b = builder(changed, dirty)
        pdt.assert_frame_equal(a, b)
        assert a.index.equals(rows.index)
        assert len(a.columns) <= 100
        assert not {"site_id", "row_id", "target"} & set(a)
        assert not np.isinf(a.to_numpy()).any()
        reordered = builder(rows.iloc[::-1], history)
        pdt.assert_frame_equal(a.iloc[::-1], reordered)


def test_context_excludes_own_site_from_group_and_prior():
    rows, history = fixture()
    a = build_context_features(rows.iloc[:1], history)
    changed = history.copy()
    own = changed.site_id == "s1"
    changed.loc[own, "result"] = "conditional"
    changed.loc[own, "violation_codes"] = "1 2 3 4 5 6 7 8"
    changed.loc[own, "facility_type"] = "school"
    b = build_context_features(rows.iloc[:1], changed)
    pdt.assert_frame_equal(a, b)
    assert a.loc[8, "ctx_area_all_routine_n"] == 1
    # Other-site prior is 1 fail / 2 visits; own-site records contribute nowhere.
    assert a.loc[8, "ctx_area_all_routine_fail_smooth"] == 10 / 21


def test_dynamics_code_transitions_and_individual_cutoffs():
    rows, history = fixture()
    x = build_dynamics_features(rows, history)
    assert x.loc[8, "dyn_last_routine_code01"] == 1
    assert x.loc[8, "dyn_last_routine_code03"] == 0
    assert x.loc[8, "dyn_all_pair_new_codes"] == 1
    assert x.loc[8, "dyn_all_pair_repeated_codes"] == 1
    assert x.loc[8, "dyn_all_pair_resolved_codes"] == 1
    assert x.loc[8, "dyn_all_pair_codes_jaccard"] == 1 / 3
    assert x.loc[8, "dyn_all_pair_fail_to_pass"] == 1
    assert x.loc[8, "dyn_after_routine_visits"] == 1
    assert x.loc[2, "dyn_all_has_pair"] == 0
    assert pd.isna(x.loc[2, "dyn_all_ew180_fail_rate"])
    repeated = pd.concat([rows.iloc[:1], rows.iloc[:1]], ignore_index=True)
    repeated.loc[1, "history_before"] = "2024-07-01"
    y = build_dynamics_features(repeated, history)
    assert y.dyn_after_routine_visits.tolist() == [1, 0]
    z = build_context_features(repeated, history)
    assert z.ctx_area_all_routine_n.tolist() == [1, 0]


def test_empty_history_retains_schema_and_missing_semantics():
    rows, history = fixture()
    for builder in [build_context_features, build_dynamics_features]:
        normal = builder(rows, history)
        empty = builder(rows, history.iloc[:0])
        assert list(normal) == list(empty)
        assert empty.index.equals(rows.index)
        assert not np.isinf(empty.to_numpy()).any()
    empty = build_context_features(rows, history.iloc[:0])
    assert empty.ctx_area_all_routine_n.eq(0).all()
    assert empty.ctx_area_all_routine_fail_smooth.isna().all()
