"""Causal latest-site-state checks using small synthetic histories only."""
import numpy as np
import pandas as pd
import pandas.testing as pdt

from hw4_inspections.site_state import build_site_state_features, STATE_GROUPS
from test_hw4_extensions import fixture


def test_site_state_snapshot_cutoff_target_order_and_schema():
    rows, history = fixture()
    output = build_site_state_features(rows, history)
    expected = [f"state_{group}_{window}_{metric}"
                for window in ["all", "365d"] for group in STATE_GROUPS
                for metric in ["n_sites", "fail_smooth", "load_smooth"]]
    assert list(output) == expected and len(expected) == 24
    assert output.index.equals(rows.index)
    assert not {"target", "row_id", "site_id"} & set(output)
    future = history.assign(inspection_week="2024-09-23")
    other = history.assign(forecast_period="2025-01-01")
    dirty = pd.concat([history, future, other], ignore_index=True).sample(frac=1, random_state=41)
    pdt.assert_frame_equal(output, build_site_state_features(rows.assign(target=[1, 0]), dirty))
    pdt.assert_frame_equal(output.iloc[::-1], build_site_state_features(rows.iloc[::-1], history))
    assert not np.isinf(output.to_numpy()).any()


def test_site_state_excludes_own_site_from_group_and_global_prior():
    rows, history = fixture()
    one = rows.iloc[:1]
    output = build_site_state_features(one, history)
    assert output.loc[8, "state_area_all_n_sites"] == 1
    assert output.loc[8, "state_area_all_fail_smooth"] == 10 / 21
    assert output.loc[8, "state_area_all_load_smooth"] == 41 / 21
    changed = history.copy()
    own = changed.site_id.eq("s1")
    changed.loc[own, "result"] = "pass"
    changed.loc[own, "violation_codes"] = "1 2 3 4 5 6 7 8 9"
    changed.loc[own, "area_id"] = "changed"
    changed.loc[own, "risk_level"] = "changed"
    pdt.assert_frame_equal(output, build_site_state_features(one, changed))


def test_site_state_one_latest_visit_and_unknown_latest_has_no_old_fallback():
    rows, history = fixture()
    output = build_site_state_features(rows, history)
    older = history.loc[history.site_id.eq("s2")].copy()
    older["inspection_week"] = "2024-01-08"
    older["result"] = "fail"
    older["violation_codes"] = "1 2 3 4 5 6 7"
    pdt.assert_frame_equal(output, build_site_state_features(rows, pd.concat([history, older])))
    # An unknown latest outcome removes the site, rather than using July pass.
    unknown = older.assign(inspection_week="2024-08-12", result="unknown")
    changed = build_site_state_features(rows.iloc[:1], pd.concat([history, older, unknown]))
    assert changed.loc[8, "state_area_all_n_sites"] == 0
    assert pd.isna(changed.loc[8, "state_area_all_fail_smooth"])
    assert pd.isna(changed.loc[8, "state_area_all_load_smooth"])
    # Other global sites still exist: group-empty rates intentionally stay NaN.
    assert changed.loc[8, "state_risk_all_n_sites"] == 0


def test_site_state_empty_history_individual_cutoffs_and_window_membership():
    rows, history = fixture()
    empty = build_site_state_features(rows, history.iloc[:0])
    assert len(empty.columns) == 24 and empty.index.equals(rows.index)
    counts = [c for c in empty if c.endswith("_n_sites")]
    rates = [c for c in empty if not c.endswith("_n_sites")]
    assert empty[counts].eq(0).all().all() and empty[rates].isna().all().all()
    repeated = pd.concat([rows.iloc[:1], rows.iloc[:1]], ignore_index=True)
    repeated.loc[1, "history_before"] = "2024-07-01"
    output = build_site_state_features(repeated, history)
    assert output.state_area_all_n_sites.tolist() == [1, 0]
    old = history.copy()
    old.loc[old.site_id.eq("s2"), "inspection_week"] = "2022-01-03"
    result = build_site_state_features(rows.iloc[:1], old)
    assert result.loc[8, "state_area_all_n_sites"] == 1
    assert result.loc[8, "state_area_365d_n_sites"] == 0


def test_site_state_rejects_conflicting_latest_week_but_exact_copy_is_harmless():
    rows, history = fixture()
    pdt.assert_frame_equal(build_site_state_features(rows, history),
                           build_site_state_features(rows, pd.concat([history, history])))
    conflict = history.loc[history.site_id.eq("s2")].assign(result="fail")
    try:
        build_site_state_features(rows, pd.concat([history, conflict]))
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting same-site/week routine states were accepted")
