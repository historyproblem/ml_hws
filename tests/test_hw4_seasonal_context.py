"""Synthetic boundary, first-event, other-site and snapshot invariance checks."""
import numpy as np
import pandas as pd
import pandas.testing as pdt

from hw4_inspections.seasonal_context import COLUMNS, build_seasonal_context_features


def fixture():
    rows = pd.DataFrame({"site_id": ["own", "cold"], "forecast_period": ["2024-10-01"] * 2,
                         "history_before": ["2024-09-23"] * 2,
                         "area_id": ["A", "Z"], "risk_level": ["high", "high"],
                         "facility_type": ["restaurant", "restaurant"], "target": [0, 1]}, index=[8, 2])
    events = [
        ("coverage", "2021-01-04", "other", "unknown", "", "C", "low", "school"),
        ("own", "2022-10-03", "routine", "fail", "1 2 3", "A", "high", "restaurant"),
        ("own", "2023-10-02", "routine", "fail", "1 2 3 4 5 6 7 8 9 10", "A", "high", "restaurant"),
        ("B", "2022-10-03", "routine", "pass", "1", "A", "high", "restaurant"),
        ("B", "2022-12-05", "routine", "fail", "1 2 3 4 5", "A", "high", "restaurant"),
        ("B", "2023-10-02", "routine", "fail", "1 2", "A", "high", "restaurant"),
        ("C", "2023-11-06", "routine", "conditional", "1 2 3", "B", "low", "school"),
    ]
    history = pd.DataFrame(events, columns=["site_id", "inspection_week", "inspection_type", "result",
                                           "violation_codes", "area_id", "risk_level", "facility_type"])
    history["forecast_period"] = "2024-10-01"
    return rows, history


def test_seasonal_first_not_last_and_same_site_across_years():
    rows, history = fixture()
    x = build_seasonal_context_features(rows, history)
    assert list(x) == list(COLUMNS) and x.shape == (2, 12)
    assert x.index.equals(rows.index)
    # B contributes once in2022 (pass,one code) and once in2023 (fail,two).
    # C contributes conditional/three codes to the other-site global prior.
    for name in ["area", "risk", "area_risk", "facility_risk"]:
        assert x.loc[8, f"seas_{name}_n_siteyears"] == 2
        np.testing.assert_allclose(x.loc[8, f"seas_{name}_fail_smooth"], 23 / 66)
        np.testing.assert_allclose(x.loc[8, f"seas_{name}_recorded_load_smooth"], 43 / 22)
    # Empty group falls back to all other eligible site-years, not zero risk.
    assert x.loc[2, "seas_area_n_siteyears"] == 0
    np.testing.assert_allclose(x.loc[2, "seas_area_fail_smooth"], 3 / 5)
    np.testing.assert_allclose(x.loc[2, "seas_area_recorded_load_smooth"], 19 / 5)


def test_seasonal_snapshot_cutoff_target_and_order_invariance():
    rows, history = fixture()
    expected = build_seasonal_context_features(rows, history)
    extra = pd.concat([history.assign(forecast_period="2025-01-01"),
                       history.assign(inspection_week="2024-09-23"),
                       history.assign(inspection_week="2024-10-07")], ignore_index=True)
    dirty = pd.concat([history, extra], ignore_index=True).sample(frac=1, random_state=17)
    actual = build_seasonal_context_features(rows.assign(target=[1, 0]), dirty.assign(target=1))
    pdt.assert_frame_equal(expected, actual)
    pdt.assert_frame_equal(build_seasonal_context_features(rows.iloc[::-1], history.iloc[::-1]), expected.iloc[::-1])


def test_seasonal_own_exclusion_including_profiles_unknowns_and_coverage():
    rows, history = fixture()
    rows = rows.iloc[:1]
    # This old site's quarter is left-truncated without a September coverage row.
    older = history.iloc[[0]].assign(site_id="older", inspection_week="2020-10-05",
                                    inspection_type="routine", result="fail", area_id="A",
                                    risk_level="high", facility_type="restaurant", violation_codes="1")
    history = pd.concat([history, older], ignore_index=True)
    before = build_seasonal_context_features(rows, history)
    changed = history.copy()
    own = changed.site_id.eq("own")
    changed.loc[own, ["result", "area_id", "risk_level", "facility_type", "violation_codes"]] = [
        "unknown", "Z", "low", "school", "1 2 3 4 5 6 7 8"]
    own_early = older.assign(site_id="own", inspection_week="2020-09-28", inspection_type="other")
    changed = pd.concat([changed, own_early], ignore_index=True)
    pdt.assert_frame_equal(before, build_seasonal_context_features(rows, changed))


def test_seasonal_unknown_first_is_not_replaced_by_later_known():
    rows, history = fixture()
    first = history.site_id.eq("B") & history.inspection_week.eq("2023-10-02")
    later = history.loc[first].assign(inspection_week="2023-11-13", result="fail")
    history.loc[first, "result"] = "unknown"
    history = pd.concat([history, later], ignore_index=True)
    x = build_seasonal_context_features(rows.iloc[:1], history)
    assert x.loc[8, "seas_area_n_siteyears"] == 1
    assert x.loc[8, "seas_area_fail_smooth"] == 0
    np.testing.assert_allclose(x.loc[8, "seas_area_recorded_load_smooth"], 41 / 21)


def test_seasonal_first_boundary_cell_excludes_quarter_but_later_boundary_does_not():
    rows, original = fixture()
    history = original.iloc[[0]].copy()
    events = [
        ("bad_start", "2022-09-26", "fail"),
        ("bad_start", "2022-10-10", "pass"),
        ("bad_end_only", "2022-12-26", "fail"),
        ("good_early", "2022-10-03", "pass"),
        ("good_early", "2022-12-26", "fail"),
    ]
    for site, date, result in events:
        row = original.iloc[[3]].assign(site_id=site, inspection_week=date, result=result)
        history = pd.concat([history, row], ignore_index=True)
    x = build_seasonal_context_features(rows.iloc[:1], history)
    assert x.loc[8, "seas_area_n_siteyears"] == 1
    assert x.loc[8, "seas_area_fail_smooth"] == 0
    assert x.loc[8, "seas_area_recorded_load_smooth"] == 1
    # The same December/January cell could also be the FIRST routine ofQ1.
    q1 = rows.iloc[:1].assign(forecast_period="2024-01-01", history_before="2023-12-25")
    later = original.iloc[[3]].assign(site_id="bad_end_only", inspection_week="2023-01-09", result="pass")
    good = original.iloc[[3]].assign(site_id="good_january", inspection_week="2023-01-02", result="fail")
    history = pd.concat([history, later, good], ignore_index=True).assign(forecast_period="2024-01-01")
    y = build_seasonal_context_features(q1, history)
    assert y.loc[8, "seas_area_n_siteyears"] == 1
    assert y.loc[8, "seas_area_fail_smooth"] == 1


def test_seasonal_left_truncation_and_not_yet_completed_quarters():
    rows, history = fixture()
    rows = rows.iloc[:1]
    # Earliest available OTHER weekOct3 cannot certify the Oct1 boundary week.
    history = history.loc[history.site_id.eq("B") & history.inspection_week.isin(["2022-10-03", "2023-10-02"])].copy()
    own = history.iloc[[0]].assign(site_id="own", inspection_week="2022-09-26", inspection_type="other")
    incomplete = build_seasonal_context_features(rows, pd.concat([history, own], ignore_index=True))
    assert incomplete.loc[8, "seas_area_n_siteyears"] == 1
    assert incomplete.loc[8, "seas_area_fail_smooth"] == 1
    coverage = own.assign(site_id="coverage")
    complete = build_seasonal_context_features(rows, pd.concat([history, coverage], ignore_index=True))
    assert complete.loc[8, "seas_area_n_siteyears"] == 2
    # A much earlier per-row cutoff cannot include an unfinished2023Q4.
    early = rows.assign(history_before="2023-11-20")
    x = build_seasonal_context_features(early, pd.concat([history, coverage], ignore_index=True))
    assert x.loc[8, "seas_area_n_siteyears"] == 1
    assert x.loc[8, "seas_area_fail_smooth"] == 0


def test_seasonal_empty_history_no_other_sites_and_duplicate_input_indices():
    rows, history = fixture()
    normal = build_seasonal_context_features(rows, history)
    for empty in [history.iloc[:0], history.loc[history.site_id.eq("own")]]:
        actual = build_seasonal_context_features(rows.iloc[:1], empty)
        assert list(actual) == list(normal)
        assert actual.filter(like="_n_siteyears").eq(0).all().all()
        assert actual.drop(columns=actual.filter(like="_n_siteyears").columns).isna().all().all()
    duplicated = pd.concat([rows.iloc[:1], rows.iloc[:1]])
    expected = pd.concat([normal.iloc[:1], normal.iloc[:1]])
    pdt.assert_frame_equal(build_seasonal_context_features(duplicated, history), expected)
    empty_rows = build_seasonal_context_features(rows.iloc[:0], history)
    assert empty_rows.shape == (0, 12) and empty_rows.index.equals(rows.iloc[:0].index)


def test_seasonal_duplicate_first_cells_are_conservative_and_exact_duplicates_do_not_inflate():
    rows, history = fixture()
    original = build_seasonal_context_features(rows.iloc[:1], history)
    pdt.assert_frame_equal(original, build_seasonal_context_features(rows.iloc[:1], pd.concat([history, history], ignore_index=True)))
    conflict = history.loc[history.site_id.eq("B") & history.inspection_week.eq("2022-10-03")].assign(result="fail")
    x = build_seasonal_context_features(rows.iloc[:1], pd.concat([history, conflict], ignore_index=True))
    assert x.loc[8, "seas_area_n_siteyears"] == 1


def test_seasonal_rejects_non_monday_used_history_and_future_cutoff():
    rows, history = fixture()
    for r, h in [(rows, history.assign(inspection_week="2023-10-03")),
                 (rows.assign(history_before="2024-10-07"), history)]:
        try:
            build_seasonal_context_features(r, h)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid weekly/date contract accepted")
