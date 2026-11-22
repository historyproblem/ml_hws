"""Causal, adjacent and leave-focal-site-out routine transition checks."""
import numpy as np
import pandas as pd
import pandas.testing as pdt

from hw4_inspections.transition_features import (
    FEATURE_COLUMNS, _burden_bucket, build_transition_features,
)


def fixture():
    rows = pd.DataFrame({"site_id": ["q", "cold"], "forecast_period": "2024-10-01",
                         "history_before": "2024-09-23", "risk_level": "high",
                         "target": [0, 1]}, index=[8, 2])
    records = [
        ("q", "2024-01-01", "fail", "44"),
        ("q", "2024-03-01", "fail", "7 8 9"),
        ("q", "2024-05-01", "pass", "1 2"),
        ("a", "2024-01-01", "pass", "1 2"),
        ("a", "2024-05-01", "fail", "4 5 6"),
        ("b", "2024-01-01", "pass", "1 3"),
        ("b", "2024-05-01", "pass", "4"),
        ("c", "2024-01-01", "conditional", "1 2"),
        ("c", "2024-05-01", "fail", "5 6"),
        ("d", "2024-01-01", "pass", "4"),
        ("d", "2024-05-01", "pass", "7"),
    ]
    history = pd.DataFrame(records, columns=["site_id", "inspection_week", "result", "violation_codes"])
    history = history.assign(forecast_period="2024-10-01", risk_level="high", inspection_type="routine")
    return rows, history


def test_transition_state_code_adjustments_support_and_recorded_load():
    rows, history = fixture()
    x = build_transition_features(rows, history)
    assert list(x) == list(FEATURE_COLUMNS) and x.index.equals(rows.index)
    # Four other-site pairs, two failures; query's two completed pairs excluded.
    assert x.loc[8, "transition_next_fail_risk_burden"] == .5
    assert x.loc[8, "transition_next_fail_state"] == 11 / 23
    assert x.loc[8, "transition_state_log1p_pairs"] == np.log1p(3)
    assert x.loc[8, "transition_next_load_state"] == 40 / 23
    # Code1: 2 fails/3 pairs, code2: 2 fails/2 pairs, parent .5.
    np.testing.assert_allclose(x.loc[8, "transition_code_fail_effect_mean"], ((.5 / 23) + (1 / 22)) / 2, atol=1e-15)
    assert x.loc[8, "transition_code_fail_effect_max"] == 1 / 22
    assert x.loc[8, "transition_code_min_pairs"] == 2
    assert x.loc[2, [FEATURE_COLUMNS[2], FEATURE_COLUMNS[5]]].eq(0).all()
    assert x.loc[2].drop([FEATURE_COLUMNS[2], FEATURE_COLUMNS[5]]).isna().all()
    assert not {"site_id", "row_id", "target"} & set(x)


def test_transition_snapshot_cutoff_targets_and_input_order_invariant():
    rows, history = fixture()
    expected = build_transition_features(rows, history)
    future = history.assign(inspection_week="2024-09-23", result="fail")
    later = history.assign(inspection_week="2024-09-24", result="fail")
    other = history.assign(forecast_period="2025-01-01", result="fail")
    dirty = pd.concat([future, history, other, later], ignore_index=True).sample(frac=1, random_state=2)
    pdt.assert_frame_equal(expected, build_transition_features(rows.assign(target=1), dirty))
    pdt.assert_frame_equal(expected.iloc[::-1], build_transition_features(rows.iloc[::-1], history.iloc[::-1]))
    repeated = pd.concat([rows.iloc[:1], rows.iloc[:1]], ignore_index=True)
    repeated.index = [3, 3]
    result = build_transition_features(repeated, history)
    assert result.index.tolist() == [3, 3]
    np.testing.assert_array_equal(result.iloc[0], result.iloc[1])


def test_transition_own_endpoint_mutation_cannot_change_any_other_site_prior():
    rows, history = fixture()
    before = build_transition_features(rows.iloc[:1], history)
    changed = history.copy()
    own_older = changed.site_id.eq("q") & changed.inspection_week.lt("2024-05-01")
    changed.loc[own_older, "result"] = "pass"
    changed.loc[own_older, "risk_level"] = "different-risk"
    changed.loc[own_older, "violation_codes"] = "1 2 3 4 5 6 7 8 9 10"
    # Latest query state remains exactly fixed, but own global/risk/code totals
    # and both own endpoint outcomes have been changed.
    pdt.assert_frame_equal(before, build_transition_features(rows.iloc[:1], changed))


def test_transition_code_residuals_use_risk_burden_parents_not_global_rate():
    rows, history = fixture()
    # Move the entirely negative d pair to a second risk stratum. High-risk
    # code effects must use high-risk's own smoothed parent (12/23), rather
    # than merely subtracting the global failure probability .5.
    history.loc[history.site_id.eq("d"), "risk_level"] = "low"
    x = build_transition_features(rows.iloc[:1], history)
    parent = 12 / 23
    code1 = (2 - 3 * parent) / 23
    code2 = (2 - 2 * parent) / 22
    np.testing.assert_allclose(x["transition_next_fail_risk_burden"].iloc[0], parent, atol=1e-15)
    np.testing.assert_allclose(x["transition_code_fail_effect_mean"].iloc[0], (code1 + code2) / 2, atol=1e-15)
    assert abs(x["transition_code_fail_effect_mean"].iloc[0] - ((.5 / 23) + (1 / 22)) / 2) > .001
    changed = history.copy()
    own_older = changed.site_id.eq("q") & changed.inspection_week.lt("2024-05-01")
    changed.loc[own_older, "risk_level"] = "low"
    changed.loc[own_older, "result"] = "pass"
    changed.loc[own_older, "violation_codes"] = "1 2"
    pdt.assert_frame_equal(x, build_transition_features(rows.iloc[:1], changed))


def test_transition_unknown_outcomes_do_not_bridge_or_become_negative_labels():
    rows, history = fixture()
    extra = pd.DataFrame({"site_id": ["unknown"] * 3, "inspection_week": ["2024-01-01", "2024-03-01", "2024-05-01"],
                          "result": ["pass", "unknown", "fail"], "violation_codes": ["1 2"] * 3,
                          "risk_level": "high", "forecast_period": "2024-10-01", "inspection_type": "routine"})
    before = build_transition_features(rows.iloc[:1], history)
    pdt.assert_frame_equal(before, build_transition_features(rows.iloc[:1], pd.concat([history, extra], ignore_index=True)))
    unknown_query = rows.iloc[:1].assign(site_id="latest-unknown")
    u = extra.iloc[:2].assign(site_id="latest-unknown")
    unknown = build_transition_features(unknown_query, pd.concat([history, u], ignore_index=True))
    assert unknown[[FEATURE_COLUMNS[2], FEATURE_COLUMNS[5]]].eq(0).all().all()
    assert unknown.drop(columns=[FEATURE_COLUMNS[2], FEATURE_COLUMNS[5]]).isna().all().all()


def test_transition_exact_duplicate_invariance_conflicting_week_rejected():
    rows, history = fixture()
    before = build_transition_features(rows, history)
    duplicated = pd.concat([history, history], ignore_index=True)
    pdt.assert_frame_equal(before, build_transition_features(rows, duplicated))
    conflict = pd.concat([history, history.iloc[:1].assign(result="pass")], ignore_index=True)
    try:
        build_transition_features(rows, conflict)
        raise AssertionError("Ambiguous same-site/week routines accepted")
    except ValueError:
        pass


def test_transition_no_cross_snapshot_pairs_individual_cutoffs_and_empty_history():
    rows, history = fixture()
    # Moving every endpoint to another snapshot removes every completed pair.
    fragmented = history.copy()
    fragmented.loc[fragmented.inspection_week.ne("2024-01-01"), "forecast_period"] = "2025-01-01"
    query = rows.iloc[:1].assign(site_id="a")
    result = build_transition_features(query, fragmented)
    assert result["transition_state_log1p_pairs"].iloc[0] == 0
    assert result["transition_next_fail_state"].isna().all()
    cuts = pd.concat([rows.iloc[:1], rows.iloc[:1]], ignore_index=True)
    cuts.loc[1, "history_before"] = "2024-05-01"
    separate = build_transition_features(cuts, history)
    assert separate["transition_next_fail_state"].iloc[0] == 11 / 23
    assert separate["transition_next_fail_state"].isna().iloc[1]  # Only own q pair before cutoff.
    empty = build_transition_features(rows, history.iloc[:0])
    assert list(empty) == list(FEATURE_COLUMNS) and empty.index.equals(rows.index)
    assert empty[[FEATURE_COLUMNS[2], FEATURE_COLUMNS[5]]].eq(0).all().all()
    assert empty.drop(columns=[FEATURE_COLUMNS[2], FEATURE_COLUMNS[5]]).isna().all().all()
    assert not np.isinf(empty.to_numpy()).any()


def test_transition_blank_codes_and_fixed_burden_buckets():
    np.testing.assert_array_equal(_burden_bucket([0, 1, 2, 3, 5, 6, 8, 9, 44]), [0, 1, 1, 2, 2, 3, 3, 4, 4])
    rows, history = fixture()
    history.loc[history.site_id.eq("q") & history.inspection_week.eq("2024-05-01"), "violation_codes"] = ""
    x = build_transition_features(rows.iloc[:1], history)
    assert x["transition_next_fail_risk_burden"].iloc[0] == .5
    assert x[[FEATURE_COLUMNS[3], FEATURE_COLUMNS[4]]].isna().all().all()
    assert x["transition_code_min_pairs"].iloc[0] == 0
