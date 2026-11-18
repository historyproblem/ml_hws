"""Alignment, tied AP, fixed blends and shared-site bootstrap checks."""
import warnings

import numpy as np
import pandas as pd
import pandas.testing as pdt
from sklearn.metrics import average_precision_score

from hw4_inspections.classical_analysis import analyze_predictions, prepare_weighted_ap


def fixture():
    parts = []
    for quarter in ["2023-07-01", "2023-10-01", "2024-01-01"]:
        common = pd.DataFrame({"row_id": [f"{quarter}_{i}" for i in range(6)],
                               "site_id": ["a", "b", "c", "a", "b", "c"],
                               "forecast_period": quarter, "target": [0, 1, 1, 0, 0, 1],
                               "seen_site": [True, False, True, True, False, True],
                               "history_violation_mean": [1., 3., np.nan, 2., 7., 0.]})
        for model, p in [("cb_reference", [.1, .5, .5, .4, .2, .8]),
                         ("candidate", [.2, .4, .8, .1, .4, .6])]:
            parts.append(common.assign(model=model, prediction=p))
    return pd.concat(parts, ignore_index=True)


def raises_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_weighted_average_precision_matches_sklearn_with_ties_and_zero_weights():
    rng = np.random.default_rng(319)
    for _ in range(40):
        y = rng.integers(0, 2, size=50)
        p = rng.integers(0, 5, size=50) / 4
        weight = rng.integers(0, 4, size=50)
        score = prepare_weighted_ap(y, p)
        expected = average_precision_score(y, p, sample_weight=weight)
        np.testing.assert_allclose(score(weight), expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(score(np.ones(50)), average_precision_score(y, p), rtol=0, atol=1e-14)
    score = prepare_weighted_ap([0, 1, 0], [.5, .5, .2])
    assert score([1, 0, 1]) == 0
    assert score([0, 0, 0]) == 0
    raises_value_error(lambda: score([1, -1, 1]))
    raises_value_error(lambda: score([1, np.nan, 1]))


def test_analysis_fixed_blends_alignment_metadata_and_order_invariance():
    source = fixture()
    result = analyze_predictions(source, bootstrap=30, seed=15)
    shuffled = analyze_predictions(source.sample(frac=1, random_state=9), bootstrap=30, seed=15)
    for name in ["predictions", "quarter_scores", "summary", "diagnostics"]:
        pdt.assert_frame_equal(result[name], shuffled[name])
    assert result["comparison"] == shuffled["comparison"]
    assert set(result["summary"].model) == {"cb_reference", "candidate", "candidate_blend50", "candidate_rankblend50"}
    wide = result["predictions"].pivot(index="row_id", columns="model", values="prediction")
    np.testing.assert_array_equal(wide.candidate_blend50, .5 * (wide.cb_reference + wide.candidate))
    for quarter in result["comparison"]["quarters"]:
        part = source[source.forecast_period.eq(quarter)].pivot(index="row_id", columns="model", values="prediction")
        expected = .5 * (part.cb_reference.rank(pct=True) + part.candidate.rank(pct=True))
        np.testing.assert_array_equal(wide.loc[expected.index, "candidate_rankblend50"], expected)
    diag = result["diagnostics"]
    assert {"warm", "cold", "burden_le2", "burden_gt2", "burden_missing"}.issubset(set(diag["slice"]))
    assert len(result["quarter_scores"]) == 12
    ref = result["summary"].set_index("model").loc["cb_reference"]
    assert ref.delta_vs_reference == ref.ci95_low == ref.ci95_high == 0


def test_analysis_rejects_misalignment_invalid_probabilities_and_closed_quarters():
    base = fixture()
    dirty = pd.concat([base, base.iloc[:1]], ignore_index=True)
    raises_value_error(lambda: analyze_predictions(dirty, bootstrap=0))
    for name, value in [("site_id", "wrong"), ("target", 1), ("seen_site", False),
                        ("forecast_period", "2023-10-01"), ("history_violation_mean", 8.)]:
        dirty = base.copy()
        dirty.loc[0, name] = value
        raises_value_error(lambda: analyze_predictions(dirty, bootstrap=0))
    for value in [np.nan, np.inf, -0.1, 1.1]:
        dirty = base.copy()
        dirty.loc[0, "prediction"] = value
        raises_value_error(lambda: analyze_predictions(dirty, bootstrap=0))
    dirty = base.copy()
    dirty.loc[0, "forecast_period"] = "2024-04-01"
    raises_value_error(lambda: analyze_predictions(dirty, bootstrap=0))
    raises_value_error(lambda: analyze_predictions(base.iloc[1:], bootstrap=0))
    raises_value_error(lambda: analyze_predictions(base[base.model.eq("candidate")], bootstrap=0))


def test_bootstrap_matches_manual_draws_shared_across_sites_models_and_quarters():
    base = fixture()
    repeats, seed = 40, 641
    result = analyze_predictions(base, bootstrap=repeats, seed=seed)
    reference = base.loc[base.model.eq("cb_reference")].sort_values(["forecast_period", "row_id"])
    wide = result["predictions"].pivot(index="row_id", columns="model", values="prediction").loc[reference.row_id]
    codes, sites = pd.factorize(reference.site_id, sort=True)
    rng = np.random.default_rng(seed)
    deltas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(repeats):
            weights = rng.multinomial(len(sites), np.full(len(sites), 1 / len(sites)))[codes]
            values = []
            for model in ["cb_reference", "candidate"]:
                aps = []
                for quarter in ["2023-07-01", "2023-10-01", "2024-01-01"]:
                    mask = reference.forecast_period.eq(quarter).to_numpy()
                    aps.append(average_precision_score(reference.target.to_numpy()[mask], wide[model].to_numpy()[mask],
                                                       sample_weight=weights[mask]))
                values.append(np.mean(aps))
            deltas.append(values[1] - values[0])
    actual = result["summary"].set_index("model").loc["candidate"]
    np.testing.assert_allclose([actual.ci95_low, actual.ci95_high], np.quantile(deltas, [.025, .975]),
                               rtol=0, atol=1e-14)
