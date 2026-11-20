"""Checks for fixed per-export gates and correctly paired old-dev intervals."""
import warnings

import numpy as np
import pandas as pd
import pandas.testing as pdt
from sklearn.metrics import average_precision_score

from hw4_inspections.next_round import evaluate_candidates, rank_scores


def fixture():
    rows = []
    for quarter in ["2023-07-01", "2023-10-01", "2024-01-01"]:
        for j, target in enumerate([1, 0, 1, 0]):
            rows.append({"row_id": f"{quarter}_{j}", "site_id": f"s{j % 3}",
                         "forecast_period": quarter, "target": target})
    frame = pd.DataFrame(rows)
    reference = np.tile([.8, .9, .7, .6], 3)  # AP7/12 per quarter.
    perfect = np.array([.9, .7, .8, .6])
    worse = np.array([.6, .9, .7, .8])  # AP5/12.
    moderate = np.array([.8, .9, .6, .7])  # AP1/2.
    predictions = {"context_3seed": reference,
                   "useful_blend": np.concatenate([perfect, perfect, worse]),
                   "one_quarter_only": np.concatenate([perfect, moderate, moderate]),
                   "weak_standalone": np.tile(worse, 3),
                   "two_ties_mean_loss": np.concatenate([reference[:4], reference[:4], worse])}
    return frame, predictions


def raises_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_next_round_rank_casts_float64_before_percentiles_and_accepts_raw_scores():
    values = np.array([.1, .2, .3], dtype=np.float32)
    before = values.copy()
    actual = rank_scores(values)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.array([1 / 3, 2 / 3, 1.], dtype=np.float64))
    np.testing.assert_array_equal(rank_scores([-8, 0, 4, 4]), [.25, .5, .875, .875])
    np.testing.assert_array_equal(rank_scores([42]), [1.])
    np.testing.assert_array_equal(values, before)
    for bad in [[], np.nan, [[1., 2.]], [1., np.nan], [np.inf], [1 + 2j], ["bad"]]:
        raises_value_error(lambda bad=bad: rank_scores(bad))


def test_next_round_gates_each_export_not_whole_model_family():
    rows, predictions = fixture()
    result = evaluate_candidates(rows, predictions, bootstrap=20)
    eligibility = result["eligibility"]
    assert eligibility == {"context_3seed": False, "one_quarter_only": False,
                           "two_ties_mean_loss": False, "useful_blend": True,
                           "weak_standalone": False}
    summary = result["summary"].set_index("model")
    assert summary.loc["one_quarter_only", "delta_vs_reference"] > 0
    assert summary.loc["one_quarter_only", "nonnegative_quarters"] == 1
    assert summary.loc["two_ties_mean_loss", "nonnegative_quarters"] == 2
    assert summary.loc["two_ties_mean_loss", "delta_vs_reference"] < 0
    assert summary.loc["useful_blend", "latest_q1_delta"] < 0
    assert result["protocol"]["hard_ci_gate"] is False
    assert result["protocol"]["generated_blends"] == 0
    assert len(result["quarter_scores"]) == 3 * len(predictions)
    # A monotonic rescaling of raw ranker scores is accepted and keeps AP.
    scaled = evaluate_candidates(rows, {name: p * 10 - 5 for name, p in predictions.items()}, bootstrap=20)
    pdt.assert_frame_equal(result["summary"], scaled["summary"])


def test_next_round_bootstrap_matches_sklearn_shared_site_draws_and_is_deterministic():
    rows, predictions = fixture()
    repeats, seed = 30, 713
    result = evaluate_candidates(rows, predictions, bootstrap=repeats, seed=seed)
    again = evaluate_candidates(rows, dict(reversed(list(predictions.items()))), bootstrap=repeats, seed=seed)
    pdt.assert_frame_equal(result["summary"], again["summary"])
    pdt.assert_frame_equal(result["quarter_scores"], again["quarter_scores"])
    assert result["eligibility"] == again["eligibility"]
    codes, sites = pd.factorize(rows.site_id, sort=True)
    rng = np.random.default_rng(seed)
    deltas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(repeats):
            weight = rng.multinomial(len(sites), np.full(len(sites), 1 / len(sites)))[codes]
            means = []
            for name in ["context_3seed", "useful_blend"]:
                values = []
                for q in ["2023-07-01", "2023-10-01", "2024-01-01"]:
                    mask = rows.forecast_period.eq(q).to_numpy()
                    values.append(average_precision_score(rows.target.to_numpy()[mask], predictions[name][mask],
                                                          sample_weight=weight[mask]))
                means.append(np.mean(values))
            deltas.append(means[1] - means[0])
    actual = result["summary"].set_index("model").loc["useful_blend"]
    np.testing.assert_allclose([actual.ci95_low, actual.ci95_high], np.quantile(deltas, [.025, .975]),
                               rtol=0, atol=1e-14)


def test_next_round_rejects_later_missing_or_duplicate_metadata_before_scoring():
    rows, predictions = fixture()
    for period in ["2024-04-01", "2024-07-01", "2024-10-01", "2023-07-01 12:00:00"]:
        bad = rows.copy()
        bad.loc[0, "forecast_period"] = period
        raises_value_error(lambda bad=bad: evaluate_candidates(bad, predictions, bootstrap=0))
    for bad in [rows.iloc[:0], rows.iloc[:8], rows.assign(target=2),
                rows.assign(site_id=None), rows.drop(columns="row_id"),
                rows.assign(row_id="duplicate")]:
        raises_value_error(lambda bad=bad: evaluate_candidates(bad, predictions, bootstrap=0))


def test_next_round_rejects_misaligned_nonfinite_missing_reference_and_bad_options():
    rows, predictions = fixture()
    reference = predictions["context_3seed"]
    for bad in [{}, {"other": reference}, {"context_3seed": reference[:-1]},
                {"context_3seed": np.full(len(rows), np.nan)},
                {"context_3seed": np.zeros((len(rows), 1))},
                {"context_3seed": reference, "": reference}]:
        raises_value_error(lambda bad=bad: evaluate_candidates(rows, bad, bootstrap=0))
    for bootstrap in [-1, True, 1.5]:
        raises_value_error(lambda bootstrap=bootstrap: evaluate_candidates(rows, predictions, bootstrap=bootstrap))
    for seed in [-1, True, 1.5]:
        raises_value_error(lambda seed=seed: evaluate_candidates(rows, predictions, bootstrap=0, seed=seed))


def test_next_round_positional_alignment_is_invariant_when_rows_and_predictions_move_together():
    rows, predictions = fixture()
    original = evaluate_candidates(rows, predictions, bootstrap=30, seed=80)
    order = np.random.default_rng(5).permutation(len(rows))
    shuffled = evaluate_candidates(rows.iloc[order], {name: values[order] for name, values in predictions.items()},
                                   bootstrap=30, seed=80)
    pdt.assert_frame_equal(original["summary"], shuffled["summary"])
    pdt.assert_frame_equal(original["quarter_scores"], shuffled["quarter_scores"])
