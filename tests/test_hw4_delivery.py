"""Small, label-free checks for the frozen HW4 delivery recipes."""
import numpy as np
import pandas as pd
import pandas.testing as pdt

from hw4_inspections.delivery import (
    choose_primary, make_candidates, rank_predictions, validate_submission,
)


def assert_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def family(values):
    return {42: np.array(values, dtype=float), 17: np.array(values, dtype=float),
            2026: np.array(values, dtype=float)}


def test_delivery_rank_predictions_ties_dtype_order_and_input_unchanged():
    values = np.array([.8, .1, .8, 0], dtype=np.float32)
    before = values.copy()
    ranks = rank_predictions(values)
    np.testing.assert_array_equal(ranks, [.875, .5, .875, .25])
    assert ranks.dtype == np.float64
    np.testing.assert_array_equal(values, before)
    np.testing.assert_array_equal(rank_predictions(values[::-1]), ranks[::-1])
    np.testing.assert_array_equal(rank_predictions([.2]), [1.])
    for invalid in [[], [[.1]], [np.nan], [np.inf], [-.1], [1.1], ["invalid"]]:
        assert_value_error(lambda invalid=invalid: rank_predictions(invalid))


def test_delivery_candidate_means_precede_ranking_and_formulas_are_fixed():
    reference = {42: np.array([0., .9, .3]), 17: np.array([.8, .4, .3]),
                 2026: np.array([.8, .4, .3])}
    compact = family([.9, .1, .5])
    visit = np.array([.1, .8, .8])
    snapshots = [{seed: values.copy() for seed, values in f.items()} for f in [reference, compact]]
    visit_before = visit.copy()
    result = make_candidates(reference, compact, visit)
    assert list(result) == ["cb_rank50", "cb_visit_rank3", "compact_cb", "context_cb"]
    reference_mean = np.mean(np.stack([reference[s] for s in [42, 17, 2026]]), axis=0)
    compact_mean = np.mean(np.stack([compact[s] for s in [42, 17, 2026]]), axis=0)
    rref, rcompact, rvisit = map(rank_predictions, [reference_mean, compact_mean, visit])
    np.testing.assert_array_equal(result["context_cb"], reference_mean)
    np.testing.assert_array_equal(result["compact_cb"], compact_mean)
    np.testing.assert_array_equal(result["cb_rank50"], .5 * rref + .5 * rcompact)
    np.testing.assert_array_equal(result["cb_visit_rank3"], (rref + rcompact + rvisit) / 3)
    mean_seed_ranks = np.mean(np.stack([rank_predictions(reference[s]) for s in [42, 17, 2026]]), axis=0)
    assert not np.array_equal(mean_seed_ranks, rref)
    for output in result.values():
        assert output.dtype == np.float64 and np.isfinite(output).all()
    for original, snapshot in zip([reference, compact], snapshots):
        for seed in original:
            np.testing.assert_array_equal(original[seed], snapshot[seed])
    np.testing.assert_array_equal(visit, visit_before)


def test_delivery_candidate_seed_shape_and_probability_guards():
    good = family([.1, .2])
    for bad in [{42: [.1, .2]}, {**good, 1: [.1, .2]},
                {42: [.1, .2], 17: [.1], 2026: [.1, .2]},
                {**good, 42: [np.nan, .2]}]:
        assert_value_error(lambda bad=bad: make_candidates(bad, good, [.1, .2]))
        assert_value_error(lambda bad=bad: make_candidates(good, bad, [.1, .2]))
    for visit in [[.1], [np.nan, .2], [[.1, .2]], [1.1, .2]]:
        assert_value_error(lambda visit=visit: make_candidates(good, good, visit))


def test_delivery_submission_joins_ids_in_sample_order_without_mutation():
    test = pd.DataFrame({"row_id": ["b", "a", "c"], "other": [1, 2, 3]}, index=[8, 1, 8])
    sample = pd.DataFrame({"row_id": ["c", "b", "a"], "target": [0, 0, 0]}, index=[9, 3, 2])
    values = np.array([.8, .1, .5])
    test_before, sample_before, values_before = test.copy(), sample.copy(), values.copy()
    actual = validate_submission(test, sample, values, expected_rows=3)
    expected = pd.DataFrame({"row_id": ["c", "b", "a"], "target": [.5, .8, .1]})
    pdt.assert_frame_equal(actual, expected)
    pdt.assert_frame_equal(test, test_before)
    pdt.assert_frame_equal(sample, sample_before)
    np.testing.assert_array_equal(values, values_before)
    permuted = [2, 0, 1]
    pdt.assert_frame_equal(validate_submission(test.iloc[permuted], sample, values[permuted]), expected)


def test_delivery_submission_rejects_incomplete_duplicate_missing_extra_ids():
    test = pd.DataFrame({"row_id": ["a", "b"]})
    sample = pd.DataFrame({"row_id": ["b", "a"]})
    for bad in [pd.DataFrame({"row_id": ["a", "a"]}),
                pd.DataFrame({"row_id": ["a", None]}),
                pd.DataFrame({"row_id": ["a", "other"]}),
                pd.DataFrame({"row_id": ["a"]}), pd.DataFrame({"other": [1, 2]})]:
        assert_value_error(lambda bad=bad: validate_submission(bad, sample, [.1, .2]))
        assert_value_error(lambda bad=bad: validate_submission(test, bad, [.1, .2]))
    for probabilities in [[.1], [np.nan, .2], [-.1, .2], [[.1, .2]]]:
        assert_value_error(lambda probabilities=probabilities: validate_submission(test, sample, probabilities))
    for expected_n in [1, 3, 0, True, 2.5]:
        assert_value_error(lambda expected_n=expected_n: validate_submission(test, sample, [.1, .2], expected_n))


def scores(first, second):
    return {"2024-04-01": {"cb_rank50": .3, "cb_visit_rank3": first},
            "2024-07-01": {"cb_rank50": .3, "cb_visit_rank3": second}}


def test_delivery_primary_promotion_requires_both_quarters_and_positive_mean():
    assert choose_primary(scores(.3, .3)) == "cb_rank50"
    assert choose_primary(scores(.31, .3)) == "cb_visit_rank3"
    assert choose_primary(scores(.3, .31)) == "cb_visit_rank3"
    assert choose_primary(scores(.31, .32)) == "cb_visit_rank3"
    assert choose_primary(scores(.299, .4)) == "cb_rank50"
    assert choose_primary(scores(.4, .299)) == "cb_rank50"
    assert choose_primary(scores(.2, .2)) == "cb_rank50"


def test_delivery_primary_rejects_wrong_quarters_models_or_invalid_scores():
    good = scores(.31, .32)
    for bad in [{"2024-04-01": good["2024-04-01"]},
                {**good, "2024-01-01": good["2024-04-01"]},
                {"2024-01-01": good["2024-04-01"], "2024-07-01": good["2024-07-01"]}]:
        assert_value_error(lambda bad=bad: choose_primary(bad))
    for value in [np.nan, np.inf, -.1, 1.1, [.3], "0.3", True]:
        bad = scores(value, .32)
        assert_value_error(lambda bad=bad: choose_primary(bad))
    bad = scores(.31, .32)
    bad["2024-04-01"]["other"] = .5
    assert_value_error(lambda: choose_primary(bad))
