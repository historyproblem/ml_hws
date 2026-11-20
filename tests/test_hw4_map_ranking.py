"""Small CPU MAP training, group alignment, raw scores and AP-grid checks."""
from io import BytesIO
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pandas.testing as pdt
from sklearn.metrics import average_precision_score

from hw4_inspections import map_ranking
from hw4_inspections.map_ranking import (
    choose_ranker_iterations, fit_ranker, predict_ranker,
)


def assert_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def fixture():
    rng = np.random.default_rng(52)
    x = pd.DataFrame({"value": rng.normal(size=48),
                      "empty": [np.nan] * 48,
                      "category": ["a", "b", None, "a"] * 12})
    x.loc[::9, "value"] = np.inf
    y = (np.nan_to_num(x.value.to_numpy(), posinf=0) + rng.normal(size=48) > 0).astype(int)
    # Interleaving preserves a deterministic row order within each group.
    quarters = np.array(["2023-04-01", "2023-01-01"] * 24)
    valid = pd.DataFrame({"value": [2., np.nan, -2., .2], "empty": [1., np.nan, 0., 0.],
                          "category": ["unseen", "a", None, "b"]}, index=[8, 2, 8, 1])
    return x, y, quarters, valid


def test_map_ranker_tiny_cpu_native_missing_raw_scores_and_reload():
    x, y, quarters, valid = fixture()
    before = x.copy(deep=True)
    bundle = fit_ranker(x, y, quarters, ["category"], iterations=5, threads=1)
    scores = predict_ranker(bundle, valid)
    assert scores.shape == (len(valid),) and scores.dtype == np.float64
    assert np.isfinite(scores).all()
    params = bundle["model"].get_params()
    assert params["loss_function"] == "YetiRank:mode=MAP" and params["task_type"] == "CPU"
    assert params["depth"] == 5 and params["learning_rate"] == .04 and params["l2_leaf_reg"] == 10
    assert params["allow_writing_files"] is False
    assert bundle["training_quarters"] == ["2023-01-01", "2023-04-01"]
    assert bundle["group_sizes"] == {"2023-01-01": 24, "2023-04-01": 24}
    np.testing.assert_array_equal(scores, bundle["model"].predict(
        valid.assign(category=valid.category.fillna("<none>").astype(str)), thread_count=1))
    np.testing.assert_array_equal(scores[::-1], predict_ranker(bundle, valid.iloc[::-1]))
    saved = BytesIO()
    joblib.dump(bundle, saved)
    saved.seek(0)
    np.testing.assert_array_equal(scores, predict_ranker(joblib.load(saved), valid))
    pdt.assert_frame_equal(x, before)


def test_map_ranker_quarter_sort_is_stable_and_keeps_targets_aligned():
    x, y, quarters, valid = fixture()
    order = np.argsort(quarters, kind="stable")
    a = fit_ranker(x, y, quarters, ["category"], iterations=4, threads=1)
    b = fit_ranker(x.iloc[order], y[order], quarters[order], ["category"], iterations=4, threads=1)
    np.testing.assert_array_equal(predict_ranker(a, valid), predict_ranker(b, valid))
    # A single-class quarter is allowed when another group supplies both labels.
    changed = y.copy()
    changed[quarters == "2023-01-01"] = 0
    c = fit_ranker(x, changed, quarters, ["category"], iterations=2, threads=1)
    assert np.isfinite(predict_ranker(c, valid)).all()


def test_map_ranker_inner_selection_fits_once_and_uses_exact_sklearn_ap():
    x, y, quarters, valid = fixture()
    inner_y = np.array([1, 0, 0, 1])
    original = map_ranking.fit_ranker
    with patch.object(map_ranking, "fit_ranker", wraps=original) as fitting:
        selected, curve = choose_ranker_iterations(x, y, quarters, valid, inner_y,
                                                   ["category"], max_iterations=5,
                                                   step=2, threads=1)
        assert fitting.call_count == 1
    assert [row["iterations"] for row in curve] == [2, 4, 5]
    maximum = original(x, y, quarters, ["category"], iterations=5, threads=1)
    prepared = valid.copy()
    prepared["category"] = prepared.category.fillna("<none>").astype(str)
    for row in curve:
        scores = maximum["model"].predict(prepared, ntree_end=row["iterations"], thread_count=1)
        expected = average_precision_score(inner_y, scores)
        assert row["ap"] == expected
    assert selected == max(curve, key=lambda row: (row["ap"], -row["iterations"]))["iterations"]
    # The helper reports only sklearn AP; it does not select a CatBoost MAP value.
    assert all(set(row) == {"iterations", "ap"} for row in curve)


def test_map_ranker_rejects_raw_ids_dates_bad_groups_targets_and_schema():
    x, y, quarters, valid = fixture()
    for name in ["target", "row_id", "site_id", "forecast_period", "history_before", "inspection_week"]:
        assert_value_error(lambda name=name: fit_ranker(x.assign(**{name: "bad"}), y,
                                                       quarters, ["category"], iterations=1))
    assert_value_error(lambda: fit_ranker(x.assign(raw_date=pd.Timestamp("2023-01-01")),
                                         y, quarters, ["category"], iterations=1))
    for bad in [np.zeros(len(y)), np.full(len(y), np.nan), y[:-1], y.reshape(-1, 1)]:
        assert_value_error(lambda bad=bad: fit_ranker(x, bad, quarters, ["category"], iterations=1))
    for bad in [quarters[:-1], np.array(["2023-02-01"] * len(x)),
                np.array([None] * len(x)), quarters.reshape(-1, 1)]:
        assert_value_error(lambda bad=bad: fit_ranker(x, y, bad, ["category"], iterations=1))
    bundle = fit_ranker(x, y, quarters, ["category"], iterations=1, threads=1)
    assert_value_error(lambda: predict_ranker(bundle, valid[["empty", "value", "category"]]))
    assert_value_error(lambda: predict_ranker(bundle, valid.assign(target=0)))
    assert_value_error(lambda: fit_ranker(x, y, quarters, ["missing"], iterations=1))
    for bad in [0, -1, 1.5, True]:
        assert_value_error(lambda bad=bad: fit_ranker(x, y, quarters, ["category"], iterations=bad))
        assert_value_error(lambda bad=bad: choose_ranker_iterations(x, y, quarters, valid,
                           [1, 0, 0, 1], ["category"], max_iterations=1, step=bad))
    assert_value_error(lambda: choose_ranker_iterations(x, y, quarters, valid,
                                                       [0, 0, 0, 0], ["category"], max_iterations=1))
