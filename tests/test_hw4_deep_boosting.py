"""Tiny meaningful checks for bounded CatBoost/CPU-XGB and logloss selection."""
from copy import deepcopy
from io import BytesIO
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd
import pandas.testing as pdt
from sklearn.metrics import average_precision_score, log_loss

from hw4_inspections import deep_boosting
from hw4_inspections.deep_boosting import SPECS, choose_tree_count, fit_booster, predict_booster


def assert_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def fixture():
    rng = np.random.default_rng(12)
    x = pd.DataFrame({"value": rng.normal(size=100), "empty": [np.nan] * 100,
                      "category": ["a", "b", None, "c"] * 25,
                      "zero_valid": [0., 1., 0., 2.] * 25})
    x.loc[::11, "value"] = np.inf
    y = (np.nan_to_num(x.value.to_numpy(), posinf=0) + rng.normal(size=100) > 0).astype(int)
    valid = pd.DataFrame({"value": [0., 0., np.nan, -2.], "empty": [np.nan, np.nan, 1., 0.],
                          "category": ["unknown", None, "a", "b"], "zero_valid": [0., 0., 0., 2.]},
                         index=[8, 2, 8, 1])
    return x, y, valid


def tiny(name, count=5):
    spec = deepcopy(SPECS[name])
    spec["max_iterations"] = count
    return spec


def test_deep_specs_are_exact_and_bounded():
    assert len(SPECS) == 8
    assert sum(s["family"] == "catboost" for s in SPECS.values()) == 4
    assert all(s["max_iterations"] == 1200 for s in SPECS.values())
    assert SPECS["cb_ordered_d5"]["params"]["boosting_type"] == "Ordered"
    assert SPECS["cb_plain_d5_low_l2"]["params"]["l2_leaf_reg"] == 3
    assert all(s["params"]["rsm"] == 1 for s in SPECS.values() if s["family"] == "catboost")
    assert [(SPECS[n]["params"]["max_depth"], SPECS[n]["params"]["min_child_weight"],
             SPECS[n]["params"]["reg_lambda"]) for n in ["xgb_d2", "xgb_d4", "xgb_d6", "xgb_d3_reg"]] == [
                 (2, 1, 1), (4, 5, 3), (6, 10, 10), (3, 5, 10)]


def test_deep_tiny_families_ordered_native_missing_reload_and_order():
    x, y, valid = fixture()
    before = x.copy(deep=True)
    for name in ["cb_ordered_d5", "cb_plain_d3", "xgb_d2", "xgb_d4"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bundle = fit_booster(tiny(name), x, y, ["category"], threads=1)
            probability = predict_booster(bundle, valid)
        assert not caught, [str(w.message) for w in caught]
        assert probability.dtype == np.float64 and probability.shape == (4,)
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        payload = BytesIO()
        joblib.dump(bundle, payload)
        payload.seek(0)
        np.testing.assert_array_equal(probability, predict_booster(joblib.load(payload), valid))
        np.testing.assert_array_equal(probability[::-1], predict_booster(bundle, valid.iloc[::-1]))
        pdt.assert_frame_equal(x, before)
        if name.startswith("xgb"):
            assert bundle["preprocessor"]["category"] == ["a", "b", "c"]
            assert "unknown" not in bundle["preprocessor"]["category"]
            np.testing.assert_array_equal(probability[:1], probability[1:2])
            assert bundle["library_version"] == "3.4.1"
        if name == "cb_ordered_d5":
            assert bundle["effective_params"]["boosting_type"] == "Ordered"


def test_deep_all_eight_frozen_configs_accept_tiny_cpu_fit():
    x, y, valid = fixture()
    for name in SPECS:
        bundle = fit_booster(tiny(name, 2), x, y, ["category"], threads=1)
        assert np.isfinite(predict_booster(bundle, valid)).all()


def test_deep_repeated_multithread_cpu_fits_are_exact():
    x, y, valid = fixture()
    for name in ["cb_ordered_d5", "xgb_d2"]:
        first = fit_booster(tiny(name, 8), x, y, ["category"], threads=4)
        second = fit_booster(tiny(name, 8), x, y, ["category"], threads=4)
        np.testing.assert_array_equal(predict_booster(first, valid), predict_booster(second, valid))


def test_deep_xgb_categories_are_native_missing_and_zero_is_not_missing():
    x, y, valid = fixture()
    bundle = fit_booster(tiny("xgb_d2"), x, y, ["category"], threads=1)
    prepared = deep_boosting._transform(bundle, valid)
    assert isinstance(prepared.category.dtype, pd.CategoricalDtype)
    assert prepared.category.cat.codes.tolist() == [-1, -1, 0, 1]
    assert prepared.zero_valid.iloc[0] == 0 and not pd.isna(prepared.zero_valid.iloc[0])
    assert pd.isna(prepared.value.iloc[2])
    all_missing = x.assign(category=None)
    empty_vocab = fit_booster(tiny("xgb_d2"), all_missing, y, ["category"], threads=1)
    assert empty_vocab["preprocessor"]["category"] == []
    assert np.isfinite(predict_booster(empty_vocab, valid)).all()


def test_deep_inner_one_fit_prefixes_match_logloss_and_ap_exactly():
    x, y, valid = fixture()
    inner_y = np.array([1, 0, 0, 1])
    original = deep_boosting.fit_booster
    for name in ["cb_ordered_d5", "xgb_d2"]:
        spec = tiny(name, 55)
        with patch.object(deep_boosting, "fit_booster", wraps=original) as training:
            selected, curve = choose_tree_count(spec, x, y, valid, inner_y, ["category"], threads=1)
            assert training.call_count == 1
        assert [row["iterations"] for row in curve] == [50, 55]
        model = original(spec, x, y, ["category"], threads=1)
        for row in curve:
            p = predict_booster(model, valid, iterations=row["iterations"])
            assert row["logloss"] == log_loss(inner_y, p, labels=[0, 1])
            assert row["ap"] == average_precision_score(inner_y, p)
        assert selected == min(curve, key=lambda row: (row["logloss"], row["iterations"]))["iterations"]


def test_deep_tree_selection_uses_logloss_not_ap_and_breaks_ties_early():
    x, y, valid = fixture()
    labels = np.array([1, 0, 0, 1])
    better_ap_bad_loss = np.array([.99, .98, .97, .995])
    better_loss = np.array([.8, .2, .5, .4])
    # Explicit synthetic scorer outputs isolate the selector, with no extra fit.
    with patch.object(deep_boosting, "fit_booster", return_value={}), patch.object(
            deep_boosting, "predict_booster", side_effect=[better_ap_bad_loss, better_loss]):
        selected, curve = choose_tree_count(tiny("cb_plain_d3", 100), x, y, valid, labels, ["category"])
    assert curve[0]["ap"] > curve[1]["ap"]
    assert curve[1]["logloss"] < curve[0]["logloss"] and selected == 100
    with patch.object(deep_boosting, "fit_booster", return_value={}), patch.object(
            deep_boosting, "predict_booster", return_value=better_loss):
        selected, _ = choose_tree_count(tiny("cb_plain_d3", 100), x, y, valid, labels, ["category"])
    assert selected == 50


def test_deep_guards_target_ids_schema_labels_and_caps():
    x, y, valid = fixture()
    for name in ["target", "row_id", "site_id", "forecast_period", "history_before", "inspection_week"]:
        assert_value_error(lambda name=name: fit_booster(tiny("cb_plain_d3"), x.assign(**{name: 0}),
                                                         y, ["category"], threads=1))
    for bad in [y[:-1], np.zeros(len(y)), y.reshape(-1, 1), np.full(len(y), np.nan)]:
        assert_value_error(lambda bad=bad: fit_booster(tiny("cb_plain_d3"), x, bad, ["category"], threads=1))
    model = fit_booster(tiny("xgb_d2"), x, y, ["category"], threads=1)
    assert_value_error(lambda: predict_booster(model, valid[list(valid)[::-1]]))
    assert_value_error(lambda: predict_booster(model, valid.assign(target=0)))
    for count in [0, 6, -1, 1.5, True]:
        assert_value_error(lambda count=count: predict_booster(model, valid, iterations=count))
    assert_value_error(lambda: choose_tree_count(tiny("cb_plain_d3"), x, y, valid,
                                                [0, 0, 0, 0], ["category"], threads=1))
    assert_value_error(lambda: choose_tree_count(tiny("cb_plain_d3"), x, y, valid,
                                                [0, 1, 0, 1], ["category"], max_iterations=6, threads=1))
