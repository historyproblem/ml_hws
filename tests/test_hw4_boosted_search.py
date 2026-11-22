"""Synthetic-only tests of in-time DART checkpoints and native preprocessing."""
from copy import deepcopy
from io import BytesIO
import warnings

import joblib
import numpy as np
import pandas as pd
import pandas.testing as pdt
from sklearn.metrics import average_precision_score, log_loss

from hw4_inspections import boosted_search as backend
from hw4_inspections.boosted_search import RECIPES, fit_fixed, fit_path, predict, select_columns


def expect_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def fixture():
    rng = np.random.default_rng(71)
    x = pd.DataFrame(dict(value=rng.normal(size=320), empty=np.nan,
                          category=["a", "b", None, "c"] * 80,
                          code01_count=rng.integers(0, 4, size=320)))
    y = (x.value.to_numpy() + .4 * rng.normal(size=len(x)) > 0).astype(int)
    x.loc[::19, "value"] = np.inf
    valid = pd.DataFrame(dict(value=[.5, .5, np.nan, -1., 2., -2.], empty=np.nan,
                              category=["unknown", None, "a", "b", "c", "a"], code01_count=[0, 0, 1, 2, 0, 1]),
                         index=[7, 1, 7, 3, 4, 5])
    return x, y, valid, np.array([1, 0, 1, 0, 1, 0])


def test_boosted_recipe_grid_exact24_and_view_removes_only_code_prefix():
    assert len(RECIPES) == 24
    assert sum(r["family"] == "lgb_gbdt" for r in RECIPES.values()) == 16
    assert sum(r["family"] == "lgb_dart" for r in RECIPES.values()) == 4
    assert sum(r["family"] == "catboost" for r in RECIPES.values()) == 4
    assert all(r["cap"] == 1600 and r["id"] == n for n, r in RECIPES.items())
    x, _, _, _ = fixture()
    assert select_columns(x, "no_codes") == ["value", "empty", "category"]
    assert select_columns(x.assign(ctx_code_mean=0), "no_codes")[-1] == "ctx_code_mean"
    assert select_columns(x, "full") == list(x)
    assert RECIPES["gbdt_full_07"]["params"]["extra_trees"] is True
    for name in ["cb_bayesian_d4_l2_3", "cb_bayesian_d6_l2_10"]:
        assert "subsample" not in RECIPES[name]["params"]


def test_boosted_native_families_reload_unknowns_missing_order_and_views():
    x, y, valid, _ = fixture()
    original = x.copy(deep=True)
    for name in ["gbdt_full_01", "gbdt_no_codes_03", "dart_full_01", "cb_bernoulli_d5_l2_3", "cb_bayesian_d4_l2_3"]:
        recipe = RECIPES[name]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bundle = fit_fixed(recipe, x, y, ["category"], count=4, threads=2)
            expected = predict(bundle, valid)
        assert not caught, [str(w.message) for w in caught]
        assert expected.dtype == np.float64 and np.isfinite(expected).all()
        assert bundle["threads"] == 2
        stream = BytesIO()
        joblib.dump(bundle, stream)
        stream.seek(0)
        np.testing.assert_array_equal(expected, predict(joblib.load(stream), valid))
        np.testing.assert_array_equal(expected[::-1], predict(bundle, valid.iloc[::-1]))
        np.testing.assert_array_equal(expected, predict(bundle, valid[bundle["features"]]))
        pdt.assert_frame_equal(original, x)
        if bundle["family"].startswith("lgb_"):
            assert bundle["preprocessor"]["category"] == ["a", "b", "c"]
            assert backend._transform(bundle, valid).category.cat.codes.tolist()[:2] == [-1, -1]
            assert bundle["library_version"] == "4.6.0"
            assert bundle["effective_params"]["zero_as_missing"] is False
            assert bundle["effective_params"]["deterministic"] is True
            np.testing.assert_array_equal(expected[:1], expected[1:2])


def test_boosted_dart_checkpoint_is_state_at_N_not_final_model_prefix():
    x, y, valid, yvalid = fixture()
    recipe = RECIPES["dart_full_01"]
    result = fit_path(recipe, x, y, valid, yvalid, ["category"], cap=80, grid=[40], threads=1)
    assert result["best_count"] == 40 and result["bundle"]["trained_cap"] == 80
    expected = predict(fit_fixed(recipe, x, y, ["category"], count=40, threads=1), valid)
    np.testing.assert_array_equal(predict(result["bundle"], valid), expected)
    maximum = fit_fixed(recipe, x, y, ["category"], count=80, threads=1)
    wrong = maximum["model"].predict(backend._transform(maximum, valid), num_iteration=40, num_threads=1)
    assert np.max(np.abs(expected - wrong)) > 1e-10
    assert result["curve"][0]["ap"] == average_precision_score(yvalid, expected)
    assert result["curve"][0]["logloss"] == log_loss(yvalid, expected, labels=[0, 1])


def test_boosted_cb_and_gbdt_inner_paths_return_actual_best_checkpoint():
    x, y, valid, yvalid = fixture()
    for name in ["gbdt_full_01", "cb_bernoulli_d5_l2_3"]:
        result = fit_path(RECIPES[name], x, y, valid, yvalid, ["category"], cap=9, grid=[3, 6, 9], threads=1)
        assert [row["count"] for row in result["curve"]] == [3, 6, 9]
        chosen = max(result["curve"], key=lambda r: (r["ap"], -r["count"]))
        assert result["best_count"] == chosen["count"]
        p = predict(result["bundle"], valid)
        assert chosen["ap"] == average_precision_score(yvalid, p)
        assert chosen["logloss"] == log_loss(yvalid, p, labels=[0, 1])
        direct = fit_fixed(RECIPES[name], x, y, ["category"], count=chosen["count"], threads=1)
        np.testing.assert_array_equal(p, predict(direct, valid))


def test_boosted_bounds_schema_targets_and_thread_budget():
    x, y, valid, yvalid = fixture()
    for column in backend.FORBIDDEN:
        expect_error(lambda column=column: select_columns(x.assign(**{column: 0}), "no_codes"))
    expect_error(lambda: select_columns(x, "unknown"))
    expect_error(lambda: fit_fixed(RECIPES["gbdt_full_01"], x, y, ["category"], count=1, threads=3))
    expect_error(lambda: fit_fixed(RECIPES["gbdt_full_01"], x, np.zeros(len(y)), ["category"], count=1))
    for grid in [[0, 3], [3, 2], [3, 3], [10], []]:
        expect_error(lambda grid=grid: fit_path(RECIPES["gbdt_full_01"], x, y, valid, yvalid,
                                               ["category"], cap=9, grid=grid))
    model = fit_fixed(RECIPES["gbdt_no_codes_01"], x, y, ["category"], count=2)
    expect_error(lambda: predict(model, valid[list(valid)[::-1]]))
    expect_error(lambda: predict(model, valid.assign(extra=1)))
    expect_error(lambda: fit_fixed(RECIPES["gbdt_full_01"], x, y, ["category"], count=1601))


def test_boosted_empty_categorical_fit_maps_future_values_to_nan_and_zero_valid():
    x, y, valid, _ = fixture()
    x["category"] = None
    x["zero_valid"] = 0.
    valid["zero_valid"] = 0.
    model = fit_fixed(RECIPES["gbdt_full_01"], x, y, ["category"], count=3)
    prepared = backend._transform(model, valid)
    assert model["preprocessor"]["category"] == [] and prepared.category.isna().all()
    assert prepared.zero_valid.eq(0).all() and not prepared.zero_valid.isna().any()
    assert np.isfinite(predict(model, valid)).all()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS", flush=True)
