"""Synthetic-only RBF geometry, convergence and native probability checks."""
from copy import deepcopy
from io import BytesIO
from unittest.mock import patch
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from hw4_inspections import rbf_search as backend
from hw4_inspections.rbf_search import RBF_RECIPES, RBFTransform, fit_rbf, predict_rbf, select_columns
from test_hw4_ebm_search import fixture, expect_error


def test_rbf_exact8_recipes_views_and_geometry_pipeline():
    assert len(RBF_RECIPES) == 8
    assert {r["C"] for r in RBF_RECIPES.values()} == {.5, 5}
    assert {r["gamma_multiplier"] for r in RBF_RECIPES.values()} == {.25, 1}
    x, _, cats, _ = fixture()
    assert len(select_columns(x, "own22")) == 22
    assert len(select_columns(x, "compact94")) == 94
    transform = RBFTransform().fit(x, cats, "compact94")
    assert transform.categorical == ["facility_type", "risk_level", "area_id", "forecast_quarter"]
    assert all(name.endswith("fail_smooth") for name in transform.numeric
               if name not in transform.log_count_names and name != backend.SIGNED)
    assert transform.transform(x).dtype == np.float64 and transform.transform(x).flags.c_contiguous


def test_rbf_native_probability_gamma_reload_order_unknown_and_zero_valid():
    x, y, cats, valid = fixture()
    original = x.copy(deep=True)
    for name in ["rbf_own22_C0p5_g0p25", "rbf_compact94_C5_g1"]:
        bundle = fit_rbf(RBF_RECIPES[name], x, y, cats, threads=2)
        p = predict_rbf(bundle, valid)
        assert p.dtype == np.float64 and np.isfinite(p).all() and p.shape == (4,)
        assert bundle["fit_status"] == 0
        assert bundle["effective_params"]["probability"] is True
        assert bundle["effective_params"]["max_iter"] == 100000
        assert bundle["effective_params"]["tol"] == 1e-3
        assert bundle["effective_params"]["class_weight"] is None
        design = bundle["transformer"].transform(x)
        assert bundle["gamma_base"] == 1/(design.shape[1] * design.var())
        assert bundle["gamma"] == RBF_RECIPES[name]["gamma_multiplier"] * bundle["gamma_base"]
        saved = BytesIO()
        joblib.dump(bundle, saved)
        saved.seek(0)
        np.testing.assert_array_equal(p, predict_rbf(joblib.load(saved), valid))
        np.testing.assert_array_equal(p[::-1], predict_rbf(bundle, valid.iloc[::-1]))
        np.testing.assert_array_equal(p, predict_rbf(bundle, valid[bundle["selected_features"]]))
        assert x.equals(original)


def test_rbf_fit_only_future_extremes_missing_log_rules_and_OHE_unknowns():
    x, y, cats, valid = fixture()
    x[backend.SIGNED] = np.linspace(-3, 3, len(x))
    x["recent90_violation_max"] = np.nan
    bundle = fit_rbf(RBF_RECIPES["rbf_compact94_C0p5_g1"], x, y, cats)
    transform = bundle["transformer"]
    state = joblib.hash(transform)
    values, masks = transform._numeric(valid)
    count = transform.numeric.index("history_violation_mean")
    assert values[0, count] == 0 and masks[0, count] == 0
    assert masks[2, count] == 1
    signed = transform.numeric.index(backend.SIGNED)
    input_signed = valid[backend.SIGNED].to_numpy()
    np.testing.assert_allclose(values[:, signed], np.sign(input_signed)*np.log1p(np.abs(input_signed)))
    assert "recent90_violation_max" in transform.all_missing_numeric
    unseen = valid.copy()
    unseen["facility_type"] = "unseen-only"
    unseen["history_violation_mean"] = [0, 1e20, np.nan, np.inf]
    assert np.isfinite(predict_rbf(bundle, unseen)).all()
    assert joblib.hash(transform) == state
    category_design = transform.encoder.transform(transform._categorical(unseen))
    facility_indices = [i for i, n in enumerate(transform.encoder.get_feature_names_out(transform.categorical))
                        if n.startswith("facility_type_")]
    assert np.all(category_design[:, facility_indices] == 0)


def test_rbf_invalid_partial_models_warnings_and_fit_status_are_rejected():
    x, y, cats, _ = fixture()
    def nonconverged(*a, **kw):
        warnings.warn("Synthetic iteration cap reached", ConvergenceWarning)
    with patch.object(backend.SVC, "fit", side_effect=nonconverged):
        expect_error(lambda: fit_rbf(RBF_RECIPES["rbf_own22_C0p5_g1"], x, y, cats))
    original = backend.SVC.fit
    def bad_status(model, *a, **kw):
        original(model, *a, **kw)
        model.fit_status_ = 1
        return model
    with patch.object(backend.SVC, "fit", new=bad_status):
        expect_error(lambda: fit_rbf(RBF_RECIPES["rbf_own22_C0p5_g1"], x, y, cats))


def test_rbf_threads_target_schema_and_nonnegative_count_guards():
    x, y, cats, valid = fixture()
    for threads in [0, 3, -1, True]:
        expect_error(lambda threads=threads: fit_rbf(RBF_RECIPES["rbf_own22_C0p5_g1"], x, y, cats, threads=threads))
    for field in ["target", "row_id", "site_id", "forecast_period", "history_before"]:
        expect_error(lambda field=field: fit_rbf(RBF_RECIPES["rbf_own22_C0p5_g1"], x.assign(**{field: 0}), y, cats))
    for labels in [y[:-1], np.zeros(len(y)), np.full(len(y), np.nan), y.reshape(-1, 1)]:
        expect_error(lambda labels=labels: fit_rbf(RBF_RECIPES["rbf_own22_C0p5_g1"], x, labels, cats))
    bundle = fit_rbf(RBF_RECIPES["rbf_own22_C0p5_g1"], x, y, cats)
    expect_error(lambda: predict_rbf(bundle, valid[list(valid)[::-1]]))
    expect_error(lambda: predict_rbf(bundle, valid.assign(extra=0)))
    expect_error(lambda: predict_rbf(bundle, valid.assign(history_violation_mean=-1)))
    expect_error(lambda: predict_rbf(bundle, valid.assign(forecast_quarter=5)))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS", flush=True)
