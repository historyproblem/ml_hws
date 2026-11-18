"""Small meaningful preprocessing, serialization and tree-selection checks."""
from copy import deepcopy
from importlib.util import find_spec
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from hw4_inspections.classical_models import (
    SPECS, choose_iterations, fit_model, predict_model,
)


def _fixture():
    rng = np.random.default_rng(42)
    x = pd.DataFrame({"value": rng.normal(size=80),
                      "empty": [np.nan] * 80,
                      "category": ["a", "b", "a", None] * 20})
    x.loc[::7, "value"] = np.nan
    y = (np.nan_to_num(x.value.to_numpy()) + rng.normal(size=80) > 0).astype(int)
    valid = pd.DataFrame({"value": [1., np.nan, -1.], "empty": [np.nan, 9., 0.],
                          "category": ["brand-new", "a", None]}, index=[9, 3, 5])
    return x, y, valid


def _tiny(name):
    spec = deepcopy(SPECS[name])
    spec["max_iterations"] = 5
    if spec["family"] == "lightgbm":
        spec["params"]["min_child_samples"] = 3
    if spec["family"] == "extra_trees":
        spec["params"]["min_samples_leaf"] = 2
    return spec


def test_classical_preprocessing_reload_unknown_and_row_order():
    x, y, valid = _fixture()
    original = x.copy(deep=True)
    names = ["cb_reference", "et_compact"]
    if find_spec("lightgbm"):
        names.append("lgb_small")
    for name in names:
        bundle = fit_model(_tiny(name), x, y, ["category"], 42, 1)
        expected = predict_model(bundle, valid)
        payload = BytesIO()
        joblib.dump(bundle, payload)
        payload.seek(0)
        reloaded = joblib.load(payload)
        np.testing.assert_array_equal(expected, predict_model(reloaded, valid))
        np.testing.assert_array_equal(expected[::-1], predict_model(bundle, valid.iloc[::-1]))
        assert np.isfinite(expected).all()
        if name == "lgb_small":
            assert "brand-new" not in bundle["preprocessor"]["category"]
        pd.testing.assert_frame_equal(original, x)
        try:
            predict_model(bundle, valid[["empty", "value", "category"]])
            raise AssertionError("Changed feature order accepted")
        except ValueError:
            pass
def test_classical_inner_selection_ap_matches_explicit_prefix_predictions():
    from sklearn.metrics import average_precision_score
    x, y, _ = _fixture()
    names = ["cb_regularized"]
    if find_spec("lightgbm"):
        names.append("lgb_small")
    for name in names:
        spec = _tiny(name)
        count, curve = choose_iterations(spec, x.iloc[:60], y[:60], x.iloc[60:],
                                        y[60:], ["category"], 42, 1, smoke=True)
        assert [r["iterations"] for r in curve] == [1, 2, 3, 4, 5]
        for row in curve:
            bundle = fit_model(spec, x.iloc[:60], y[:60], ["category"], 42, 1,
                               iterations=row["iterations"])
            # For LGBM every prefix of the same fit and a fresh N-tree fit have
            # identical seeded sampling; for CB, defaults are pinned explicitly.
            actual = average_precision_score(y[60:], predict_model(bundle, x.iloc[60:]))
            assert abs(row["ap"] - actual) < 1e-14
        assert count == max(curve, key=lambda r: (r["ap"], -r["iterations"]))["iterations"]
    fixed, curve = choose_iterations(_tiny("cb_reference"), x, y, None, None,
                                    ["category"], 42, 1)
    assert fixed == 5 and curve == []
    partial = deepcopy(SPECS["cb_regularized"])
    partial["max_iterations"] = 55
    _, curve = choose_iterations(partial, x.iloc[:60], y[:60], x.iloc[60:],
                                 y[60:], ["category"], 42, 1)
    assert [r["iterations"] for r in curve] == [50, 55]


def test_classical_rejects_identifiers_and_invalid_weights():
    x, y, _ = _fixture()
    for bad in [x.assign(target=y), x.assign(row_id=np.arange(len(x))),
                x.assign(site_id="site")]:
        try:
            fit_model(_tiny("cb_reference"), bad, y, ["category"], 42, 1)
            raise AssertionError("Forbidden identifier accepted")
        except ValueError:
            pass
    for weights in [np.zeros(len(x)), np.ones(len(x) - 1), np.full(len(x), np.nan)]:
        try:
            fit_model(_tiny("cb_reference"), x, y, ["category"], 42, 1,
                      sample_weight=weights)
            raise AssertionError("Invalid weights accepted")
        except ValueError:
            pass


def test_extra_trees_parallel_fit_deterministic_prediction_and_reload():
    x, y, valid = _fixture()
    spec = deepcopy(SPECS["et_compact"])
    spec["max_iterations"] = 97
    spec["params"]["min_samples_leaf"] = 1
    bundle = fit_model(spec, x, y, ["category"], 42, 4)
    assert bundle["training_threads"] == 4
    assert bundle["inference_threads"] == 1
    assert bundle["model"].get_params()["n_jobs"] == 1
    probes = pd.concat([x, valid], ignore_index=True)
    expected = predict_model(bundle, probes)
    for _ in range(5):
        np.testing.assert_array_equal(expected, predict_model(bundle, probes))
    payload = BytesIO()
    joblib.dump(bundle, payload)
    payload.seek(0)
    restored = joblib.load(payload)
    np.testing.assert_array_equal(expected, predict_model(restored, probes))
    np.testing.assert_array_equal(expected[::-1], predict_model(restored, probes.iloc[::-1]))
