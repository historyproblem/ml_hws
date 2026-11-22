"""Synthetic full-schema native EBM tests; no HW4 learning or metrics."""
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import joblib
import numpy as np
import pandas as pd
import pandas.testing as pdt

from hw4_inspections import ebm_search as backend
from hw4_inspections.ebm_search import EBM_RECIPES, FIXED_PAIRS, fit_ebm, load_runtime, predict_ebm, select_columns
from hw4_inspections.features import build_features
from hw4_inspections.extensions import build_context_features
from test_hw4_extensions import fixture as history_fixture


RUNTIME = Path(os.environ.get("HW4_EBM_RUNTIME", "/Users/khlystov/Documents/ChatGPT/ML ДЗ/analysis/hw4-six-hour-20261122/ebm-runtime"))


def expect_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def fixture():
    rows, history = history_fixture()
    base, cats = build_features(rows, history)
    frame = pd.concat([base, build_context_features(rows, history)], axis=1)
    x = pd.concat([frame.iloc[[0]]] * 80, ignore_index=True)
    # Generate a finite synthetic constant for absent inspection-type fields;
    # this is fixture construction, never production missing-value imputation.
    for name in x:
        if name not in cats and x[name].isna().all():
            x[name] = 0.
    x["history_violation_mean"] = np.arange(80) / 10
    x["last_routine_violation_count"] = np.arange(80) % 10
    x["forecast_quarter"] = np.tile([1, 2, 3, 4], 20)
    x["facility_type"] = np.tile(["restaurant", "school"], 40)
    x.loc[::9, "history_violation_mean"] = np.nan
    x.loc[::11, "facility_type"] = None
    y = (np.arange(80) % 5 >= 3).astype(int)
    valid = x.iloc[[0, 1, 2, 3]].copy()
    valid.index = [8, 2, 8, 1]
    valid.loc[:, "history_violation_mean"] = [0., 0., np.nan, 1e6]
    valid.loc[:, "facility_type"] = ["new-type", None, "school", "restaurant"]
    valid.iloc[1] = valid.iloc[0]
    valid.iloc[1, valid.columns.get_loc("facility_type")] = None
    return x, y, cats, valid


def tiny(name, rounds=2):
    result = deepcopy(EBM_RECIPES[name])
    result["params"]["max_rounds"] = rounds
    return result


def test_ebm_exact8_recipes_views_and_fixed_pair_scope():
    assert len(EBM_RECIPES) == 8
    assert {r["params"]["max_rounds"] for r in EBM_RECIPES.values()} == {300, 800}
    assert len(FIXED_PAIRS) == len(set(FIXED_PAIRS)) == 10
    x, _, _, _ = fixture()
    assert len(select_columns(x, "compact94")) == 94
    assert len(select_columns(x, "no_codes168")) == 168
    for view in ["compact94", "no_codes168"]:
        columns = select_columns(x, view)
        assert not set(columns) & {"forecast_year", "cutoff_gap_days"}
        assert "forecast_quarter" in columns
        assert all(a in columns and b in columns for a, b in FIXED_PAIRS)
    for recipe in EBM_RECIPES.values():
        p = recipe["params"]
        assert p["validation_size"] == p["early_stopping_rounds"] == p["inner_bags"] == 0
        assert p["outer_bags"] == 1 and p["n_jobs"] == 2
        assert p["interactions"] == 0 or [tuple(pair) for pair in p["interactions"]] == list(FIXED_PAIRS)


def test_ebm_isolated_runtime_version_and_path_guards():
    assert load_runtime(RUNTIME) == RUNTIME.resolve()
    with TemporaryDirectory() as temp:
        expect_error(lambda: load_runtime(Path(temp)))


def test_ebm_native_fit_missing_unknown_zero_term_scope_and_reload():
    x, y, cats, valid = fixture()
    before = x.copy(deep=True)
    for name in ["ebm_compact94_main_300", "ebm_compact94_pairs10_300", "ebm_no_codes168_main_300"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bundle = fit_ebm(tiny(name), x, y, cats, runtime=RUNTIME,
                             threads=2 if name == "ebm_compact94_main_300" else 1)
            p = predict_ebm(bundle, valid)
        assert not caught, [str(w.message) for w in caught]
        assert p.dtype == np.float64 and np.isfinite(p).all() and p.shape == (4,)
        assert bundle["library_version"] == "0.7.8"
        assert bundle["trained_rows"] == 80 and bundle["outer_labels_seen"] is False
        assert len(bundle["terms"]) == len(bundle["features"]) + (10 if "pairs10" in name else 0)
        quarter = bundle["features"].index("forecast_quarter")
        assert bundle["feature_types"][quarter] == "nominal"
        assert set(bundle["vocabularies"]["forecast_quarter"]) == {"1", "2", "3", "4"}
        prepared = backend._transform(bundle, valid)
        assert prepared.facility_type.iloc[:2].isna().all()
        assert prepared.history_violation_mean.iloc[0] == 0 and not pd.isna(prepared.history_violation_mean.iloc[0])
        assert pd.isna(prepared.history_violation_mean.iloc[2])
        np.testing.assert_array_equal(p[:1], p[1:2])
        payload = BytesIO()
        joblib.dump(bundle, payload)
        payload.seek(0)
        np.testing.assert_array_equal(p, predict_ebm(joblib.load(payload), valid))
        np.testing.assert_array_equal(p[::-1], predict_ebm(bundle, valid.iloc[::-1]))
        np.testing.assert_array_equal(p, predict_ebm(bundle, valid[bundle["features"]]))
        pdt.assert_frame_equal(x, before)


def test_ebm_prediction_extremes_cannot_refit_bins_or_category_vocabulary():
    x, y, cats, valid = fixture()
    model = fit_ebm(tiny("ebm_compact94_main_800", 3), x, y, cats, runtime=RUNTIME, threads=1)
    bins = deepcopy(model["model"].bins_)
    vocabulary = deepcopy(model["vocabularies"])
    bounds = model["model"].feature_bounds_.copy()
    modified = valid.copy()
    modified["history_violation_mean"] = [-1e30, 1e30, np.inf, np.nan]
    modified["facility_type"] = ["unseen-a", "unseen-b", None, "unseen-c"]
    assert np.isfinite(predict_ebm(model, modified)).all()
    assert model["vocabularies"] == vocabulary
    np.testing.assert_array_equal(model["model"].feature_bounds_, bounds)
    for original, current in zip(bins, model["model"].bins_):
        for a, b in zip(original, current):
            if isinstance(a, dict):
                assert a == b
            else:
                np.testing.assert_array_equal(a, b)


def test_ebm_schema_thread_target_and_no_auto_interaction_guards():
    x, y, cats, valid = fixture()
    for name in backend.FORBIDDEN:
        expect_error(lambda name=name: select_columns(x.assign(**{name: 0}), "compact94"))
    for threads in [0, 3, -1, True]:
        expect_error(lambda threads=threads: fit_ebm(tiny("ebm_compact94_main_300"), x, y, cats,
                                                    runtime=RUNTIME, threads=threads))
    for labels in [y[:-1], np.zeros(len(y)), np.full(len(y), np.nan), y.reshape(-1, 1)]:
        expect_error(lambda labels=labels: fit_ebm(tiny("ebm_compact94_main_300"), x, labels, cats,
                                                  runtime=RUNTIME, threads=1))
    bad = tiny("ebm_compact94_pairs10_300")
    bad["params"]["interactions"] = 10
    expect_error(lambda: fit_ebm(bad, x, y, cats, runtime=RUNTIME, threads=1))
    model = fit_ebm(tiny("ebm_compact94_main_300"), x, y, cats, runtime=RUNTIME, threads=1)
    expect_error(lambda: predict_ebm(model, valid[list(valid)[::-1]]))
    expect_error(lambda: predict_ebm(model, valid.assign(extra=1)))
    expect_error(lambda: predict_ebm(model, valid.assign(forecast_quarter=5)))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS", flush=True)
