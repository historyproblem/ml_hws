"""Eight bounded CPU boosting hypotheses with inner logloss tree selection.

The caller owns all temporal splits and supplies the same causal feature view.
No validation frame is passed to model.fit. A single maximum-size inner model
supplies a fixed 50-tree prefix grid; logloss selects the count, sklearn AP is
reported only. Predictions retain native missing values and genuine categories.
"""
from __future__ import annotations

from copy import deepcopy
import json
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss


XGBOOST_VERSION = "3.4.1"
FORBIDDEN_COLUMNS = frozenset({
    "target", "row_id", "site_id", "forecast_period", "history_before", "inspection_week",
})


def _spec(family, **params):
    return {"family": family, "params": params, "max_iterations": 1200}


_CB = dict(learning_rate=.04, rsm=1., bootstrap_type="MVS", subsample=.8,
           grow_policy="SymmetricTree", border_count=254)
_XGB = dict(learning_rate=.03, subsample=.8, colsample_bytree=.8, max_bin=256)
SPECS = {
    "cb_ordered_d5": _spec("catboost", boosting_type="Ordered", depth=5,
                            l2_leaf_reg=10., **_CB),
    "cb_plain_d3": _spec("catboost", boosting_type="Plain", depth=3,
                          l2_leaf_reg=3., **_CB),
    "cb_plain_d6": _spec("catboost", boosting_type="Plain", depth=6,
                          l2_leaf_reg=10., **_CB),
    "cb_plain_d5_low_l2": _spec("catboost", boosting_type="Plain", depth=5,
                                 l2_leaf_reg=3., **_CB),
    "xgb_d2": _spec("xgboost", max_depth=2, min_child_weight=1., reg_lambda=1., **_XGB),
    "xgb_d4": _spec("xgboost", max_depth=4, min_child_weight=5., reg_lambda=3., **_XGB),
    "xgb_d6": _spec("xgboost", max_depth=6, min_child_weight=10., reg_lambda=10., **_XGB),
    "xgb_d3_reg": _spec("xgboost", max_depth=3, min_child_weight=5., reg_lambda=10., **_XGB),
}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _frame(x, cats, features=None):
    if (not isinstance(x, pd.DataFrame) or not x.columns.is_unique
            or not all(isinstance(c, str) for c in x) or not len(x.columns)):
        raise ValueError("Features must be a DataFrame with unique string column names")
    if FORBIDDEN_COLUMNS.intersection(x.columns):
        raise ValueError("Targets and raw row/site/date identifiers cannot be features")
    if len(cats) != len(set(cats)) or not set(cats).issubset(x.columns):
        raise ValueError("Categories must be unique feature column names")
    if features is not None and list(x) != list(features):
        raise ValueError("Prediction feature names/order differ from fit")
    if any(pd.api.types.is_datetime64_any_dtype(x[c]) for c in x):
        raise ValueError("Raw datetime feature values are forbidden")
    result = x.copy(deep=True)
    numeric = [name for name in x if name not in cats]
    result[numeric] = result[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    return result


def _labels(y, n, name):
    values = np.asarray(y)
    if values.ndim != 1 or len(values) != n or not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} labels must be aligned one-dimensional binary values")
    if set(np.unique(values)) != {0, 1}:
        raise ValueError(f"{name} requires both binary classes")
    return values.astype(np.int64, copy=True)


def _transform(bundle, x):
    prepared = _frame(x, bundle["cats"], bundle["features"])
    if bundle["family"] == "catboost":
        for name in bundle["cats"]:
            prepared[name] = prepared[name].fillna("<none>").astype(str)
    elif bundle["family"] == "xgboost":
        for name, vocabulary in bundle["preprocessor"].items():
            # Preserve natural missing values; only previously observed names
            # have codes. Unknowns become categorical code -1 (native missing).
            if vocabulary:
                prepared[name] = pd.Categorical(prepared[name].astype("string"), categories=vocabulary)
            else:
                # XGB3.4 rejects a categorical dictionary with zero entries.
                # An entirely missing fit field has no known future categories,
                # so keep it entirely native NaN at fit and prediction alike.
                prepared[name] = np.nan
    else:
        raise ValueError("Unknown boosting family")
    return prepared


def fit_booster(spec, x, y, cats, *, seed=42, threads=4, iterations=None):
    """Fit one CPU model and save all preprocessing and effective parameters."""
    seed = _integer(seed, "seed", minimum=0)
    threads = _integer(threads, "threads")
    maximum = _integer(spec["max_iterations"], "max_iterations")
    count = maximum if iterations is None else _integer(iterations, "iterations")
    if count > maximum:
        raise ValueError("Iteration count exceeds the frozen maximum")
    cats = list(cats)
    prepared = _frame(x, cats)
    labels = _labels(y, len(x), "Fit")
    bundle = {"family": spec["family"], "features": list(x), "cats": cats,
              "preprocessor": None, "seed": seed, "iterations": count,
              "training_threads": threads, "inference_threads": threads,
              "spec": deepcopy(spec), "selection_metric": "sklearn log_loss (minimum)",
              "report_metric": "sklearn average_precision_score"}
    params = dict(spec["params"])
    if spec["family"] == "catboost":
        from catboost import CatBoostClassifier
        import catboost

        model = CatBoostClassifier(iterations=count, loss_function="Logloss",
                                   task_type="CPU", thread_count=threads,
                                   random_seed=seed, verbose=False,
                                   allow_writing_files=False, **params)
        prepared = _transform(bundle, prepared)
        model.fit(prepared, labels, cat_features=cats)
        bundle["effective_params"] = model.get_all_params()
        bundle["library_version"] = catboost.__version__
    elif spec["family"] == "xgboost":
        import xgboost

        if xgboost.__version__ != XGBOOST_VERSION:
            raise ValueError(f"This experiment requires xgboost=={XGBOOST_VERSION}")
        bundle["preprocessor"] = {
            name: sorted(prepared[name].astype("string").dropna().unique().tolist())
            for name in cats}
        bundle["all_missing_categorical_columns"] = [
            name for name, vocabulary in bundle["preprocessor"].items() if not vocabulary]
        prepared = _transform(bundle, prepared)
        model = xgboost.XGBClassifier(n_estimators=count, objective="binary:logistic",
                                     tree_method="hist", device="cpu", enable_categorical=True,
                                     sampling_method="uniform", random_state=seed,
                                     n_jobs=threads, eval_metric="logloss", verbosity=0, **params)
        model.fit(prepared, labels, verbose=False)
        # XGB get_params includes a NaN missing sentinel. The native config is
        # JSON-safe and records the actual CPU histogram learner configuration.
        bundle["effective_params"] = json.loads(model.get_booster().save_config())
        bundle["library_version"] = xgboost.__version__
        bundle["categorical_handling"] = "fit-only string vocabulary; unknown/natural missing -> native NaN"
    else:
        raise ValueError("Unknown boosting family")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected binary target class order")
    bundle["model"] = model
    return bundle


def predict_booster(bundle, x, iterations=None):
    """Return class-1 probabilities; optional iterations selects a tree prefix."""
    count = bundle["iterations"] if iterations is None else _integer(iterations, "iterations")
    if count > bundle["iterations"]:
        raise ValueError("Requested tree prefix exceeds the fitted model")
    prepared = _transform(bundle, x)
    if not len(x):
        return np.empty(0, dtype=np.float64)
    if bundle["family"] == "catboost":
        values = bundle["model"].predict_proba(prepared, ntree_end=count,
                                                thread_count=bundle["inference_threads"])
    elif bundle["family"] == "xgboost":
        # Actual XGBoost3.x API: exclusive iteration_range, not ntree_limit.
        values = bundle["model"].predict_proba(prepared, iteration_range=(0, count))
    else:
        raise ValueError("Unknown boosting family")
    probability = np.asarray(values[:, 1], dtype=np.float64)
    if (probability.shape != (len(x),) or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("Invalid probability output")
    return probability


def choose_tree_count(spec, xfit, yfit, xeval, yeval, cats, seed=42, threads=4,
                      max_iterations=None):
    """Fit once to the cap, then choose the earliest minimum-logloss prefix.

    This helper only receives the earlier inner pair from the caller. AP is
    retained in each curve record as a diagnostic, never a tree-count selector.
    """
    maximum = _integer(spec["max_iterations"], "max_iterations")
    if max_iterations is not None:
        maximum = _integer(max_iterations, "max_iterations")
        if maximum > spec["max_iterations"]:
            raise ValueError("Inner cap exceeds the frozen specification")
    cats = list(cats)
    _frame(xeval, cats, list(xfit))
    labels = _labels(yeval, len(xeval), "Inner validation")
    bundle = fit_booster(spec, xfit, yfit, cats, seed=seed, threads=threads, iterations=maximum)
    grid = sorted(set(range(50, maximum + 1, 50)) | {maximum})
    curve = []
    for count in grid:
        prediction = predict_booster(bundle, xeval, iterations=count)
        curve.append({"iterations": count, "logloss": float(log_loss(labels, prediction, labels=[0, 1])),
                      "ap": float(average_precision_score(labels, prediction))})
    selected = min(curve, key=lambda row: (row["logloss"], row["iterations"]))
    return int(selected["iterations"]), curve
