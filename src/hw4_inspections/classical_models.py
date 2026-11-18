"""Fixed CPU model hypotheses with fit-only preprocessing and inner tree choice.

The caller owns all temporal splits. ``choose_iterations`` sees only the inner
training/validation pair; the final outer fold is never passed to model.fit.
Bundles are plain joblib-serializable dictionaries, including preprocessing.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder


def _spec(family, feature_set="context", *, add_site=False, recency=False,
          max_iterations=500, tune_iterations=False, **params):
    return dict(family=family, feature_set=feature_set, add_site=add_site,
                recency=recency, max_iterations=max_iterations,
                tune_iterations=tune_iterations, params=params)


_CB = dict(depth=5, learning_rate=.04, l2_leaf_reg=10.)
_LGB = dict(learning_rate=.025, colsample_bytree=.8, subsample=.8,
            subsample_freq=1)
SPECS: dict[str, dict[str, Any]] = {
    "cb_reference": _spec("catboost", **_CB),
    "cb_compact": _spec("catboost", "compact_context", **_CB),
    "cb_regularized": _spec("catboost", max_iterations=1600,
                            tune_iterations=True, depth=4,
                            learning_rate=.025, l2_leaf_reg=20.),
    "cb_site": _spec("catboost", add_site=True, depth=5,
                     learning_rate=.04, l2_leaf_reg=30.),
    "cb_recency": _spec("catboost", recency=True, **_CB),
    "cb_relative": _spec("catboost", "context_relative", **_CB),
    "lgb_small": _spec("lightgbm", max_iterations=1600,
                       tune_iterations=True, num_leaves=15,
                       min_child_samples=120, reg_lambda=10., **_LGB),
    "lgb_medium": _spec("lightgbm", max_iterations=1600,
                        tune_iterations=True, num_leaves=31,
                        min_child_samples=80, reg_lambda=15., **_LGB),
    "et_compact": _spec("extra_trees", "compact_context",
                        max_iterations=700, min_samples_leaf=15,
                        max_features=.7),
}


def _validate_x(x, cats):
    if not isinstance(x, pd.DataFrame) or not x.columns.is_unique:
        raise ValueError("Features must be a DataFrame with unique columns")
    if not set(cats) <= set(x):
        raise ValueError("Categorical columns absent from features")
    if {"target", "row_id", "forecast_period", "history_before"} & set(x):
        raise ValueError("Target or row/date identifiers cannot be features")


def _clean(x, cats):
    """Copy input; normalize natural missing categories, preserve numeric NaN."""
    out = x.copy()
    for name in cats:
        out[name] = out[name].fillna("<none>").astype(str)
    numeric = [name for name in out if name not in cats]
    out[numeric] = out[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    return out


def _transform(bundle, x):
    if list(x.columns) != bundle["features"]:
        raise ValueError("Prediction feature names/order differ from fit")
    out = _clean(x, bundle["cats"])
    if bundle["family"] == "catboost":
        # The measured context reference used CatBoost's native missing-value
        # treatment. Preserve NaN for every CB ablation to keep it comparable.
        pass
    elif bundle["family"] == "lightgbm":
        for name, vocabulary in bundle["preprocessor"].items():
            # Unknown categories become missing, never ordinal numeric values.
            out[name] = pd.Categorical(out[name], categories=vocabulary)
    elif bundle["family"] == "extra_trees":
        out = bundle["preprocessor"].transform(out)
    else:
        raise ValueError(f"Unknown model family {bundle['family']}")
    return out


def fit_model(spec, xtrain, ytrain, cats, seed, threads, iterations=None,
              sample_weight=None):
    """Fit one model, without any evaluation set, and return its full bundle."""
    _validate_x(xtrain, cats)
    if "site_id" in xtrain and not spec["add_site"]:
        raise ValueError("site_id is allowed only in the explicit ID ablation")
    if set(np.unique(ytrain)) != {0, 1}:
        raise ValueError("Fit requires both binary target classes")
    count = int(spec["max_iterations"] if iterations is None else iterations)
    if count < 1 or count > int(spec["max_iterations"]):
        raise ValueError("Iteration count outside the frozen specification")
    if threads < 1:
        raise ValueError("threads must be positive")
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if (sample_weight.shape != (len(xtrain),) or
                not np.isfinite(sample_weight).all() or (sample_weight <= 0).any()):
            raise ValueError("Sample weights must be finite positive fit-row weights")
    bundle = dict(family=spec["family"], features=list(xtrain), cats=list(cats),
                  preprocessor=None, seed=int(seed), iterations=count,
                  training_threads=int(threads),
                  inference_threads=1 if spec["family"] == "extra_trees" else int(threads))
    clean = _clean(xtrain, cats)
    params = dict(spec["params"])
    if spec["family"] == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=count, loss_function="Logloss",
                                   random_seed=seed, thread_count=threads,
                                   task_type="CPU", verbose=False,
                                   allow_writing_files=False, **params)
        prepared = _transform(bundle, clean)
        model.fit(prepared, ytrain, cat_features=list(cats),
                  sample_weight=sample_weight)
    elif spec["family"] == "lightgbm":
        from lightgbm import LGBMClassifier
        bundle["preprocessor"] = {
            name: sorted(clean[name].unique().tolist()) for name in cats}
        prepared = _transform(bundle, clean)
        model = LGBMClassifier(n_estimators=count, objective="binary",
                               random_state=seed, n_jobs=threads,
                               deterministic=True, force_col_wise=True,
                               verbosity=-1, **params)
        model.fit(prepared, ytrain, sample_weight=sample_weight,
                  categorical_feature=list(cats))
    elif spec["family"] == "extra_trees":
        numeric = [name for name in clean if name not in cats]
        transformer = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", add_indicator=True,
                                      keep_empty_features=True), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore",
                                          sparse_output=True), list(cats)),
        ])
        bundle["preprocessor"] = transformer.fit(clean)
        prepared = transformer.transform(clean)
        model = ExtraTreesClassifier(n_estimators=count, random_state=seed,
                                     n_jobs=threads, **params)
        model.fit(prepared, ytrain, sample_weight=sample_weight)
        # Parallel forest prediction accumulates tree outputs in completion
        # order, which can change floating-point rounding after serialization.
        # Keep parallel fitting, then make every inference sum deterministic.
        model.set_params(n_jobs=1)
    else:
        raise ValueError(f"Unknown model family {spec['family']}")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected target class order")
    bundle["model"] = model
    bundle["effective_params"] = model.get_params()
    return bundle


def predict_model(bundle, x):
    """Predict class-1 probabilities using only saved fit preprocessing."""
    probability = np.asarray(bundle["model"].predict_proba(
        _transform(bundle, x))[:, 1], dtype=float)
    if (probability.shape != (len(x),) or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()):
        raise ValueError("Invalid probabilities")
    return probability


def choose_iterations(spec, xtrain, ytrain, xinner, yinner, cats, seed, threads,
                      sample_weight=None, smoke=False):
    """Choose a tree count on the earlier inner quarter using sklearn AP.

    Untuned hypotheses return their fixed count without fitting. Tuned models
    fit once to the maximum, score prefixes every 50 trees, and choose the
    earliest prefix at maximum AP. The maximum is also scored if not divisible
    by 50. ``smoke`` changes only the prefix grid, never the objective.
    """
    count = int(spec["max_iterations"])
    if not spec["tune_iterations"]:
        return count, []
    if set(np.unique(yinner)) != {0, 1}:
        raise ValueError("Inner validation requires both binary target classes")
    bundle = fit_model(spec, xtrain, ytrain, cats, seed, threads,
                       sample_weight=sample_weight)
    values = _transform(bundle, xinner)
    step = 1 if smoke else 50
    grid = sorted(set(range(step, count + 1, step)) | {count})
    curve = []
    if spec["family"] == "catboost":
        # staged_predict_proba emits every prefix for step=1 and includes the
        # final partial period with the ordinary 50-tree grid.
        for position, probability in enumerate(bundle["model"].staged_predict_proba(
                values, eval_period=step), start=1):
            trees = min(position * step, count)
            curve.append(dict(iterations=trees, ap=float(
                average_precision_score(yinner, probability[:, 1]))))
    elif spec["family"] == "lightgbm":
        for trees in grid:
            probability = bundle["model"].booster_.predict(
                values, num_iteration=trees, num_threads=threads)
            curve.append(dict(iterations=trees, ap=float(
                average_precision_score(yinner, probability))))
    else:
        raise ValueError("Iteration selection supports only boosting families")
    if [row["iterations"] for row in curve] != grid:
        raise AssertionError("Tree-prefix grid differs from requested counts")
    best = max(curve, key=lambda row: (row["ap"], -row["iterations"]))
    return int(best["iterations"]), curve
