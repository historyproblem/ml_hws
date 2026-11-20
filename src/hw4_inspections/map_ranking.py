"""CPU MAP-oriented CatBoost with quarter groups and external sklearn AP.

The caller owns temporal train/inner/outer splits. Training never receives an
evaluation set; selecting a tree prefix uses only the supplied inner labels.
The YetiRank MAP approximation does not replace the competition's sklearn AP.
"""
from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


LOSS = "YetiRank:mode=MAP"
PARAMS = {"depth": 5, "learning_rate": .04, "l2_leaf_reg": 10.}
FORBIDDEN_COLUMNS = frozenset({
    "target", "row_id", "site_id", "forecast_period", "history_before",
    "inspection_week",
})


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _features(frame, cats, expected=None):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Features must be a DataFrame with unique columns")
    if any(not isinstance(c, str) for c in frame):
        raise ValueError("Feature names must be strings")
    if not len(frame.columns):
        raise ValueError("At least one feature is required")
    if expected is not None and list(frame) != list(expected):
        raise ValueError("Prediction feature names/order differ from fit")
    forbidden = FORBIDDEN_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise ValueError(f"Raw identifiers/labels/dates cannot be features: {sorted(forbidden)}")
    if len(set(cats)) != len(cats) or not set(cats).issubset(frame.columns):
        raise ValueError("Categorical names must be unique feature columns")
    if any(pd.api.types.is_datetime64_any_dtype(frame[c]) for c in frame):
        raise ValueError("Raw datetime features are not allowed")
    clean = frame.copy(deep=True)
    for name in cats:
        clean[name] = clean[name].fillna("<none>").astype(str)
    numeric = [c for c in frame if c not in cats]
    clean[numeric] = clean[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    return clean


def _labels(labels, expected_n, *, name):
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != expected_n or not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} must be aligned one-dimensional binary labels")
    if set(np.unique(values)) != {0, 1}:
        raise ValueError(f"{name} requires both binary target classes overall")
    return values.astype(np.float64, copy=True)


def _quarters(quarters, expected_n):
    if np.ndim(quarters) != 1 or len(quarters) != expected_n:
        raise ValueError("Quarter groups must be one-dimensional and row aligned")
    dates = pd.DatetimeIndex(pd.to_datetime(quarters, errors="raise"))
    if dates.isna().any() or dates.tz is not None:
        raise ValueError("Quarter groups must be nonmissing timezone-free dates")
    starts = dates.to_period("Q").to_timestamp()
    if not dates.equals(starts):
        raise ValueError("Every group date must be an exact quarter start")
    return dates


def fit_ranker(x, y, quarters, cats, *, seed=42, threads=4, iterations=500):
    """Fit one fixed MAP-oriented CPU ranker and return a joblib bundle.

    Quarter groups are sorted chronologically and made contiguous. A stable
    permutation preserves within-quarter order and aligns features and labels.
    Individual single-class groups are permitted when training overall has both
    classes. NaN remains native CatBoost missing input; IDs are never features.
    """
    from catboost import CatBoostRanker

    iterations = _positive_integer(iterations, "iterations")
    threads = _positive_integer(threads, "threads")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cats = list(cats)
    clean = _features(x, cats)
    labels = _labels(y, len(x), name="Training labels")
    dates = _quarters(quarters, len(x))
    order = np.argsort(dates.asi8, kind="stable")
    group_ids = dates.strftime("%Y-%m-%d").to_numpy()[order]
    model = CatBoostRanker(iterations=iterations, loss_function=LOSS,
                          random_seed=int(seed), thread_count=threads,
                          task_type="CPU", verbose=False,
                          allow_writing_files=False, **PARAMS)
    model.fit(clean.iloc[order], labels[order], group_id=group_ids,
              cat_features=cats)
    groups, counts = np.unique(group_ids, return_counts=True)
    return {"family": "catboost_map_ranker", "model": model,
            "features": list(x), "cats": cats, "preprocessor": None,
            "seed": int(seed), "iterations": iterations,
            "training_threads": threads, "inference_threads": threads,
            "training_quarters": groups.tolist(),
            "group_sizes": {str(q): int(n) for q, n in zip(groups, counts)},
            "loss_function": LOSS, "effective_params": model.get_all_params(),
            "selection_metric": "sklearn.average_precision_score on raw scores"}


def _predict_prefix(bundle, x, iterations=0):
    prepared = _features(x, bundle["cats"], bundle["features"])
    if not len(x):
        return np.empty(0, dtype=np.float64)
    # CatBoostRanker.predict is raw-score prediction; unlike the classifier,
    # its API has no prediction_type parameter or probability output.
    scores = np.asarray(bundle["model"].predict(
        prepared, ntree_end=iterations,
        thread_count=bundle["inference_threads"]), dtype=np.float64)
    if scores.shape != (len(x),) or not np.isfinite(scores).all():
        raise ValueError("Ranker predictions must be finite row-aligned raw scores")
    return scores


def predict_ranker(bundle, x):
    """Return finite raw scores in input order, without probability clipping."""
    return _predict_prefix(bundle, x)


def choose_ranker_iterations(xfit, yfit, qfit, xeval, yeval, cats,
                             max_iterations=1000, step=50, seed=42, threads=4):
    """Fit the maximum once; choose the earliest prefix at maximum inner AP.

    The prefix grid matches the existing classifier's default 50-tree grid and
    includes the maximum even when it is not divisible by ``step``. The supplied
    evaluation data must be an earlier inner fold, never the final outer fold.
    """
    max_iterations = _positive_integer(max_iterations, "max_iterations")
    step = _positive_integer(step, "step")
    cats = list(cats)
    _features(xeval, cats, expected=list(xfit))
    labels = _labels(yeval, len(xeval), name="Inner validation labels")
    bundle = fit_ranker(xfit, yfit, qfit, cats, seed=seed, threads=threads,
                        iterations=max_iterations)
    grid = sorted(set(range(step, max_iterations + 1, step)) | {max_iterations})
    curve = []
    for count in grid:
        scores = _predict_prefix(bundle, xeval, count)
        curve.append({"iterations": count,
                      "ap": float(average_precision_score(labels, scores))})
    best = max(curve, key=lambda row: (row["ap"], -row["iterations"]))
    return int(best["iterations"]), curve
