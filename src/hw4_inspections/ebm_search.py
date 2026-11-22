"""Eight fixed-round native EBM recipes, with isolated runtime and fit-only X.

There is no inner random validation, automatic interaction selection, warm
start, or round-prefix API. The caller evaluates separate fixed-round fits on
earlier inner quarters and never passes those rows into this function's fit.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import metadata
from numbers import Integral
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .classical_features import COMPACT_BASE_COLUMNS


VERSION = "0.7.8"
OMITTED = frozenset({"forecast_year", "cutoff_gap_days"})
FORBIDDEN = frozenset({"target", "row_id", "site_id", "forecast_period", "history_before", "inspection_week"})
FIXED_PAIRS = (
    ("history_violation_mean", "risk_level"),
    ("last_routine_violation_count", "risk_level"),
    ("last3_violation_mean", "risk_level"),
    ("lag1_violation_count", "risk_level"),
    ("history_violation_mean", "facility_type"),
    ("last_routine_violation_count", "facility_type"),
    ("history_violation_mean", "area_id"),
    ("history_violation_mean", "forecast_quarter"),
    ("last_routine_violation_count", "forecast_quarter"),
    ("history_violation_mean", "ctx_area_365d_routine_load_smooth"),
)
COMMON = dict(
    objective="log_loss", validation_size=0, early_stopping_rounds=0,
    outer_bags=1, inner_bags=0, n_jobs=2, random_state=42,
    learning_rate=.03, greedy_ratio=0, cyclic_progress=False,
    smoothing_rounds=75, interaction_smoothing_rounds=75,
    max_bins=128, max_interaction_bins=16, min_samples_leaf=20,
    reg_lambda=1., max_leaves=2, missing="separate",
)


def _recipes():
    recipes = {}
    for view in ("compact94", "no_codes168"):
        for terms in ("main", "pairs10"):
            for rounds in (300, 800):
                name = f"ebm_{view}_{terms}_{rounds}"
                params = dict(COMMON, max_rounds=rounds,
                              interactions=0 if terms == "main" else list(FIXED_PAIRS))
                recipes[name] = dict(id=name, view=view, params=params)
    return recipes


EBM_RECIPES = _recipes()


def load_runtime(target):
    """Import only the pinned interpret-core package from the isolated target.

    Call before joblib-loading its model objects in a fresh Python process.
    The package and selected native library must both resolve below target.
    """
    target = Path(target).resolve()
    if not target.is_dir() or not (target / "interpret").is_dir():
        raise ValueError("An installed isolated interpret-core target is required")
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    import interpret
    from interpret.glassbox import ExplainableBoostingClassifier
    from interpret.utils._native import Native

    if metadata.version("interpret-core") != VERSION or getattr(interpret, "__version__", None) != VERSION:
        raise ValueError(f"This experiment requires interpret-core=={VERSION}")
    paths = [Path(interpret.__file__).resolve(),
             Path(sys.modules[ExplainableBoostingClassifier.__module__].__file__).resolve(),
             Path(Native._get_ebm_lib_path()).resolve()]
    if not all(path.is_relative_to(target) for path in paths):
        raise ValueError("InterpretML package or native binary came from outside the isolated target")
    return target


def _check_frame(frame):
    if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
            or not all(isinstance(c, str) for c in frame) or not len(frame.columns)):
        raise ValueError("Features must have nonempty unique string column names")
    if FORBIDDEN.intersection(frame.columns):
        raise ValueError("Current labels and raw row/site/date identifiers are forbidden")
    if any(pd.api.types.is_datetime64_any_dtype(frame[c]) for c in frame):
        raise ValueError("Raw datetime feature values are forbidden")


def select_columns(frame, view):
    """Both exact native views omit year and cutoff gap; quarter is retained."""
    _check_frame(frame)
    if view == "compact94":
        required = set(COMPACT_BASE_COLUMNS) - OMITTED
        if not required.issubset(frame.columns):
            raise ValueError("Required compact feature names are missing")
        columns = [c for c in frame if c not in OMITTED and (c in required or c.startswith("ctx_"))]
        expected = 94
    elif view == "no_codes168":
        columns = [c for c in frame if c not in OMITTED and not c.startswith("code")]
        expected = 168
    else:
        raise ValueError("Unknown EBM feature view")
    if len(columns) != expected or "forecast_quarter" not in columns:
        raise ValueError(f"The frozen EBM view requires exactly{expected} selected features")
    return columns


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _categorical_values(frame, name):
    if name == "forecast_quarter":
        numbers = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
        if not (np.isnan(numbers) | np.isin(numbers, [1, 2, 3, 4])).all():
            raise ValueError("forecast_quarter must be1/2/3/4 or missing")
        return pd.Series([np.nan if np.isnan(v) else str(int(v)) for v in numbers], dtype=object)
    values = frame[name].astype("string").reset_index(drop=True)
    return values.astype(object).where(values.notna(), np.nan)


def _transform(bundle, frame):
    _check_frame(frame)
    if list(frame) not in [bundle["input_features"], bundle["features"]]:
        raise ValueError("Prediction feature schema/order differs from fit")
    prepared = frame[bundle["features"]].reset_index(drop=True).copy(deep=True)
    for name in bundle["features"]:
        if name in bundle["categorical"]:
            values = _categorical_values(prepared, name)
            prepared[name] = values.where(values.isin(bundle["vocabularies"][name]), np.nan)
        else:
            prepared[name] = pd.to_numeric(prepared[name], errors="raise").astype(float).replace([np.inf, -np.inf], np.nan)
    return prepared


def fit_ebm(recipe, x, y, cats, *, runtime, seed=42, threads=2):
    """Fit one fixed-round EBM using only this frame's numeric/category values."""
    seed = _integer(seed, "seed", minimum=0)
    threads = _integer(threads, "threads")
    if threads > 2:
        raise ValueError("EBM has a hard maximum of two CPU jobs")
    target = load_runtime(runtime)
    from interpret.glassbox import ExplainableBoostingClassifier
    from interpret.utils._native import Native

    columns = select_columns(x, recipe["view"])
    cats = list(cats)
    if len(cats) != len(set(cats)) or not set(cats).issubset(x.columns):
        raise ValueError("Categories must be unique input column names")
    categorical = [name for name in columns if name in cats or name == "forecast_quarter"]
    labels = np.asarray(y)
    if labels.shape != (len(x),) or not np.isin(labels, [0, 1]).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("EBM fit needs aligned binary targets and both classes")
    bundle = dict(recipe=deepcopy(recipe), view=recipe["view"], input_features=list(x),
                  features=columns, categorical=categorical,
                  vocabularies={name: sorted(_categorical_values(x, name).dropna().unique().tolist())
                                for name in categorical},
                  runtime_target=str(target), library_version=VERSION, seed=seed, threads=threads)
    prepared = _transform(bundle, x)
    params = deepcopy(recipe["params"])
    params["max_rounds"] = _integer(params["max_rounds"], "max_rounds")
    if params["max_rounds"] > 800:
        raise ValueError("EBM round count exceeds the frozen800-round cap")
    if (params.get("validation_size") != 0 or params.get("early_stopping_rounds") != 0
            or params.get("outer_bags") != 1 or params.get("inner_bags") != 0):
        raise ValueError("Random validation/early stopping and internal bagging are disabled")
    named_pairs = params["interactions"]
    if named_pairs == 0:
        pairs = []
        params["interactions"] = 0
    else:
        if (not isinstance(named_pairs, (list, tuple))
                or [tuple(pair) for pair in named_pairs] != list(FIXED_PAIRS)):
            raise ValueError("Only the exact ten predeclared interaction pairs are allowed")
        if any(a not in columns or b not in columns for a, b in FIXED_PAIRS):
            raise ValueError("An interaction field is missing from the selected view")
        pairs = [(columns.index(a), columns.index(b)) for a, b in FIXED_PAIRS]
        params["interactions"] = pairs
    types = ["nominal" if name in categorical else "continuous" for name in columns]
    params.update(feature_names=columns, feature_types=types, random_state=seed, n_jobs=threads)
    model = ExplainableBoostingClassifier(**params)
    with threadpool_limits(limits=threads), joblib.parallel_config(backend="loky", inner_max_num_threads=1), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module=r"interpret\.glassbox\._ebm\._ebm",
            message=r"Missing values detected\. Our visualizations do not currently display missing values\..*",
        )
        model.fit(prepared, labels.astype(np.int64))
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected EBM binary target class order")
    expected = {(i,) for i in range(len(columns))} | {tuple(sorted(pair)) for pair in pairs}
    actual = [tuple(term) for term in model.term_features_]
    if len(actual) != len(expected) or {tuple(sorted(term)) for term in actual} != expected:
        raise ValueError("EBM terms differ from main effects and the exact fixed interactions")
    native = Path(Native._get_ebm_lib_path())
    with native.open("rb") as stream:
        native_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    bundle.update(model=model, effective_params=model.get_params(deep=False),
                  native_library=str(native), native_library_sha256=native_sha,
                  feature_types=types, raw_known_feature_types=dict(zip(columns, types)),
                  terms=[list(term) for term in actual], interaction_names=[] if not pairs else list(FIXED_PAIRS),
                  interaction_indices=pairs, max_rounds=params["max_rounds"],
                  stage_update_counts=np.asarray(model.best_iteration_, dtype=np.int64).tolist(),
                  update_count_meaning="best_iteration_[stage,bag] records boosting update steps, not a tree/round prefix API",
                  cyclic_note="With greedy_ratio0, InterpretML internally forces cyclic progress even though cyclic_progress=False is the public input",
                  trained_rows=len(x), outer_labels_seen=False)
    return bundle


def predict_ebm(bundle, x):
    """Native class1 probabilities; no ranks, random calibration or clipping."""
    load_runtime(bundle["runtime_target"])
    prepared = _transform(bundle, x)
    if not len(x):
        return np.empty(0, dtype=np.float64)
    with threadpool_limits(limits=bundle["threads"]):
        probability = np.asarray(bundle["model"].predict_proba(prepared)[:, 1], dtype=np.float64)
    if probability.shape != (len(x),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise FloatingPointError("EBM predictions must be row-aligned finite probabilities")
    return probability
