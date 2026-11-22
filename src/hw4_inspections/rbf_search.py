"""Eight bounded RBF-SVC hypotheses with strictly fit-only preprocessing.

Native probability=True performs the explicitly permitted internal Platt CV on
the caller's past FIT pool. No external evaluation frame/labels are accepted.
The product is class1 probability only, never a retrospective margin/rank choice.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .classical_features import COMPACT_BASE_COLUMNS
from .ebm_search import _check_frame, _categorical_values, select_columns as compact_columns


SIGNED = "violation_change_last_vs_mean"
OMITTED = frozenset({"forecast_year", "cutoff_gap_days"})
RBF_RECIPES = {}
for _view in ("own22", "compact94"):
    for _c in (.5, 5.):
        for _multiplier in (.25, 1.):
            _name = f"rbf_{_view}_C{'0p5' if _c == .5 else '5'}_g{'0p25' if _multiplier == .25 else '1'}"
            RBF_RECIPES[_name] = dict(id=_name, view=_view, C=_c, gamma_multiplier=_multiplier)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def select_columns(frame, view):
    """Select exact own22 or compact94, always omitting raw year/cutoff gap."""
    _check_frame(frame)
    if view == "compact94":
        return compact_columns(frame, "compact94")
    if view != "own22":
        raise ValueError("Unknown RBF feature view")
    required = set(COMPACT_BASE_COLUMNS) - OMITTED
    if len(frame.columns) == 22:
        if set(frame.columns) != required:
            raise ValueError("The own22 feature schema differs")
        return list(frame.columns)
    columns = [name for name in compact_columns(frame, "compact94") if not name.startswith("ctx_")]
    if len(columns) != 22 or set(columns) != required:
        raise ValueError("The RBF own view requires the exact22 static/calendar/burden fields")
    return columns


class RBFTransform:
    """Fit-only log values, medians, robust scaling, raw masks and dense OHE."""

    def _categorical(self, frame):
        result = pd.DataFrame(index=pd.RangeIndex(len(frame)))
        for name in self.categorical:
            result[name] = _categorical_values(frame, name).fillna("<none>").astype(str)
        return result

    def _numeric(self, frame):
        values = frame[self.numeric].to_numpy(dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        missing = np.isnan(values)
        for j, name in enumerate(self.numeric):
            if name == SIGNED:
                values[:, j] = np.sign(values[:, j]) * np.log1p(np.abs(values[:, j]))
            elif name in self.log_count_names:
                if np.any(values[:, j] < 0):
                    raise ValueError(f"Declared nonnegative count/load field is negative: {name}")
                values[:, j] = np.log1p(values[:, j])
        return values, missing

    def fit(self, frame, cats, view):
        self.features = select_columns(frame, view)
        self.input_features = list(frame)
        self.view = view
        cats = list(cats)
        if len(cats) != len(set(cats)) or not set(cats).issubset(frame.columns):
            raise ValueError("Category names must be unique input fields")
        self.categorical = [name for name in self.features if name in cats or name == "forecast_quarter"]
        self.numeric = [name for name in self.features if name not in self.categorical]
        if not self.numeric or not len(frame):
            raise ValueError("RBF preprocessing requires fit rows and numeric fields")
        self.log_count_names = [name for name in self.numeric if name != SIGNED
                               and not (name.startswith("ctx_") and name.endswith("fail_smooth"))]
        values, _ = self._numeric(frame)
        self.medians = np.array([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.
                                for column in values.T], dtype=np.float64)
        self.all_missing_numeric = [name for j, name in enumerate(self.numeric) if np.isnan(values[:, j]).all()]
        filled = np.where(np.isnan(values), self.medians, values)
        self.scaler = RobustScaler(quantile_range=(25., 75.), with_centering=True,
                                   with_scaling=True, unit_variance=False).fit(filled)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                      dtype=np.float64).fit(self._categorical(frame))
        self.feature_names_out = ([f"scaled::{name}" for name in self.numeric]
                                  + [f"missing::{name}" for name in self.numeric]
                                  + [f"onehot::{name}" for name in self.encoder.get_feature_names_out(self.categorical)])
        self.n_features_out = len(self.feature_names_out)
        return self

    def transform(self, frame):
        _check_frame(frame)
        if list(frame) not in [self.input_features, self.features]:
            raise ValueError("Prediction feature schema/order differs from fit")
        if not len(frame):
            return np.empty((0, self.n_features_out), dtype=np.float64)
        values, missing = self._numeric(frame)
        filled = np.where(np.isnan(values), self.medians, values)
        transformed = np.ascontiguousarray(np.concatenate([
            self.scaler.transform(filled), missing.astype(np.float64),
            self.encoder.transform(self._categorical(frame))], axis=1), dtype=np.float64)
        if transformed.shape != (len(frame), self.n_features_out) or not np.isfinite(transformed).all():
            raise ValueError("RBF design must be a finite row-aligned float64 matrix")
        return transformed


def fit_rbf(recipe, x, y, cats, *, seed=42, threads=2):
    """Fit one fully converged probability SVC; partial models are invalid."""
    seed = _integer(seed, "seed", minimum=0)
    threads = _integer(threads, "threads")
    if threads > 2:
        raise ValueError("RBF has a hard ceiling of two CPU threads")
    c = _positive(recipe["C"], "C")
    multiplier = _positive(recipe["gamma_multiplier"], "gamma_multiplier")
    labels = np.asarray(y)
    if labels.shape != (len(x),) or not np.isin(labels, [0, 1]).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("RBF fit requires aligned binary targets and both classes")
    started = time.monotonic()
    transform = RBFTransform().fit(x, cats, recipe["view"])
    design = transform.transform(x)
    variance = float(design.var(ddof=0))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("The full fit design must have positive finite variance")
    gamma_base = 1. / (design.shape[1] * variance)
    gamma = gamma_base * multiplier
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Computed RBF gamma must be finite and positive")
    params = dict(kernel="rbf", C=c, gamma=gamma, probability=True, random_state=seed,
                  cache_size=512, class_weight=None, shrinking=True, tol=1e-3, max_iter=100000)
    model = SVC(**params)
    try:
        with threadpool_limits(limits=threads), warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            model.fit(design, labels.astype(np.int64))
    except ConvergenceWarning as error:
        raise ValueError("SVC did not converge; no partial model is valid") from error
    if model.fit_status_ != 0 or list(model.classes_) != [0, 1]:
        raise ValueError("SVC did not converge or has an unexpected class order")
    if not np.isfinite(model.probA_).all() or not np.isfinite(model.probB_).all():
        raise ValueError("Native Platt calibration parameters must be finite")
    return dict(recipe=deepcopy(recipe), model=model, transformer=transform, seed=seed, threads=threads,
                library_version=sklearn.__version__, effective_params=model.get_params(deep=False),
                gamma_base=gamma_base, gamma=gamma, fit_design_variance=variance,
                design_features=design.shape[1], selected_features=transform.features,
                n_support=model.n_support_.tolist(), n_iter=model.n_iter_.tolist(), fit_status=int(model.fit_status_),
                fit_seconds=time.monotonic() - started, trained_rows=len(x), outer_labels_seen=False,
                C_normalization="constantC, not scaled with fit sample count",
                missing_policy="Raw nonfinite masks, fit-logspace medians with allmissing0, no clipping; positivecount/loadlog1p and only signedlog1p for violation_change_last_vs_mean",
                gamma_policy="multiplier/(D*Var(full transformed FIT matrix)), no evaluation rows",
                probability_policy="Native5fold Platt CV ONLY inside caller-supplied PAST FITpool; productclass1P, no retrospective margin/rank choice",
                concurrency="One native LIBSVM solver is approximately oneCPU; threadpool ceiling applies to preprocessing/BLAS, no n_jobs or nested backend fits")


def predict_rbf(bundle, x):
    """Native class1 probability only; never choose margins after seeing AP."""
    design = bundle["transformer"].transform(x)
    if not len(x):
        return np.empty(0, dtype=np.float64)
    with threadpool_limits(limits=bundle["threads"]):
        probability = np.asarray(bundle["model"].predict_proba(design)[:, 1], dtype=np.float64)
    if probability.shape != (len(x),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise FloatingPointError("RBF probabilities must be finite, row-aligned and in[0,1]")
    return probability
