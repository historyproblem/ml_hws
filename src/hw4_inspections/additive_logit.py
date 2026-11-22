"""One fixed, fully regularized additive logistic HW4 hypothesis.

This is a local adaptation of source-grounded additive/logistic approaches, not
the original Chicago model with penalties applied only to inspector effects.
Every learned slope is L2-regularized here; no monotonicity constraint is used.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .classical_features import FORBIDDEN_COLUMNS, select_feature_columns
from .features import MISSING


CATEGORICAL_COLUMNS = ("facility_type", "risk_level", "area_id", "forecast_quarter")
OMITTED_COLUMNS = ("forecast_year", "cutoff_gap_days")
SPLINE_COLUMNS = ("history_violation_mean", "last_routine_violation_count",
                  "lag1_violation_count", "last3_violation_mean")
KNOT_QUANTILES = (0., .25, .5, .75, 1.)
CLASSIFIER_PARAMS = dict(C=.1, penalty="l2", solver="lbfgs", tol=1e-7,
                         max_iter=1500, fit_intercept=True, class_weight=None)


def _validate_frame(frame):
    if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
            or not all(isinstance(c, str) for c in frame.columns)):
        raise ValueError("Features must have unique string column names")
    if FORBIDDEN_COLUMNS.intersection(frame.columns):
        raise ValueError("Targets and raw identifiers are forbidden")


def _log_count(name):
    return name.endswith(("_n", "_violation_total", "_violation_count"))


class AdditiveLogitTransform:
    """Inspectable fit-only compact features, spline bases and OHE categories.

    Continuous and spline basis columns share one fitted StandardScaler. Raw
    numeric missing flags and categorical indicators remain binary/unscaled.
    Numeric bounds/medians and spline knots never inspect prediction rows.
    """

    @staticmethod
    def _raw_numeric(frame, name):
        value = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float, copy=True)
        value[~np.isfinite(value)] = np.nan
        return value

    @staticmethod
    def _categories(frame):
        result = pd.DataFrame(index=pd.RangeIndex(len(frame)))
        for name in CATEGORICAL_COLUMNS[:-1]:
            values = frame[name].astype(object).reset_index(drop=True)
            result[name] = values.where(values.notna(), MISSING).astype(str)
        quarter = pd.to_numeric(frame["forecast_quarter"], errors="raise").to_numpy(dtype=float)
        if not (np.isnan(quarter) | np.isin(quarter, [1, 2, 3, 4])).all():
            raise ValueError("forecast_quarter must be 1,2,3,4 or missing")
        result["forecast_quarter"] = [MISSING if np.isnan(v) else str(int(v)) for v in quarter]
        return result

    def _continuous(self, frame):
        columns = []
        missing = []
        for name in self.numeric_columns_:
            raw = self._raw_numeric(frame, name)
            missing.append(np.isnan(raw).astype(float))
            state = self.numeric_stats_[name]
            if state["mode"] == "constant_zero":
                columns.append(np.zeros((len(frame), 1)))
                continue
            value = np.clip(np.where(np.isnan(raw), state["median"], raw),
                            state["minimum"], state["maximum"])
            if state["mode"] == "spline":
                columns.append(self.spline_transformers_[name].transform(value[:, None]))
            else:
                if state["log1p"]:
                    value = np.log1p(value)
                columns.append(value[:, None])
        return np.concatenate(columns, axis=1), np.column_stack(missing)

    def fit(self, frame):
        _validate_frame(frame)
        if not len(frame):
            raise ValueError("An additive transform needs nonempty training rows")
        compact = select_feature_columns(frame, "compact_context")
        if len(compact) != 96:
            raise ValueError("The frozen additive view requires exactly compact96")
        self.input_columns_ = list(frame)
        self.selected_columns_ = [c for c in compact if c not in OMITTED_COLUMNS]
        self.categorical_columns_ = list(CATEGORICAL_COLUMNS)
        self.numeric_columns_ = [c for c in self.selected_columns_ if c not in CATEGORICAL_COLUMNS]
        self.numeric_stats_ = {}
        self.spline_knots_ = {}
        self.spline_transformers_ = {}
        continuous_names = []
        for name in self.numeric_columns_:
            raw = self._raw_numeric(frame, name)
            observed = raw[np.isfinite(raw)]
            all_missing = not len(observed)
            minimum = float(observed.min()) if not all_missing else 0.
            maximum = float(observed.max()) if not all_missing else 0.
            median = float(np.median(observed)) if not all_missing else 0.
            constant = all_missing or minimum == maximum
            state = dict(minimum=minimum, maximum=maximum, median=median,
                         all_missing=all_missing, constant=constant,
                         mode="constant_zero" if constant else "linear", log1p=False)
            if name in SPLINE_COLUMNS:
                knots = np.unique(np.quantile(observed, KNOT_QUANTILES)) if not all_missing else np.empty(0)
                self.spline_knots_[name] = knots
                if not constant and len(knots) >= 3:
                    value = np.clip(np.where(np.isnan(raw), median, raw), minimum, maximum)
                    spline = SplineTransformer(
                        degree=2, knots=knots[:, None], extrapolation="constant",
                        include_bias=False, sparse_output=False,
                    ).fit(value[:, None])
                    self.spline_transformers_[name] = spline
                    state["mode"] = "spline"
                    continuous_names.extend(f"{name}__spline{i}" for i in range(spline.n_features_out_))
                else:
                    if not constant:
                        state["mode"] = "linear_fallback"
                    continuous_names.append(name)
            else:
                if _log_count(name):
                    if minimum < 0:
                        raise ValueError(f"Nonnegative count feature has negative training values: {name}")
                    state["log1p"] = True
                continuous_names.append(name)
            self.numeric_stats_[name] = state
        self.continuous_feature_names_ = continuous_names
        self.missing_feature_names_ = [f"{name}__missing" for name in self.numeric_columns_]
        continuous, _ = self._continuous(frame)
        self.scaler_ = StandardScaler().fit(continuous)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                       dtype=np.float64).fit(self._categories(frame))
        category_names = [f"onehot::{name}" for name in self.encoder_.get_feature_names_out(CATEGORICAL_COLUMNS)]
        self.feature_names_ = np.asarray([*continuous_names, *self.missing_feature_names_, *category_names], dtype=object)
        if len(set(self.feature_names_)) != len(self.feature_names_):
            raise AssertionError("Additive output feature names must be unique")
        self.n_features_out_ = len(self.feature_names_)
        self.feature_budget_ = dict(
            compact_input=96, selected_input=len(self.selected_columns_),
            raw_numeric=len(self.numeric_columns_), raw_categorical=len(CATEGORICAL_COLUMNS),
            continuous_bases=len(continuous_names), missing_flags=len(self.numeric_columns_),
            category_indicators=len(category_names), total=self.n_features_out_,
        )
        return self

    def transform(self, frame):
        _validate_frame(frame)
        if not hasattr(self, "scaler_"):
            raise ValueError("AdditiveLogitTransform must be fitted first")
        if list(frame) != self.input_columns_:
            raise ValueError("Prediction feature schema/order differs from training")
        if not len(frame):
            return np.empty((0, self.n_features_out_), dtype=float)
        continuous, missing = self._continuous(frame)
        categories = self.encoder_.transform(self._categories(frame))
        value = np.concatenate([self.scaler_.transform(continuous), missing, categories], axis=1)
        if value.shape != (len(frame), self.n_features_out_) or not np.isfinite(value).all():
            raise ValueError("Additive design matrix must be row-aligned and finite")
        return value

    def get_feature_names_out(self):
        return self.feature_names_.copy()


def fit_additive(frame, y):
    """One fixed full-L2 logistic fit; no weights, target encoding or C search."""
    target = np.asarray(y)
    if (target.shape != (len(frame),) or not np.isfinite(target).all()
            or set(np.unique(target)) != {0, 1}):
        raise ValueError("Additive logistic fit needs both binary classes in row order")
    transform = AdditiveLogitTransform().fit(frame)
    design = transform.transform(frame)
    model = LogisticRegression(**CLASSIFIER_PARAMS)
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model.fit(design, target)
    if list(model.classes_) != [0, 1]:
        raise AssertionError("Unexpected additive target class order")
    recipe = dict(
        input_view="exact compact96 selected from causal base346",
        omitted_columns=list(OMITTED_COLUMNS), categorical=list(CATEGORICAL_COLUMNS),
        spline_fields=list(SPLINE_COLUMNS), spline_degree=2,
        spline_knot_quantiles=list(KNOT_QUANTILES), spline_tail="constant",
        spline_include_bias=False,
        insufficient_knots="<3 unique fit quantile knots: standardized linear; constant/all-missing: zero",
        numeric_imputation="finite fit median; all-missing zero",
        numeric_tails="clip every numeric field to observed fit minimum/maximum",
        log1p_rule="non-spline fields ending _n, _violation_total or _violation_count only",
        actual_log1p_fields=[name for name in transform.numeric_columns_ if transform.numeric_stats_[name]["log1p"]],
        continuous_scaling="one fit StandardScaler over linear/spline basis columns",
        indicator_scaling="all raw numeric missing flags and OHE categories are binary/unscaled",
        classifier=dict(CLASSIFIER_PARAMS), class_weights=None, sample_weights=None,
        penalty_scope="L2 on all slope coefficients; lbfgs intercept unpenalized",
        interactions=False, target_encoding=False, monotonicity_constraints=False,
        feature_budget=dict(transform.feature_budget_),
    )
    return dict(transformer=transform, model=model, recipe=recipe,
                feature_names=transform.get_feature_names_out().tolist(),
                iterations=int(model.n_iter_.max()), converged=True)


def predict_additive(bundle, frame):
    """Finite class-1 probabilities using saved fit-only transformations."""
    design = bundle["transformer"].transform(frame)
    if not len(frame):
        return np.empty(0, dtype=float)
    with threadpool_limits(limits=4):
        probability = bundle["model"].predict_proba(design)[:, 1]
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise FloatingPointError("Invalid additive logistic probabilities")
    return probability
