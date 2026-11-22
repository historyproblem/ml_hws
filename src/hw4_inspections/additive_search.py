"""Frozen 24-recipe additive search, selected only on earlier inner quarters."""
from __future__ import annotations

from collections import Counter
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .additive_logit import (
    AdditiveLogitTransform, CATEGORICAL_COLUMNS, KNOT_QUANTILES,
    OMITTED_COLUMNS, SPLINE_COLUMNS, _validate_frame,
)
from .classical_features import recency_weights, select_feature_columns
from .features import MISSING


VIEWS = ("core", "compact_all_splines", "own_history", "full_context",
         "core_interactions", "core_recency4")
COMPLEXITY = dict(core=0, core_recency4=1, core_interactions=2,
                  own_history=3, full_context=4, compact_all_splines=5)
CS = (.01, .1, 1., 10.)
RECIPES = tuple(dict(id=f"{view}_C{c:g}", view=view, C=c,
                     complexity_order=COMPLEXITY[view]) for view in VIEWS for c in CS)
CLASSIFIER = dict(penalty="l2", solver="lbfgs", tol=1e-7, max_iter=5000,
                  fit_intercept=True, class_weight=None)
FULL_CATEGORIES = (*CATEGORICAL_COLUMNS[:3], "forecast_quarter",
                   "lag1_result", "lag1_inspection_type", "lag2_result",
                   "lag2_inspection_type", "lag3_result", "lag3_inspection_type",
                   "last_routine_result")


def recipe_by_id(name):
    for recipe in RECIPES:
        if recipe["id"] == name:
            return dict(recipe)
    raise ValueError(f"Unknown frozen additive recipe: {name}")


def _count_or_age(name):
    return name.endswith(("_n", "_count", "_total", "_n_visits", "_days"))


def _categorical(frame, names):
    values = pd.DataFrame(index=pd.RangeIndex(len(frame)))
    for name in names:
        column = frame[name].astype(object).reset_index(drop=True)
        if name == "forecast_quarter":
            quarter = pd.to_numeric(column, errors="raise").to_numpy(float)
            if not (np.isnan(quarter) | np.isin(quarter, [1, 2, 3, 4])).all():
                raise ValueError("Invalid forecast quarter")
            values[name] = [MISSING if np.isnan(v) else str(int(v)) for v in quarter]
        else:
            values[name] = column.where(column.notna(), MISSING).astype(str)
    return values


def _interaction_categories(frame):
    cats = _categorical(frame, CATEGORICAL_COLUMNS)
    return pd.DataFrame({
        "risk_area": [json.dumps([r, a], separators=(",", ":"), ensure_ascii=False)
                      for r, a in zip(cats.risk_level, cats.area_id)],
        "risk_quarter": [json.dumps([r, q], separators=(",", ":"), ensure_ascii=False)
                         for r, q in zip(cats.risk_level, cats.forecast_quarter)],
    })


class GeneralizedTransform:
    """Fit-only bounded additive bases for the three wider feature views."""

    def __init__(self, view):
        if view not in {"compact_all_splines", "own_history", "full_context"}:
            raise ValueError("Not a generalized frozen view")
        self.view = view

    def _continuous(self, frame):
        output, missing = [], []
        for name in self.numeric_columns_:
            raw = AdditiveLogitTransform._raw_numeric(frame, name)
            missing.append(np.isnan(raw).astype(float))
            state = self.numeric_stats_[name]
            if state["constant"]:
                output.append(np.zeros((len(frame), 1)))
                continue
            value = np.clip(np.where(np.isnan(raw), state["median"], raw),
                            state["minimum"], state["maximum"])
            if state["log1p"]:
                value = np.log1p(value)
            output.append(self.splines_[name].transform(value[:, None])
                          if name in self.splines_ else value[:, None])
        return np.concatenate(output, axis=1), np.column_stack(missing)

    def fit(self, frame):
        _validate_frame(frame)
        if not len(frame):
            raise ValueError("Empty fit")
        self.input_columns_ = list(frame)
        if self.view == "compact_all_splines":
            selected = select_feature_columns(frame, "compact_context")
            if len(selected) != 96:
                raise ValueError("Expected compact96")
        elif self.view == "own_history":
            selected = [c for c in frame if not c.startswith("ctx_")]
            if len(selected) != 274:
                raise ValueError("Expected own-history274")
        else:
            selected = list(frame)
            if len(selected) != 346:
                raise ValueError("Expected full346")
        self.selected_columns_ = [c for c in selected if c not in OMITTED_COLUMNS]
        self.categorical_columns_ = [c for c in self.selected_columns_ if c in FULL_CATEGORIES]
        self.numeric_columns_ = [c for c in self.selected_columns_ if c not in self.categorical_columns_]
        self.numeric_stats_, self.knots_, self.splines_ = {}, {}, {}
        names = []
        for name in self.numeric_columns_:
            raw = AdditiveLogitTransform._raw_numeric(frame, name)
            observed = raw[np.isfinite(raw)]
            state = dict(minimum=float(observed.min()) if len(observed) else 0.,
                         maximum=float(observed.max()) if len(observed) else 0.,
                         median=float(np.median(observed)) if len(observed) else 0.)
            state["constant"] = not len(observed) or state["minimum"] == state["maximum"]
            state["log1p"] = _count_or_age(name) and (
                self.view == "compact_all_splines" or name not in SPLINE_COLUMNS)
            if state["log1p"] and state["minimum"] < 0:
                raise ValueError(f"Negative fit values in nonnegative count/age field {name}")
            to_spline = self.view == "compact_all_splines" or name in SPLINE_COLUMNS
            if to_spline and not state["constant"]:
                transformed = np.log1p(observed) if state["log1p"] else observed
                knots = np.unique(np.quantile(transformed, KNOT_QUANTILES))
                self.knots_[name] = knots
                if len(knots) >= 3:
                    value = np.clip(np.where(np.isnan(raw), state["median"], raw),
                                    state["minimum"], state["maximum"])
                    if state["log1p"]:
                        value = np.log1p(value)
                    self.splines_[name] = SplineTransformer(
                        knots=knots[:, None], degree=2, extrapolation="constant",
                        include_bias=False).fit(value[:, None])
                    names.extend(f"{name}__spline{i}" for i in range(self.splines_[name].n_features_out_))
                else:
                    names.append(name)
            else:
                names.append(name)
            self.numeric_stats_[name] = state
        self.scaler_ = StandardScaler().fit(self._continuous(frame)[0])
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
            _categorical(frame, self.categorical_columns_))
        self.feature_names_ = [*names, *[f"{c}__missing" for c in self.numeric_columns_],
                               *[f"onehot::{c}" for c in self.encoder_.get_feature_names_out(self.categorical_columns_)]]
        return self

    def transform(self, frame):
        _validate_frame(frame)
        if list(frame) != self.input_columns_:
            raise ValueError("Feature schema/order differs")
        if not len(frame):
            return np.empty((0, len(self.feature_names_)))
        continuous, missing = self._continuous(frame)
        result = np.concatenate([self.scaler_.transform(continuous), missing,
                                 self.encoder_.transform(_categorical(frame, self.categorical_columns_))], axis=1)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite design matrix")
        return result


class SearchTransform:
    """Original stable core or a frozen representation extension."""

    def __init__(self, view):
        if view not in VIEWS:
            raise ValueError(view)
        self.view = view

    def fit(self, frame):
        self.base_ = (AdditiveLogitTransform() if self.view.startswith("core")
                      else GeneralizedTransform(self.view)).fit(frame)
        self.feature_names_ = list(self.base_.feature_names_)
        self.interactions_ = None
        if self.view == "core_interactions":
            self.interactions_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
                _interaction_categories(frame))
            self.feature_names_.extend(f"interaction::{c}" for c in self.interactions_.get_feature_names_out())
        return self

    def transform(self, frame):
        value = self.base_.transform(frame)
        if self.interactions_ is not None and len(frame):
            value = np.concatenate([value, self.interactions_.transform(_interaction_categories(frame))], axis=1)
        elif self.interactions_ is not None:
            value = np.empty((0, len(self.feature_names_)))
        return value


def fit_recipe(frame, y, recipe, *, periods=None, reference=None):
    recipe = recipe_by_id(recipe["id"])
    target = np.asarray(y)
    if target.shape != (len(frame),) or not np.isfinite(target).all() or set(np.unique(target)) != {0, 1}:
        raise ValueError("Binary aligned fit targets required")
    transform = SearchTransform(recipe["view"]).fit(frame)
    weights = None
    if recipe["view"] == "core_recency4":
        if periods is None or reference is None or len(periods) != len(frame):
            raise ValueError("Recency recipe requires aligned dates and a future reference quarter")
        weights = recency_weights(periods, reference, half_life_quarters=4)
    model = LogisticRegression(C=recipe["C"], **CLASSIFIER)
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model.fit(transform.transform(frame), target, sample_weight=weights)
    return dict(transformer=transform, model=model, recipe=recipe,
                classifier={"C": recipe["C"], **CLASSIFIER}, iterations=int(model.n_iter_.max()),
                feature_names=transform.feature_names_, threads=2,
                sample_weights="mean-one quarter half-life4" if weights is not None else None)


def predict_recipe(bundle, frame):
    values = bundle["transformer"].transform(frame)
    if not len(frame):
        return np.empty(0, dtype=np.float64)
    with threadpool_limits(limits=2):
        probability = bundle["model"].predict_proba(values)[:, 1]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise FloatingPointError("Invalid probabilities")
    return probability


def choose_inner(records):
    valid = [r for r in records if r["status"] == "complete" and np.isfinite(r["ap"])]
    if not valid:
        raise RuntimeError("Every recipe is numerically invalid in this inner quarter; no quarter may be dropped")
    return min(valid, key=lambda r: (-r["ap"], recipe_by_id(r["recipe"])["C"],
                                     recipe_by_id(r["recipe"])["complexity_order"], r["recipe"]))["recipe"]


def choose_final(winners, inner_records):
    votes = Counter(winners)
    by_recipe = {}
    for name in [r["id"] for r in RECIPES]:
        rows = [r for r in inner_records if r["recipe"] == name]
        if len(rows) == 3 and all(r["status"] == "complete" for r in rows):
            by_recipe[name] = float(np.mean([r["ap"] for r in rows]))
    if not by_recipe:
        raise RuntimeError("No recipe converged on all three inner quarters")
    return min(by_recipe, key=lambda name: (-votes[name], -by_recipe[name],
                    recipe_by_id(name)["C"], recipe_by_id(name)["complexity_order"], name))
