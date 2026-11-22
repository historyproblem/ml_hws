"""Controlled train-only bounded transforms for the pinned official TabM.

The original TabM helpers stay unchanged. The subclass supplies transformed
numeric values while inheriting training-only categories and official PLE bins.
Only a local, sequential fit temporarily selects this subclass in the trainer.
"""
from __future__ import annotations

from functools import partial
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import QuantileTransformer

from . import tabm_training
from .classical_features import select_feature_columns
from .tabm_models import TabMPreprocessor


CASES = {
    "tabm_clip_full": dict(mode="clip", view="full"),
    "tabm_quantile_full": dict(mode="quantile", view="full"),
    "tabm_quantile_compact": dict(mode="quantile", view="compact94"),
}


def feature_view(frame, cats, view):
    if view == "full":
        return frame.copy(), list(cats)
    if view != "compact94":
        raise ValueError("Unknown bounded TabM feature view")
    columns = [c for c in select_feature_columns(frame, "compact_context")
               if c not in {"forecast_year", "cutoff_gap_days"}]
    if len(columns) != 94:
        raise ValueError("Expected the fixed compact94 input")
    result = frame[columns].copy()
    result["forecast_quarter"] = result.forecast_quarter.astype(str)
    return result, ["facility_type", "risk_level", "area_id", "forecast_quarter"]


class BoundedTabMPreprocessor(TabMPreprocessor):
    def __init__(self, bins=16, mode="clip"):
        super().__init__(bins=bins)
        if mode not in {"clip", "quantile"}:
            raise ValueError("Expected clip or quantile")
        self.mode = mode
        self.numeric_low_ = self.numeric_high_ = self.quantile_ = None

    def fit(self, frame, cats):
        self.numeric_low_ = self.numeric_high_ = self.quantile_ = None
        return super().fit(frame, cats)

    def _base_transform(self, frame):
        numeric, categories = super()._base_transform(frame)
        if not len(frame):
            return numeric, categories
        count = len(self.numeric_names)
        if self.mode == "clip":
            if self.numeric_low_ is None:
                self.numeric_low_ = numeric[:, :count].min(axis=0)
                self.numeric_high_ = numeric[:, :count].max(axis=0)
            numeric[:, :count] = np.clip(numeric[:, :count], self.numeric_low_, self.numeric_high_)
            return numeric, categories
        raw = frame[self.numeric_names].to_numpy(dtype=float, copy=True)
        raw[~np.isfinite(raw)] = np.nan
        values = self.base.imputer.transform(raw)
        if self.quantile_ is None:
            self.constant_raw_ = (values == values[0]).all(axis=0)
            self.quantile_ = QuantileTransformer(
                n_quantiles=min(512, len(frame)), output_distribution="normal",
                subsample=None, random_state=42,
            ).fit(values)
        values = self.quantile_.transform(values)
        values[:, self.constant_raw_] = 0.
        numeric[:, :count] = values.astype(np.float32)
        return numeric, categories


def fit_bounded(xtrain, ytrain, cats, *, mode, **kwargs):
    factory = partial(BoundedTabMPreprocessor, mode=mode)
    with patch.object(tabm_training, "TabMPreprocessor", factory):
        return tabm_training.fit_tabm(xtrain, ytrain, cats, **kwargs)
