"""Training-only numeric bases for the HW4 residual tabular networks."""
from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

from .neural import TabularTransform


FORBIDDEN_COLUMNS = frozenset({
    "target", "row_id", "site_id", "forecast_period", "history_before",
})


class ResidualTransform:
    """Compose the existing tabular transform with per-field linear bases.

    Quantile boundaries are fitted only on finite, observed training values.
    Every numeric field receives at most ``bins`` cumulative linear segments,
    with zero padding for fewer distinct quantiles. Missing values, constants
    and entirely missing training fields have zero basis. The separate numeric
    branch retains the original standardized values and missing indicators.
    """

    def __init__(self, bins: int = 16):
        if isinstance(bins, bool) or not isinstance(bins, numbers.Integral) or not 1 <= bins <= 64:
            raise ValueError("bins must be an integer between 1 and 64")
        self.bins = int(bins)

    @staticmethod
    def _validate_frame(frame):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("Expected a feature DataFrame")
        if not frame.columns.is_unique or any(not isinstance(c, str) for c in frame.columns):
            raise ValueError("Feature names must be unique strings")
        forbidden = FORBIDDEN_COLUMNS.intersection(frame.columns)
        if forbidden:
            raise ValueError(f"Raw identifiers/labels are not features: {sorted(forbidden)}")

    def fit(self, frame: pd.DataFrame, categorical) -> "ResidualTransform":
        self._validate_frame(frame)
        categories = list(categorical)
        if len(set(categories)) != len(categories) or set(categories).difference(frame.columns):
            raise ValueError("Categorical names must be unique feature columns")
        numeric_names = [c for c in frame if c not in categories]
        if not len(frame) or not numeric_names:
            raise ValueError("At least one training row and numeric field are required")
        # TabularTransform replaces nonfinite numeric values before imputation.
        # A private copy prevents its NumPy view from modifying caller data.
        base = TabularTransform().fit(frame.copy(deep=True), categories)
        knots = []
        quantiles = np.linspace(0.0, 1.0, self.bins + 1)
        for name in numeric_names:
            values = frame[name].to_numpy(dtype=np.float64, copy=True)
            observed = values[np.isfinite(values)]
            if observed.size:
                boundaries = np.unique(np.quantile(observed, quantiles))
            else:
                boundaries = np.empty(0, dtype=np.float64)
            knots.append(boundaries.astype(np.float64, copy=False))
        self.base = base
        self.columns = list(frame.columns)
        self.categorical = categories
        self.numeric_names = numeric_names
        self.numeric = list(numeric_names)
        self.cardinalities = list(base.cardinalities)
        self.knots = knots
        return self

    def transform(self, frame: pd.DataFrame):
        self._validate_frame(frame)
        if not hasattr(self, "base"):
            raise ValueError("ResidualTransform must be fitted before transform")
        if list(frame.columns) != self.columns:
            raise ValueError("Feature schema mismatch")
        n, n_numeric = len(frame), len(self.numeric_names)
        basis = np.zeros((n, n_numeric, self.bins), dtype=np.float32)
        if not n:
            return (np.empty((0, 2 * n_numeric), dtype=np.float32),
                    np.empty((0, len(self.categorical)), dtype=np.int64), basis)
        if self.categorical:
            numeric, categories = self.base.transform(frame.copy(deep=True))
        else:
            # The existing transformer uses column_stack on category columns;
            # preserve its exact numeric path for a numeric-only feature view.
            values = frame[self.numeric_names].to_numpy(dtype=np.float64, copy=True)
            values[~np.isfinite(values)] = np.nan
            missing = np.isnan(values)
            scaled = np.clip(self.base.scaler.transform(self.base.imputer.transform(values)), -8, 8)
            numeric = np.concatenate([scaled, missing], axis=1).astype(np.float32)
            categories = np.empty((n, 0), dtype=np.int64)
        for j, (name, boundaries) in enumerate(zip(self.numeric_names, self.knots)):
            if len(boundaries) < 2:
                continue
            values = frame[name].to_numpy(dtype=np.float64, copy=True)
            observed = np.isfinite(values)
            if not observed.any():
                continue
            observed_values = np.clip(values[observed], boundaries[0], boundaries[-1])
            segments = np.clip((observed_values[:, None] - boundaries[:-1]) /
                               np.diff(boundaries), 0.0, 1.0)
            basis[observed, j, :len(boundaries) - 1] = segments.astype(np.float32)
        return numeric, categories, basis
