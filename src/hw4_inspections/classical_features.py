"""Fixed feature views and temporal weights for the HW4 classical screen.

These helpers never estimate statistics or inspect labels. They operate on the
already causal snapshot features built by ``features`` and ``extensions``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Exactly the static/calendar/burden selection used by the measured 24-column
# ablation. Keep this explicit so unrelated future "violation" columns cannot
# silently change that experiment.
COMPACT_BASE_COLUMNS = (
    "facility_type", "risk_level", "area_id",
    "forecast_year", "forecast_quarter", "cutoff_gap_days",
    "history_violation_mean", "history_violation_max", "history_violation_total",
    "recent90_violation_mean", "recent90_violation_max", "recent90_violation_total",
    "recent180_violation_mean", "recent180_violation_max", "recent180_violation_total",
    "recent365_violation_mean", "recent365_violation_max", "recent365_violation_total",
    "lag1_violation_count", "lag2_violation_count", "lag3_violation_count",
    "violation_change_last_vs_mean", "last3_violation_mean",
    "last_routine_violation_count",
)
FORBIDDEN_COLUMNS = frozenset({
    "target", "row_id", "site_id", "forecast_period", "history_before",
})
RELATIVE_PAIRS = (
    ("history", "history_violation_mean", "area", "ctx_area_365d_routine_load_smooth"),
    ("history", "history_violation_mean", "area_risk", "ctx_area_risk_365d_routine_load_smooth"),
    ("last_routine", "last_routine_violation_count", "area", "ctx_area_365d_routine_load_smooth"),
    ("last_routine", "last_routine_violation_count", "area_risk", "ctx_area_risk_365d_routine_load_smooth"),
)


def select_feature_columns(frame: pd.DataFrame, variant: str) -> list[str]:
    """Return a fixed feature view in the supplied frame's column order.

    ``context`` keeps all supplied feature columns. ``compact_context`` uses
    the exact prior 24-column ablation plus all ``ctx_`` columns (96 with the
    current 72-column context builder). Raw identifiers/labels fail explicitly;
    an intentional site-ID ablation must add its ID after this selection.
    """
    if variant not in {"context", "compact_context"}:
        raise ValueError(f"Unknown classical feature variant: {variant}")
    if not frame.columns.is_unique:
        raise ValueError("Feature columns must be unique")
    if any(not isinstance(c, str) for c in frame.columns):
        raise ValueError("Feature names must be strings")
    forbidden = FORBIDDEN_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise ValueError(f"Raw identifiers/labels are not features: {sorted(forbidden)}")
    if variant == "context":
        return list(frame.columns)
    missing = set(COMPACT_BASE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing compact feature columns: {sorted(missing)}")
    compact = set(COMPACT_BASE_COLUMNS)
    return [c for c in frame.columns if c in compact or c.startswith("ctx_")]


def build_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return eight fixed own-burden versus peer-context interactions.

    Each pair yields ``own - context`` and ``own / (1 + context)``. Missing
    input remains missing. A zero denominator yields missing rather than inf;
    real code-load means from the context builder are nonnegative.
    """
    required = {name for _, own, _, context in RELATIVE_PAIRS for name in (own, context)}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing relative feature columns: {sorted(missing)}")
    if not frame.columns.is_unique:
        raise ValueError("Feature columns must be unique")
    values = {name: pd.to_numeric(frame[name], errors="raise").astype(float) for name in required}
    columns = {}
    for own_label, own, context_label, context in RELATIVE_PAIRS:
        prefix = f"rel_{own_label}_vs_{context_label}"
        columns[f"{prefix}_difference"] = values[own] - values[context]
        denominator = 1.0 + values[context]
        columns[f"{prefix}_ratio"] = values[own] / denominator.where(denominator != 0)
    return pd.DataFrame(columns, index=frame.index)


def recency_weights(periods, reference, half_life_quarters: float = 4) -> np.ndarray:
    """Return mean-one weights with a fixed half-life in calendar quarters.

    All training periods must precede the reference quarter. Only dates enter
    this function; normalization uses training weights and never any targets.
    """
    if not np.isfinite(half_life_quarters) or half_life_quarters <= 0:
        raise ValueError("half_life_quarters must be finite and positive")
    dates = pd.DatetimeIndex(pd.to_datetime(periods, errors="raise"))
    reference_date = pd.Timestamp(reference)
    if len(dates) == 0 or dates.isna().any() or pd.isna(reference_date):
        raise ValueError("Training periods and reference must be nonempty valid dates")
    ordinals = dates.to_period("Q").asi8
    reference_ordinal = reference_date.to_period("Q").ordinal
    age = reference_ordinal - ordinals
    if np.any(age <= 0):
        raise ValueError("All training quarters must strictly precede the reference quarter")
    # Subtract a common exponent before exp2 to avoid underflow for long ranges;
    # normalization removes this common factor exactly mathematically.
    exponent = -age.astype(np.float64) / float(half_life_quarters)
    weights = np.exp2(exponent - exponent.max())
    return weights / weights.mean()
