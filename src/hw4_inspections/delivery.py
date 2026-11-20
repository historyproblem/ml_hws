"""Fixed ensemble recipes, holdout choice and strict HW4 CSV validation.

All ranking is performed on the complete quarter supplied by the caller. These
helpers neither fit models nor read labels, files, or competition services.
"""
from __future__ import annotations

from collections.abc import Mapping
import numbers

import numpy as np
import pandas as pd


SEEDS = frozenset({42, 17, 2026})
HOLDOUT_QUARTERS = frozenset({"2024-04-01", "2024-07-01"})
FINALISTS = frozenset({"cb_rank50", "cb_visit_rank3"})


def _probabilities(values, *, name="probabilities", expected_n=None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric probabilities") from error
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if expected_n is not None and len(array) != expected_n:
        raise ValueError(f"{name} has a different number of rows")
    if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError(f"{name} must be finite probabilities in [0, 1]")
    return array


def rank_predictions(probabilities) -> np.ndarray:
    """Return float64 percentile ranks with average ranks for tied values."""
    values = _probabilities(probabilities)
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def make_candidates(ref_by_seed: dict[int, np.ndarray],
                    compact_by_seed: dict[int, np.ndarray], visit) -> dict[str, np.ndarray]:
    """Average each CB family's three seed probabilities, then rank once.

    The two ensemble formulas are frozen: equal halves of the CB ranks, or
    equal thirds of those same two CB ranks and the visit-model rank. Component
    candidates retain their mean probabilities, rather than percentile ranks.
    """
    for name, family in [("reference", ref_by_seed), ("compact", compact_by_seed)]:
        if not isinstance(family, Mapping) or set(family) != SEEDS:
            raise ValueError(f"{name} requires exactly seeds 42, 17 and 2026")
    expected_n = len(_probabilities(ref_by_seed[42], name="reference seed 42"))

    def probability_mean(family, name):
        arrays = [_probabilities(family[seed], name=f"{name} seed {seed}", expected_n=expected_n)
                  for seed in [42, 17, 2026]]
        return np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float64)

    reference = probability_mean(ref_by_seed, "reference")
    compact = probability_mean(compact_by_seed, "compact")
    visit = _probabilities(visit, name="visit", expected_n=expected_n)
    reference_rank, compact_rank = rank_predictions(reference), rank_predictions(compact)
    visit_rank = rank_predictions(visit)
    return {
        "cb_rank50": .5 * reference_rank + .5 * compact_rank,
        "cb_visit_rank3": (reference_rank + compact_rank + visit_rank) / 3.,
        "compact_cb": compact,
        "context_cb": reference,
    }


def validate_submission(test_rows: pd.DataFrame, sample: pd.DataFrame, probabilities,
                        expected_rows: int | None = None) -> pd.DataFrame:
    """Join test-aligned predictions by ID and return the exact sample order.

    Both ID collections must be nonmissing, unique, and identical. The generic
    helper has no fixed row count; production callers can require 2246 through
    ``expected_rows``. Neither input frame nor prediction array is modified.
    """
    for name, frame in [("test", test_rows), ("sample", sample)]:
        if not isinstance(frame, pd.DataFrame) or "row_id" not in frame:
            raise ValueError(f"{name} must be a DataFrame containing row_id")
        if not frame.columns.is_unique:
            raise ValueError(f"{name} columns must be unique")
        if frame.row_id.isna().any() or not frame.row_id.is_unique:
            raise ValueError(f"{name} row_id must be nonmissing and unique")
    if expected_rows is not None:
        if (isinstance(expected_rows, bool) or not isinstance(expected_rows, numbers.Integral)
                or expected_rows < 1):
            raise ValueError("expected_rows must be a positive integer or None")
        if len(test_rows) != expected_rows or len(sample) != expected_rows:
            raise ValueError("Submission row count differs from expected_rows")
    values = _probabilities(probabilities, expected_n=len(test_rows))
    if (len(test_rows) != len(sample) or not test_rows.row_id.isin(sample.row_id).all()
            or not sample.row_id.isin(test_rows.row_id).all()):
        raise ValueError("Test and sample row_id sets must match exactly")
    predictions = pd.DataFrame({"row_id": test_rows.row_id.to_numpy(copy=True),
                                "target": values.copy()})
    sample_ids = sample[["row_id"]].reset_index(drop=True).copy()
    submission = sample_ids.merge(predictions, on="row_id", how="left", sort=False,
                                  validate="one_to_one")
    if not submission.row_id.equals(sample_ids.row_id):
        raise ValueError("Submission join failed to preserve sample order")
    _probabilities(submission.target.to_numpy(), expected_n=len(sample))
    return submission[["row_id", "target"]]


def choose_primary(quarter_scores) -> str:
    """Promote the visit blend only if it ties/wins both frozen holdouts.

    The default is the two-CB ensemble. The visit ensemble must have nonnegative
    AP differences on both quarters and a strictly positive mean difference.
    """
    if not isinstance(quarter_scores, Mapping) or set(quarter_scores) != HOLDOUT_QUARTERS:
        raise ValueError("Scores require exactly the 2024Q2 and 2024Q3 holdouts")
    differences = []
    for quarter in sorted(HOLDOUT_QUARTERS):
        scores = quarter_scores[quarter]
        if not isinstance(scores, Mapping) or set(scores) != FINALISTS:
            raise ValueError("Each holdout requires exactly the two frozen finalists")
        converted = {}
        for name, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError("Holdout AP scores must be numeric scalars")
            value = float(value)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Holdout AP scores must be finite and in [0, 1]")
            converted[name] = value
        differences.append(converted["cb_visit_rank3"] - converted["cb_rank50"])
    return "cb_visit_rank3" if min(differences) >= 0 and np.mean(differences) > 0 else "cb_rank50"
