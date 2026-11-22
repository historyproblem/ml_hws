"""Fixed unlabeled query-cohort normalization of peer-context columns only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .classical_features import COMPACT_BASE_COLUMNS


def quarter_rank_context(frame, periods):
    """Replace72 ctx_ coordinates by within-period average percentile ranks.

    This is explicitly batch-transductive: each available unlabeled query
    cohort supplies its own ranks. Labels and raw IDs never enter this API.
    All own-history/calendar values and numeric missingness remain unchanged.
    """
    if (not frame.columns.is_unique or not frame.index.equals(pd.RangeIndex(len(frame)))
            or {'target','row_id','site_id','forecast_period','history_before'} & set(frame)):
        raise ValueError('Expected a canonical identifier-free feature matrix')
    periods=pd.Series(periods).reset_index(drop=True)
    if len(periods)!=len(frame) or periods.isna().any():
        raise ValueError('Unlabeled quarter groups must align with feature rows')
    columns=[c for c in frame if c.startswith('ctx_')]
    if len(columns)!=72:raise ValueError('Expected the frozen72 peer-context fields')
    out=frame.copy()
    for name in columns:
        values=pd.to_numeric(frame[name],errors='raise')
        if np.isinf(values.to_numpy(float)).any():raise ValueError('Infinite context value')
        ranked=values.groupby(periods,sort=False).rank(method='average',pct=True)
        if not np.array_equal(values.isna(),ranked.isna()):raise AssertionError('Missingness changed')
        out[name]=ranked.to_numpy()
    return out


def columns_for_view(frame,view):
    if view=='full346':
        names=list(frame)
        if len(names)!=346:raise ValueError('Expected full346')
    elif view=='compact94':
        names=[c for c in frame if (c in COMPACT_BASE_COLUMNS and c not in {'forecast_year','cutoff_gap_days'}) or c.startswith('ctx_')]
        if len(names)!=94:raise ValueError('Expected compact94 (own22+peer72)')
    else:raise ValueError('Unknown fixed rank-context view')
    return names
