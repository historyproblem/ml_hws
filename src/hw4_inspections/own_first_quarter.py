"""Own-site observed first routines in conservative completed past quarters."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .seasonal_context import _prepare, _first_routines

FEATURES=(
 'own_firstq_n','own_firstq_fail_n','own_firstq_fail_mean',
 'own_firstq_last_fail','own_firstq_last_load','own_firstq_last_age_days',
 'own_firstq_last2_fail_mean','own_firstq_last4_fail_mean',
 'own_firstq_same_season_n','own_firstq_same_season_fail_mean','own_firstq_same_season_last_fail',
)


def build_own_first_quarter(rows,history):
    """Same snapshot, strict past cutoff, no current-row target or peer outcomes.

    The observed snapshot envelope must cover a quarter's start-containing
    Monday. Only fully ended quarters are admitted. Earliest ambiguous weekly
    cells/conflicting or unknown outcomes exclude that site-quarter; a later
    known event is never substituted. This envelope is not proof of source
    completeness, nor does zero recorded load imply no actual violations.
    """
    r,h=_prepare(rows,history)
    output=pd.DataFrame(np.nan,index=np.arange(len(r)),columns=FEATURES,dtype=float)
    output[['own_firstq_n','own_firstq_fail_n','own_firstq_same_season_n']]=0.
    for (forecast,cutoff),query in r.groupby(['forecast_period','history_before'],sort=False):
        snapshot=h.loc[h.forecast_period.eq(forecast)&h._week.lt(cutoff)].copy()
        if snapshot.empty:continue
        if not snapshot._week.dt.dayofweek.eq(0).all():raise ValueError('Weekly dates must be Mondays')
        envelope=snapshot._week.min()
        periods=pd.period_range(envelope.to_period('Q'),forecast.to_period('Q')-1,freq='Q')
        parts=[]
        for quarter in periods:
            start,end=quarter.start_time,(quarter+1).start_time
            monday=start-pd.Timedelta(days=start.dayofweek)
            if end>cutoff or start>=forecast or monday<envelope:continue
            first=_first_routines(snapshot,start,end)
            first=first.loc[first.site_id.isin(query.site_id),['site_id','_week','_fail','_load']].copy()
            if len(first):
                first['_quarter']=quarter.ordinal;first['_season']=quarter.quarter;parts.append(first)
        if not parts:continue
        recorded=pd.concat(parts,ignore_index=True).sort_values(['_quarter','_week'],kind='stable')
        mapping={}
        for site,visits in recorded.groupby('site_id',sort=False):
            last=visits.iloc[-1];same=visits.loc[visits._season.eq(forecast.quarter)]
            mapping[site]=[float(len(visits)),float(visits._fail.sum()),float(visits._fail.mean()),
                float(last._fail),float(last._load),float((cutoff-last._week).days),
                float(visits._fail.tail(2).mean()),float(visits._fail.tail(4).mean()),float(len(same)),
                float(same._fail.mean())if len(same)else np.nan,float(same._fail.iloc[-1])if len(same)else np.nan]
        for pos,site in zip(query._row,query.site_id):
            if site in mapping:output.loc[int(pos),:]=mapping[site]
    output.index=rows.index
    return output
