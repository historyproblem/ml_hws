"""Historical supplied monthly targets, available only after target_end."""
import numpy as np
import pandas as pd
from spend_tuning.data import causal_history_check

LABEL_HISTORY_COLUMNS=['past_label_count','past_label_mean','past_label_zero','past_label_ewm']
LABEL_LAG_COLUMNS=[f'past_official_label_{k}' for k in range(1,7)]+['past_label_age','past_positive_label_age']


def label_history_features(objects,prior=5.,half_life=90.):
    objects=objects.reset_index(drop=True)
    basic=causal_history_check(objects,prior,half_life)
    lag=np.full((len(objects),len(LABEL_LAG_COLUMNS)),np.nan,dtype=np.float32)
    labels=objects.loc[objects.official&objects.spend_30d.notna()].sort_values('target_end')
    groups={str(cid):g for cid,g in labels.groupby('customer_id',sort=False)}
    for cid,g in objects.groupby('customer_id',sort=False):
        past=groups.get(str(cid))
        if past is None or not len(past):continue
        times=g.as_of_date.to_numpy(dtype='datetime64[ns]');ends=past.target_end.to_numpy(dtype='datetime64[ns]')
        y=np.log1p(past.spend_30d.to_numpy(float));n=np.searchsorted(ends,times,side='right')
        for k in range(1,7):
            ix=n-k;valid=ix>=0;lag[g.index[valid],k-1]=y[ix[valid]]
        valid=n>0
        lag[g.index[valid],6]=np.log1p((times[valid]-ends[n[valid]-1])/np.timedelta64(1,'D'))
        positive=ends[y>0];j=np.searchsorted(positive,times,side='right');valid=j>0
        lag[g.index[valid],7]=np.log1p((times[valid]-positive[j[valid]-1])/np.timedelta64(1,'D'))
    assert not np.isinf(basic).any() and not np.isinf(lag).any()
    return basic,lag


def restrict_to_target_cohort(objects,train_indices,cutoff):
    cutoff=pd.Timestamp(cutoff)
    target=objects.loc[objects.is_test,'as_of_date'].min()
    mask=objects.is_test if cutoff==target else objects.official&objects.as_of_date.eq(cutoff)
    ids=set(objects.loc[mask,'customer_id'].astype(str))
    return train_indices[objects.iloc[train_indices].customer_id.astype(str).isin(ids).to_numpy()]
