"""Past transaction summaries on the same log scale as the prediction target."""
import numpy as np
import pandas as pd

LAG_COLUMNS=[f'past_log_spend_{k:02d}' for k in range(1,13)]+[f'past_log_days_{k:02d}' for k in range(1,13)]
SUMMARY_COLUMNS=[f'{name}_{k}' for k in (3,6,12) for name in
                 ('mean_log_spend','std_log_spend','zero_fraction','positive_mean_log_spend',
                  'mean_log_days','covered_fraction','ewm_log_spend')]+['recent_empty_months','active_months12']
HISTORY_COLUMNS=LAG_COLUMNS+SUMMARY_COLUMNS


def monthly_history_features(objects,history,history_start):
    """Only raw purchase days in [cutoff-360,cutoff); no target columns read.

    Each lag is a completed 30-day interval. Missing source coverage is NaN;
    a covered empty interval is a real zero. EWM half-life is fixed at3months.
    """
    result=np.full((len(objects),len(HISTORY_COLUMNS)),np.nan,dtype=np.float32)
    start=np.datetime64(pd.Timestamp(history_start).date(),'D').astype(np.int64)
    for cid,group in objects.reset_index(drop=True).groupby('customer_id',sort=False):
        t=group.as_of_date.to_numpy(dtype='datetime64[D]').astype(np.int64)
        left=t[:,None]-30*np.arange(1,13)[None,:];right=left+30
        h=history.customers.get(str(cid))
        dates,cumulative=(h[0],h[2]) if h is not None else (np.array([],dtype=int),np.array([0.]))
        lo,hi=np.searchsorted(dates,left,side='left'),np.searchsorted(dates,right,side='left')
        covered=left>=start
        amounts=np.maximum(cumulative[hi]-cumulative[lo],0)
        z=np.where(covered,np.log1p(amounts),np.nan)
        days=np.where(covered,np.log1p(hi-lo),np.nan)
        features=[z,days]
        for k in (3,6,12):
            a=z[:,:k];known=np.isfinite(a);n=known.sum(axis=1);positive=(a>0)&known;np_=positive.sum(axis=1)
            total=np.nansum(a,axis=1)
            mean=np.divide(total,n,out=np.full(len(t),np.nan),where=n>0)
            variance=np.divide(np.nansum((a-mean[:,None])**2,axis=1),n,out=np.full(len(t),np.nan),where=n>0)
            zero=np.divide(((a==0)&known).sum(axis=1),n,out=np.full(len(t),np.nan),where=n>0)
            positive_mean=np.divide(total,np_,out=np.full(len(t),np.nan),where=np_>0)
            mean_days=np.divide(np.nansum(days[:,:k],axis=1),n,out=np.full(len(t),np.nan),where=n>0)
            weights=np.exp2(-np.arange(k)/3.)[None,:]*known
            weighted=np.divide((np.nan_to_num(a)*weights).sum(axis=1),weights.sum(axis=1),out=np.full(len(t),np.nan),where=weights.sum(axis=1)>0)
            features.append(np.column_stack([mean,np.sqrt(variance),zero,positive_mean,mean_days,n/k,weighted]))
        empty=(z==0)&np.isfinite(z)
        run=np.cumprod(empty.astype(int),axis=1).sum(axis=1).astype(float)
        run[~np.isfinite(z[:,0])]=np.nan
        features.append(np.column_stack([run,((z>0)&np.isfinite(z)).sum(axis=1)]))
        result[group.index]=np.column_stack(features)
    assert result.shape==(len(objects),len(HISTORY_COLUMNS)) and not np.isinf(result).any()
    return result
