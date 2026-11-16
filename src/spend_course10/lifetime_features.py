"""Observed customer exposure and recurrence; only purchase days before cutoff."""
import numpy as np
import pandas as pd

FREQUENCY_COLUMNS=['observed_log_days_all','observed_log_repeat_rate30','observed_recency_fraction',
 'observed_log_mean_gap','observed_log_gap_cv','observed_log_last_gap','observed_log_recency_over_gap',
 'observed_log_days365','observed_left_censored']
SUMMARY_COLUMNS=['observed_active_mean_log','observed_active_std_log','observed_active_zero_fraction',
 'observed_active_positive_mean_log','observed_log_active_windows','observed_active_ewm_log',
 'observed_active_mean_log12','observed_active_zero_fraction12']
LIFETIME_COLUMNS=FREQUENCY_COLUMNS+SUMMARY_COLUMNS


def lifetime_features(objects,history,history_start):
    result=np.full((len(objects),len(LIFETIME_COLUMNS)),np.nan,dtype=np.float32)
    start=np.datetime64(pd.Timestamp(history_start).date(),'D').astype(np.int64)
    for cid,group in objects.reset_index(drop=True).groupby('customer_id',sort=False):
        t=group.as_of_date.to_numpy(dtype='datetime64[D]').astype(np.int64)
        h=history.customers.get(str(cid));d,c=(h[0],h[2]) if h else (np.array([],dtype=int),np.array([0.]))
        n=np.searchsorted(d,t,side='left')
        first=np.where(n>0,d[0],t) if len(d) else t.copy()
        last=np.where(n>0,d[np.maximum(n-1,0)],t) if len(d) else t.copy()
        exposure=t-first;recency=t-last;repeats=np.maximum(n-1,0)
        mean_gap=np.divide(last-first,repeats,out=np.full(len(t),np.nan),where=repeats>0)
        cv=np.full(len(t),np.nan);lastgap=cv.copy()
        for j,right in enumerate(n):
            if right>=2:
                gaps=np.diff(d[:right]);lastgap[j]=gaps[-1]
                if len(gaps)>=2 and gaps.mean()>0:cv[j]=gaps.std()/gaps.mean()
        ratio=np.divide(recency,mean_gap,out=np.full(len(t),np.nan),where=mean_gap>0)
        freq=np.c_[np.log1p(n),np.log1p(30*repeats/np.maximum(exposure,30)),
            np.divide(recency,np.maximum(exposure,1)),np.log1p(mean_gap),np.log1p(cv),np.log1p(lastgap),
            np.log1p(ratio),np.log1p(n-np.searchsorted(d,t-365)),np.where(n>0,(first<=start+30).astype(float),np.nan)]
        left=t[:,None]-30*np.arange(1,25)[None,:];right=left+30
        lo,hi=np.searchsorted(d,left),np.searchsorted(d,right)
        known=(left>=start)&(right>first[:,None])&(n[:,None]>0)
        z=np.where(known,np.log1p(np.maximum(c[hi]-c[lo],0)),np.nan)
        count=known.sum(axis=1);total=np.nansum(z,axis=1)
        mean=np.divide(total,count,out=np.full(len(t),np.nan),where=count>0)
        std=np.sqrt(np.divide(np.nansum((z-mean[:,None])**2,axis=1),count,out=np.full(len(t),np.nan),where=count>0))
        zero=np.divide(((z==0)&known).sum(axis=1),count,out=np.full(len(t),np.nan),where=count>0)
        positive=(z>0).sum(axis=1);pm=np.divide(total,positive,out=np.full(len(t),np.nan),where=positive>0)
        w=np.exp2(-np.arange(24)/6.)[None,:]*known
        ewm=np.divide((np.nan_to_num(z)*w).sum(axis=1),w.sum(axis=1),out=np.full(len(t),np.nan),where=w.sum(axis=1)>0)
        cnt12=known[:,:12].sum(axis=1)
        mean12=np.divide(np.nansum(z[:,:12],axis=1),cnt12,out=np.full(len(t),np.nan),where=cnt12>0)
        zero12=np.divide(((z[:,:12]==0)&known[:,:12]).sum(axis=1),cnt12,out=np.full(len(t),np.nan),where=cnt12>0)
        result[group.index]=np.c_[freq,mean,std,zero,pm,np.log1p(count),ewm,mean12,zero12]
    assert not np.isinf(result).any()
    return result
