"""Product-season and popularity aggregates computed strictly before each cutoff."""
import numpy as np
import pandas as pd
from spend_embeddings.data import build_bags

ITEM_MEANS=['item_current30','item_current90','item_current180','item_current_trend',
            'item_year30','item_year90','item_year_season','item_year_growth90','item_year_share']
ITEM_SEASON_COLUMNS=ITEM_MEANS+['item_year_season_std','item_year_positive_fraction',
                              'item_year_seen_fraction','item_bag_log_unique','item_bag_retained_fraction']
CURRENT_COLUMNS=ITEM_MEANS[:4]+ITEM_SEASON_COLUMNS[-2:]
YEAR_COLUMNS=ITEM_MEANS[4:]+ITEM_SEASON_COLUMNS[9:12]


def item_season_features(objects,lines,history_start,*,prior=5.,max_items=128):
    """Weighted buyer history meets prior-year global product counts.

    Product codes are symbolic, not a learned vocabulary. Global counts use
    only windows ending before this row's cutoff; future products have no mass.
    A customer/product/day counts once. Smoothing strength is fixed, not tuned.
    """
    if prior<=0 or max_items<=0:raise ValueError('Positive smoothing and bag size required')
    objects=objects.reset_index(drop=True)
    names=np.array(sorted(lines.item_id.unique()),dtype=str)
    if not len(names):return np.full((len(objects),len(ITEM_SEASON_COLUMNS)),np.nan,dtype=np.float32)
    settings={'window_days':180,'half_life_days':90,'max_items':max_items}
    ids,weights,diagnostics=build_bags(objects,lines,names,settings)
    events=lines[['customer_id','item_id','event_ts']].copy()
    events['day']=events.event_ts.dt.normalize()
    events=events.drop_duplicates(['customer_id','item_id','day']).sort_values('day')
    codes=pd.Index(names).get_indexer(events.item_id)+1
    dates=events.day.to_numpy(dtype='datetime64[D]').astype(np.int64)
    start=np.datetime64(pd.Timestamp(history_start).date(),'D').astype(np.int64)
    output=np.full((len(objects),len(ITEM_SEASON_COLUMNS)),np.nan,dtype=np.float32)
    for date,group in objects.groupby('as_of_date',sort=True):
        cutoff=pd.Timestamp(date);t=np.datetime64(cutoff.date(),'D').astype(np.int64)
        ya=np.datetime64((cutoff-pd.DateOffset(years=1)).date(),'D').astype(np.int64)
        def counts(left,right):
            if left<start or right>t:return np.full(len(names)+1,np.nan)
            lo,hi=np.searchsorted(dates,[left,right],side='left')
            return np.bincount(codes[lo:hi],minlength=len(names)+1).astype(float)
        n30,n90,n180=counts(t-30,t),counts(t-90,t),counts(t-180,t)
        y30,y90=counts(ya,ya+30),counts(ya-90,ya)
        with np.errstate(invalid='ignore',divide='ignore'):
            season=np.log((y30+prior)/(y90/3+prior))
            table=np.column_stack([np.log1p(n30),np.log1p(n90/3),np.log1p(n180/6),
                np.log((n30+prior)/(n90/3+prior)),np.log1p(y30),np.log1p(y90/3),season,
                np.log((n90/3+prior)/(y90/3+prior)),y30/(y30+y90+prior)])
        table[0]=np.nan
        rows=group.index.to_numpy();v=table[ids[rows]];w=weights[rows].astype(float)
        known=np.isfinite(v);mass=(known*w[:,:,None]).sum(axis=1)
        mean=np.divide((np.nan_to_num(v)*w[:,:,None]).sum(axis=1),mass,out=np.full_like(mass,np.nan),where=mass>0)
        s=v[:,:,6];sk=np.isfinite(s);sm=(w*sk).sum(axis=1)
        var=np.divide((np.nan_to_num(s-mean[:,6,None])**2*w*sk).sum(axis=1),sm,out=np.full(len(rows),np.nan),where=sm>0)
        positive=np.divide((w*(s>0)*sk).sum(axis=1),sm,out=np.full(len(rows),np.nan),where=sm>0)
        seen=y30[ids[rows]];known_seen=np.isfinite(seen)&(ids[rows]>0);seen_mass=(w*known_seen).sum(axis=1)
        fraction=np.divide((w*(seen>0)*known_seen).sum(axis=1),seen_mass,out=np.full(len(rows),np.nan),where=seen_mass>0)
        output[rows]=np.c_[mean,np.sqrt(var),positive,fraction,np.log1p(diagnostics[rows,0]),diagnostics[rows,1]]
    assert not np.isinf(output).any()
    return output
