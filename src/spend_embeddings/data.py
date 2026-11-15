"""Causal product-day bags; numeric cache codes are not a learned vocabulary."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spend_baselines.suite_data import atomic_json, digest
from spend_behavior.data import positive_lines


def build_bags(objects, lines, item_names, settings):
    """A purchase-day contributes once per product, with exponential recency.

    Bags use only [t-window, t). Top-K is chosen by decayed mass, ties by item
    name, and kept weights sum to one. No prices or future item statistics.
    """
    n,k=len(objects),settings['max_items']
    ids=np.zeros((n,k),dtype=np.uint32)
    weights=np.zeros((n,k),dtype=np.float32)
    diagnostics=np.zeros((n,2),dtype=np.float32)
    code={name:i+1 for i,name in enumerate(item_names)}
    events=lines.assign(day=lines.event_ts.dt.normalize()).drop_duplicates(['customer_id','day','item_id'])
    groups={str(cid):g.sort_values('day') for cid,g in events.groupby('customer_id',sort=False)}
    for number,(cid,rows) in enumerate(objects.groupby('customer_id',sort=False)):
        g=groups.get(str(cid))
        if g is None:
            continue
        days=g.day.to_numpy().astype('datetime64[D]').astype(np.int64)
        items=np.array([code[x] for x in g.item_id],dtype=np.uint32)
        for row in rows.itertuples():
            t=int(np.datetime64(row.as_of_date,'D').astype(np.int64))
            lo,hi=np.searchsorted(days,[t-settings['window_days'],t])
            if lo==hi:
                continue
            unique,inverse=np.unique(items[lo:hi],return_inverse=True)
            mass=np.bincount(inverse,weights=np.exp2((days[lo:hi]-t)/settings['half_life_days']))
            selected=np.lexsort((unique,-mass))[:k]
            kept=mass[selected]
            ids[row.Index,:len(selected)]=unique[selected]
            weights[row.Index,:len(selected)]=kept/kept.sum()
            diagnostics[row.Index]=len(unique),kept.sum()/mass.sum()
        if number%1000==0:
            print(f'Product bags: {number}/{len(groups)} customers',flush=True)
    return ids,weights,diagnostics


class ProductBags:
    def __init__(self, data, data_dir, out_dir, settings):
        self.data,self.settings=data,settings
        root=Path(out_dir)/'product_cache'
        root.mkdir(parents=True,exist_ok=True)
        rows=np.flatnonzero(data.active)
        identity={'transactions_sha256':digest(Path(data_dir)/'transactions.csv'),
                  'objects_sha256':digest(data.base.directory/'objects.csv.gz'),
                  'active_rows_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),
                  'settings':settings,'source_sha256':digest(Path(__file__))}
        marker=root/'ready.json'
        if marker.exists():
            state=json.loads(marker.read_text())
            if state['identity']!=identity:
                raise ValueError('Product cache definition changed; use a new output directory')
            for name,sha in state['sha256'].items():
                if digest(root/name)!=sha:
                    raise ValueError(f'Corrupt product cache: {name}')
        else:
            lines,_=positive_lines(data_dir)
            objects=data.objects.iloc[rows].reset_index(drop=True)
            lines=lines[lines.customer_id.isin(objects.customer_id.unique())]
            names=sorted(lines.item_id.unique())
            ids,weights,diagnostics=build_bags(objects,lines,names,settings)
            np.save(root/'ids.npy',ids)
            np.save(root/'weights.npy',weights)
            np.save(root/'diagnostics.npy',diagnostics)
            atomic_json(root/'items.json',{'items':names})
            atomic_json(marker,{'identity':identity,'rows':len(rows),
                                'sha256':{n:digest(root/n) for n in ['ids.npy','weights.npy','diagnostics.npy','items.json']}})
        self.ids=np.load(root/'ids.npy',mmap_mode='r')
        self.weights=np.load(root/'weights.npy',mmap_mode='r')
        self.diagnostics=np.load(root/'diagnostics.npy',mmap_mode='r')
        self.item_names=json.loads((root/'items.json').read_text())['items']
        self.positions=np.full(len(data.objects),-1,dtype=np.int64)
        self.positions[rows]=np.arange(len(rows))
        self.identity={'ready_sha256':digest(marker),'items':len(self.item_names),'rows':len(rows)}
        self.vocabularies={}

    def take(self,indices):
        positions=self.positions[indices]
        if (positions<0).any():
            raise ValueError('Inactive rows requested from product cache')
        return np.asarray(self.ids[positions]),np.asarray(self.weights[positions])

    def vocabulary(self,indices,min_customers):
        key=(hashlib.sha256(np.asarray(indices,dtype=np.int64).tobytes()).hexdigest(),min_customers)
        if key not in self.vocabularies:
            ids,_=self.take(indices)
            customers=self.data.objects.iloc[indices].customer_id.to_numpy()
            mapping,support=fit_vocabulary(ids,customers,len(self.item_names),min_customers)
            self.vocabularies[key]=(mapping,support)
        return self.vocabularies[key]


def fit_vocabulary(ids,customers,item_count,min_customers):
    """Support counts unique customers in the purged training bags, never rows."""
    support=np.zeros(item_count+1,dtype=np.int64)
    grouping=pd.Series(np.arange(len(customers))).groupby(np.asarray(customers),sort=False)
    for _,positions in grouping:
        unique=np.unique(ids[positions.to_numpy()])
        support[unique]+=1
    kept=np.flatnonzero((support>=min_customers)&(np.arange(len(support))>0))
    mapping=np.zeros(item_count+1,dtype=np.int64)
    mapping[kept]=np.arange(1,len(kept)+1)
    support[0]=0
    return mapping,support
