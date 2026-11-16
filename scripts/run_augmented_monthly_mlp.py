"""One fixed MLP on already verified historical snapshots, with temporal purge."""
import argparse
import hashlib
import json
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import CachedData,FEATURES,atomic_json,digest,split_indices,row_weights
from spend_baselines.suite_models import Neural
from spend_course10.history_features import HISTORY_COLUMNS
from course10_forest_followup import SEASON_COLUMNS
from run_monthly_history import frozen
from run_season_transfer import report


def window_calendar(x,objects,names):
    """Known calendar exposure of the exact future 30-day window; no events."""
    result=x.copy()
    for date,group in objects.groupby('as_of_date',sort=False):
        share=np.bincount(pd.date_range(date,periods=30).month,minlength=13)[1:]/30.
        result[np.ix_(group.index,[names.index(f'month_{m}') for m in range(1,13)])]=share
        result[group.index,names.index('month_sin')]=share@np.sin(2*np.pi*np.arange(1,13)/12)
        result[group.index,names.index('month_cos')]=share@np.cos(2*np.pi*np.arange(1,13)/12)
    return result


def execute(args,cfg,o,x,names,date,final=False):
    name=cfg['profile']['name'];directory=args.out/('refit' if final else 'jobs')/name
    if not final:directory/=date
    marker=directory/'result.json'
    if marker.exists():
        r=json.loads(marker.read_text())
        assert r['profile']==cfg['profile']
        for filename,sha in r['sha256'].items():assert digest(directory/filename)==sha
        return None if final else pd.read_csv(directory/'predictions.csv.gz',dtype={'Id':str,'customer_id':str})
    tr=split_indices(o,'D2',date)
    vi=np.flatnonzero(o.is_test.to_numpy()) if final else np.flatnonzero((o.official&o.as_of_date.eq(date)).to_numpy())
    inner=o.loc[o.official&o.target_end.le(date),'as_of_date'].max()
    ti=split_indices(o,'D2',inner);iv=np.flatnonzero((o.official&o.as_of_date.eq(inner)).to_numpy())
    assert o.iloc[tr].target_end.max()<=pd.Timestamp(date)
    assert o.iloc[iv].target_end.max()<=pd.Timestamp(date)
    assert not o.iloc[vi].Id.isin(o.iloc[tr].Id).any()
    w=row_weights(o,tr,'D2');iw=row_weights(o,ti,'D2')
    settings=dict(cfg['neural_base'])
    if args.smoke:settings.update(max_epochs=2,patience=1)
    start=time.monotonic()
    selector=Neural('mlp','direct',settings,42,args.device,args.threads)
    selector.fit(x[ti],o.iloc[ti].spend_30d.to_numpy(float),iw,
                 validation=(x[iv],o.iloc[iv].spend_30d.to_numpy(float)))
    epochs=selector.best_epoch;directory.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(selector.history).to_csv(directory/'selection.csv',index=False);del selector
    model=Neural('mlp','direct',settings,42,args.device,args.threads)
    model.fit(x[tr],o.iloc[tr].spend_30d.to_numpy(float),w,epochs=epochs)
    assert model.network.output.out_features==1
    model.save(directory/'model.pt');pd.DataFrame(model.history).to_csv(directory/'refit.csv',index=False)
    np.savez_compressed(directory/'weights.npz',train_Id=o.iloc[tr].Id.to_numpy(str),w=w)
    fit_seconds=time.monotonic()-start
    start=time.monotonic();z=model.predict(x[vi]);inference_seconds=time.monotonic()-start
    np.testing.assert_array_equal(z,Neural.load(directory/'model.pt',args.device,args.threads).predict(x[vi]))
    meta={'profile':cfg['profile'],'settings':settings,'dataset':'D2','seed':42,'one_predictor':True,'smoke':args.smoke,
          'n_train':len(tr),'n_train_official':int(o.iloc[tr].official.sum()),'weight_sum':float(w.sum()),
          'max_target_end':str(o.iloc[tr].target_end.max()),'inner_date':str(inner),'selected_epoch':epochs,
          'fit_seconds':fit_seconds,'inference_seconds':inference_seconds,'checkpoint_forward_exact':True,'Kaggle_uploaded':False}
    if final:
        filename=('smoke_' if args.smoke else 'submission_')+name+f'_{args.device}.csv'
        sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str})
        if args.smoke:sample=sample[sample.Id.isin(o.iloc[vi].Id)]
        write_submission(o.iloc[vi],z,directory/filename,sample_ids=sample.Id)
        a=pd.read_csv(directory/filename,dtype={'Id':str})
        assert a.Id.tolist()==sample.Id.tolist() and np.isfinite(a.spend_30d).all() and (a.spend_30d>=0).all()
        ordered=pd.Series(z,index=o.iloc[vi].Id).loc[sample.Id].to_numpy()
        np.testing.assert_allclose(np.log1p(a.spend_30d),ordered,rtol=0,atol=1e-12)
        np.savez_compressed(directory/'inputs_predictions.npz',Id=o.iloc[vi].Id.to_numpy(str),X=x[vi],z=z)
        meta.update(n_test=len(vi),feature_names=names,mean_prediction_z=float(z.mean()),upload_ready=not args.smoke)
        frame=None
    else:
        frame=o.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
        frame['date']=date;frame['profile']=name;frame['z']=z;frame['target_z']=np.log1p(o.iloc[vi].spend_30d.to_numpy(float))
        frame.to_csv(directory/'predictions.csv.gz',index=False)
        meta['RMSLE']=float(np.sqrt(np.mean((frame.z-frame.target_z)**2)))
    meta['sha256']={p.name:digest(p) for p in directory.iterdir() if p.name!='result.json'}
    atomic_json(marker,meta)
    if final and not args.smoke:
        args.quick.mkdir(parents=True,exist_ok=True);dest=args.quick/filename
        if dest.exists() and digest(dest)!=digest(directory/filename):raise ValueError('Different CSV already exists')
        shutil.copyfile(directory/filename,dest);atomic_json(args.quick/(filename+'.json'),meta)
    print('CSV_READY' if final else 'VALID',name,date,meta.get('RMSLE'),flush=True)
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/course10_augmented_monthly_mlp.json'))
    parser.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    parser.add_argument('--features',type=Path,default=Path('outputs/course10-snapshot-history'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'))
    parser.add_argument('--reference',type=Path,default=Path('outputs/season-transfer'))
    parser.add_argument('--out',type=Path,default=Path('outputs/augmented-monthly-mlp'))
    parser.add_argument('--quick',type=Path,default=Path('outputs/quick-variants-20261116'))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--refit-only',action='store_true',help='Skip outer validation fits and reference-dependent reports')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    assert cfg['dataset']=='D2' and cfg['seed']==42 and cfg['task']=='direct' and cfg['max_seminar']==10
    args.out.mkdir(parents=True,exist_ok=True)
    state=json.loads((args.features/'features.ready.json').read_text());old=state['identity']
    assert digest(args.features/'features.npy')==state['sha256']
    assert digest(args.cache/'ready.json')==old['cache_sha256']
    for name,sha in old['input_sha256'].items():assert digest(args.data/name)==sha
    for name,sha in old['source_sha256'].items():assert digest(name)==sha
    cache_state=json.loads((args.cache/'ready.json').read_text())
    assert digest(args.cache/'objects.csv.gz')==cache_state['sha256']['objects.csv.gz']
    assert state['checks']['expanded_targets_checked']==194949 and state['checks']['expanded_target_max_error']==0
    source=['scripts/run_augmented_monthly_mlp.py','src/spend_baselines/suite_models.py','src/spend_baselines/suite_data.py']
    frozen(args.out/'study.json',{'config':cfg,'smoke':args.smoke,'device':args.device,'threads':args.threads,
           'libraries':{n:version(n) for n in ['numpy','pandas','torch']},'feature_ready_sha256':digest(args.features/'features.ready.json'),
           'source_sha256':{n:digest(n) for n in source}})
    with threadpool_limits(limits=args.threads):
        cache=CachedData(args.cache);o=cache.objects
        names=list(FEATURES['F0'])+SEASON_COLUMNS+HISTORY_COLUMNS
        x=np.load(args.features/'features.npy',allow_pickle=False)
        assert x.shape==(len(o),len(names))
        x=window_calendar(x,o,names)
        if args.smoke:
            keep=sorted(o.customer_id.unique(),key=lambda v:hashlib.sha256(str(v).encode()).hexdigest())[:64]
            ix=np.flatnonzero(o.customer_id.isin(keep).to_numpy());o=o.iloc[ix].reset_index(drop=True);x=x[ix]
        name=cfg['profile']['name']
        if args.refit_only:
            execute(args,cfg,o,x,names,o.loc[o.is_test,'as_of_date'].min(),True)
            return
        frames={name:pd.concat([execute(args,cfg,o,x,names,date) for date in cfg['validation_dates']],ignore_index=True)}
        execute(args,cfg,o,x,names,o.loc[o.is_test,'as_of_date'].min(),True)
        if not args.smoke:
            chunks=[]
            for date in cfg['validation_dates']:
                directory=args.reference/'jobs/reference'/date;r=json.loads((directory/'result.json').read_text())
                assert digest(directory/'predictions.csv.gz')==r['sha256']['predictions.csv.gz']
                chunks.append(pd.read_csv(directory/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}))
            frames['reference']=pd.concat(chunks,ignore_index=True)
            report(args,o,[{'name':'reference'},{'name':name}],frames)


if __name__=='__main__':main()
