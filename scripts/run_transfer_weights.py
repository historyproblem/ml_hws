"""Single monthly MLP with fixed covariate/seasonal weights; no HPO grid."""
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
from spend_baselines.suite_data import atomic_json,digest
from spend_baselines.suite_models import Neural
from spend_course10.transfer_weights import sample_weights
from run_monthly_history import prepare,frozen
from run_season_transfer import indices,report


def fit_at(args,cfg,objects,x,domain,profile,date,vi,directory):
    tr=indices(objects,date)
    inner=objects.iloc[tr].as_of_date.max()
    ti,iv=indices(objects,inner),indices(objects,inner,True)
    assert objects.iloc[tr].target_end.max()<=pd.Timestamp(date)
    assert objects.iloc[iv].target_end.max()<=pd.Timestamp(date)
    settings=dict(cfg['neural_base'])
    if args.smoke:settings.update(max_epochs=2,patience=1)
    start=time.monotonic()
    iw,inner_metadata=sample_weights(profile,objects.iloc[ti].as_of_date,domain[ti],domain[iv],inner)
    w,metadata=sample_weights(profile,objects.iloc[tr].as_of_date,domain[tr],domain[vi],date)
    selector=Neural('mlp','direct',settings,42,args.device,args.threads)
    selector.fit(x[ti],objects.iloc[ti].spend_30d.to_numpy(float),iw,
                 validation=(x[iv],objects.iloc[iv].spend_30d.to_numpy(float)))
    epochs=selector.best_epoch
    directory.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(selector.history).to_csv(directory/'selection.csv',index=False)
    del selector
    model=Neural('mlp','direct',settings,42,args.device,args.threads)
    model.fit(x[tr],objects.iloc[tr].spend_30d.to_numpy(float),w,epochs=epochs)
    assert model.network.output.out_features==1
    model.save(directory/'model.pt')
    pd.DataFrame(model.history).to_csv(directory/'refit.csv',index=False)
    np.savez_compressed(directory/'weights.npz',train_Id=objects.iloc[tr].Id.to_numpy(str),w=w,
                        inner_train_Id=objects.iloc[ti].Id.to_numpy(str),inner_w=iw)
    fit_seconds=time.monotonic()-start
    start=time.monotonic();z=model.predict(x[vi]);inference_seconds=time.monotonic()-start
    loaded=Neural.load(directory/'model.pt',args.device,args.threads)
    np.testing.assert_array_equal(z,loaded.predict(x[vi]))
    return z,{'settings':settings,'selected_epoch':epochs,'inner_date':str(inner),'n_train':len(tr),
              'max_target_end':str(objects.iloc[tr].target_end.max()),'weights':metadata,'inner_weights':inner_metadata,
              'fit_seconds':fit_seconds,'inference_seconds':inference_seconds,'checkpoint_forward_exact':True}


def execute(args,cfg,objects,x,names,domain,profile,date,final=False):
    directory=args.out/('refit' if final else 'jobs')/profile['name']
    if not final:directory/=date
    marker=directory/'result.json'
    if marker.exists():
        r=json.loads(marker.read_text());assert r['profile']==profile
        for name,sha in r['sha256'].items():assert digest(directory/name)==sha
        return None if final else pd.read_csv(directory/'predictions.csv.gz',dtype={'Id':str,'customer_id':str})
    vi=np.flatnonzero(objects.is_test.to_numpy()) if final else indices(objects,date,True)
    z,meta=fit_at(args,cfg,objects,x,domain,profile,date,vi,directory)
    if final:
        filename=('smoke_' if args.smoke else 'submission_')+profile['name']+f'_{args.device}.csv'
        sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str})
        if args.smoke:sample=sample[sample.Id.isin(objects.iloc[vi].Id)]
        write_submission(objects.iloc[vi],z,directory/filename,sample_ids=sample.Id)
        a=pd.read_csv(directory/filename,dtype={'Id':str})
        assert a.Id.tolist()==sample.Id.tolist() and np.isfinite(a.spend_30d).all() and (a.spend_30d>=0).all()
        ordered=pd.Series(z,index=objects.iloc[vi].Id).loc[sample.Id].to_numpy()
        np.testing.assert_allclose(np.log1p(a.spend_30d),ordered,rtol=0,atol=1e-12)
        np.savez_compressed(directory/'inputs_predictions.npz',Id=objects.iloc[vi].Id.to_numpy(str),X=x[vi],z=z)
        frame=None
        meta.update(n_test=len(vi),feature_names=names,mean_prediction_z=float(z.mean()),upload_ready=not args.smoke)
    else:
        frame=objects.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
        frame['date']=date;frame['profile']=profile['name'];frame['z']=z
        frame['target_z']=np.log1p(objects.iloc[vi].spend_30d.to_numpy(float))
        frame.to_csv(directory/'predictions.csv.gz',index=False)
        meta['RMSLE']=float(np.sqrt(np.mean((frame.z-frame.target_z)**2)))
    result={'profile':profile,'seed':42,'single_predictor':True,'forecast_averaging':False,'smoke':args.smoke,
            'Kaggle_uploaded':False,**meta,'sha256':{p.name:digest(p) for p in directory.iterdir() if p.name!='result.json'}}
    atomic_json(marker,result)
    if final and not args.smoke:
        args.quick.mkdir(parents=True,exist_ok=True);dest=args.quick/filename
        if dest.exists() and digest(dest)!=digest(directory/filename):raise ValueError('Different CSV already exists')
        shutil.copyfile(directory/filename,dest);atomic_json(args.quick/(filename+'.json'),result)
    print('CSV_READY' if final else 'VALID',profile['name'],date,meta.get('RMSLE'),flush=True)
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/course10_transfer_weights.json'))
    parser.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'))
    parser.add_argument('--reference',type=Path,default=Path('outputs/season-transfer'))
    parser.add_argument('--out',type=Path,default=Path('outputs/transfer-weights'))
    parser.add_argument('--quick',type=Path,default=Path('outputs/quick-variants-20261116'))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    assert cfg['seed']==42 and cfg['dataset']=='D0' and cfg['task']=='direct' and cfg['max_seminar']==10
    args.out.mkdir(parents=True,exist_ok=True)
    source=['scripts/run_transfer_weights.py','src/spend_course10/transfer_weights.py','src/spend_baselines/suite_models.py','scripts/run_season_transfer.py']
    frozen(args.out/'study.json',{'config':cfg,'device':args.device,'threads':args.threads,'smoke':args.smoke,
           'libraries':{n:version(n) for n in ['numpy','pandas','scikit-learn','torch']},'source_sha256':{n:digest(n) for n in source}})
    with threadpool_limits(limits=args.threads):
        fcfg={**cfg,'profiles':[{'name':'raw','base':'F0','season':True,'history':'full'}]}
        objects,matrices=prepare(args,fcfg);x,names=matrices['raw']
        # The domain classifier deliberately excludes calendar, prior-year,
        # and incomplete long-history fields that identify the data era.
        domain_names=[n for n in names[:15] if not n.startswith('month_')]+[
            f'past_log_{what}_{k:02d}' for what in ('spend','days') for k in range(1,7)]
        domain=x[:,[names.index(n) for n in domain_names]]
        if args.smoke:
            keep=sorted(objects.customer_id.unique(),key=lambda v:hashlib.sha256(str(v).encode()).hexdigest())[:64]
            ix=np.flatnonzero(objects.customer_id.isin(keep).to_numpy())
            objects=objects.iloc[ix].reset_index(drop=True);x=x[ix];domain=domain[ix]
        atomic_json(args.out/'domain_columns.json',domain_names)
        profiles=[{'name':'reference'}]+cfg['profiles'];frames={}
        if not args.smoke:
            chunks=[]
            for date in cfg['validation_dates']:
                directory=args.reference/'jobs/reference'/date;r=json.loads((directory/'result.json').read_text())
                assert digest(directory/'predictions.csv.gz')==r['sha256']['predictions.csv.gz']
                chunks.append(pd.read_csv(directory/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}))
            frames['reference']=pd.concat(chunks,ignore_index=True)
        for profile in cfg['profiles']:
            frames[profile['name']]=pd.concat([execute(args,cfg,objects,x,names,domain,profile,date) for date in cfg['validation_dates']],ignore_index=True)
            date=objects.loc[objects.is_test,'as_of_date'].min()
            execute(args,cfg,objects,x,names,domain,profile,date,True)
        if not args.smoke:report(args,objects,profiles,frames)


if __name__=='__main__':main()
