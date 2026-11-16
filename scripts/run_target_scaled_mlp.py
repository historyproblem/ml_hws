"""One conditioning check on the measured best MLP; same inputs and settings."""
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
from spend_course10.target_scaled_mlp import TargetScaledMLP
from run_monthly_history import frozen,prepare
from run_season_transfer import indices,report


def execute(args,cfg,o,x,date,final=False):
    d=args.out/'refit' if final else args.out/'jobs'/date
    marker=d/'result.json'
    if marker.exists():
        r=json.loads(marker.read_text())
        for filename,sha in r['sha256'].items():assert digest(d/filename)==sha
        return None if final else pd.read_csv(d/'predictions.csv.gz',dtype={'Id':str,'customer_id':str})
    tr=indices(o,date);vi=np.flatnonzero(o.is_test.to_numpy()) if final else indices(o,date,True)
    assert o.iloc[tr].target_end.max()<=pd.Timestamp(date)
    inner=o.iloc[tr].as_of_date.max();ti,iv=indices(o,inner),indices(o,inner,True)
    assert o.iloc[iv].target_end.max()<=pd.Timestamp(date)
    settings=dict(cfg['neural_base'])
    if args.smoke:settings.update(max_epochs=2,patience=1)
    start=time.monotonic()
    selector=TargetScaledMLP('mlp','direct',settings,42,args.device,args.threads)
    selector.fit(x[ti],o.iloc[ti].spend_30d.to_numpy(float),np.ones(len(ti)),
                 validation=(x[iv],o.iloc[iv].spend_30d.to_numpy(float)))
    epochs=selector.best_epoch;d.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(selector.history).to_csv(d/'selection.csv',index=False)
    inner_mean,inner_scale=selector.target_mean,selector.target_scale;del selector
    model=TargetScaledMLP('mlp','direct',settings,42,args.device,args.threads)
    model.fit(x[tr],o.iloc[tr].spend_30d.to_numpy(float),np.ones(len(tr)),epochs=epochs)
    model.save(d/'model.pt');pd.DataFrame(model.history).to_csv(d/'refit.csv',index=False)
    fit_seconds=time.monotonic()-start
    start=time.monotonic();z=model.predict(x[vi]);inference_seconds=time.monotonic()-start
    np.testing.assert_array_equal(z,TargetScaledMLP.load(d/'model.pt',args.device,args.threads).predict(x[vi]))
    r={'profile':cfg['profile'],'settings':settings,'seed':42,'n_train':len(tr),'max_target_end':str(o.iloc[tr].target_end.max()),
       'inner_date':str(inner),'selected_epoch':epochs,'target_mean':model.target_mean,'target_scale':model.target_scale,
       'inner_target_mean':inner_mean,'inner_target_scale':inner_scale,'checkpoint_forward_exact':True,
       'fit_seconds':fit_seconds,'inference_seconds':inference_seconds,'smoke':args.smoke,'Kaggle_uploaded':False}
    if final:
        sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str})
        if args.smoke:sample=sample[sample.Id.isin(o.iloc[vi].Id)]
        filename=('smoke_' if args.smoke else 'submission_')+cfg['profile']+f'_{args.device}.csv'
        write_submission(o.iloc[vi],z,d/filename,sample_ids=sample.Id)
        exported=pd.read_csv(d/filename,dtype={'Id':str})
        assert exported.Id.tolist()==sample.Id.tolist() and exported.columns.tolist()==sample.columns.tolist()
        assert np.isfinite(exported.spend_30d).all() and exported.spend_30d.ge(0).all()
        expected=pd.Series(z,index=o.iloc[vi].Id).loc[sample.Id].to_numpy()
        np.testing.assert_allclose(np.log1p(exported.spend_30d),expected,rtol=0,atol=2e-12)
        np.savez_compressed(d/'inputs_predictions.npz',Id=o.iloc[vi].Id.to_numpy(str),X=x[vi],z=z)
        r.update(n_test=len(vi),mean_prediction_z=float(z.mean()),upload_ready=not args.smoke)
        frame=None
    else:
        frame=o.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
        frame['date']=date;frame['profile']=cfg['profile'];frame['z']=z
        frame['target_z']=np.log1p(o.iloc[vi].spend_30d.to_numpy(float))
        frame.to_csv(d/'predictions.csv.gz',index=False)
        r['RMSLE']=float(np.sqrt(np.mean((frame.z-frame.target_z)**2)))
    r['sha256']={p.name:digest(p) for p in d.iterdir() if p.name!='result.json'}
    atomic_json(marker,r)
    if final and not args.smoke:
        args.quick.mkdir(parents=True,exist_ok=True);dest=args.quick/filename
        if dest.exists() and digest(dest)!=digest(d/filename):raise ValueError('Different CSV already exists')
        shutil.copyfile(d/filename,dest);atomic_json(args.quick/(filename+'.json'),r)
    print('CSV_READY' if final else 'VALID',date,r.get('RMSLE'),flush=True)
    return frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/course10_target_scaled_mlp.json'))
    p.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    p.add_argument('--data',type=Path,default=Path('data/raw'));p.add_argument('--out',type=Path,default=Path('outputs/target-scaled-mlp'))
    p.add_argument('--reference',type=Path,default=Path('outputs/season-transfer'))
    p.add_argument('--quick',type=Path,default=Path('outputs/quick-variants-20261116'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--smoke',action='store_true');p.add_argument('--refit-only',action='store_true')
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    assert cfg['dataset']=='D0' and cfg['task']=='direct' and cfg['max_seminar']==10 and cfg['seed']==42
    args.out.mkdir(parents=True,exist_ok=True)
    source=['scripts/run_target_scaled_mlp.py','src/spend_course10/target_scaled_mlp.py','src/spend_baselines/suite_models.py']
    frozen(args.out/'study.json',{'config':cfg,'device':args.device,'threads':args.threads,'smoke':args.smoke,
           'libraries':{n:version(n) for n in ['numpy','pandas','torch']},'source_sha256':{n:digest(n) for n in source}})
    with threadpool_limits(limits=args.threads):
        fcfg={**cfg,'profiles':[{'name':'raw','base':'F0','season':True,'history':'full'}]}
        o,matrices=prepare(args,fcfg);x=matrices['raw'][0]
        if args.smoke:
            keep=sorted(o.customer_id.unique(),key=lambda v:hashlib.sha256(str(v).encode()).hexdigest())[:64]
            ix=np.flatnonzero(o.customer_id.isin(keep).to_numpy());o=o.iloc[ix].reset_index(drop=True);x=x[ix]
        cutoff=o.loc[o.is_test,'as_of_date'].min()
        if args.refit_only:execute(args,cfg,o,x,cutoff,True);return
        frames={cfg['profile']:pd.concat([execute(args,cfg,o,x,date) for date in cfg['validation_dates']],ignore_index=True)}
        execute(args,cfg,o,x,cutoff,True)
        if not args.smoke:
            reference=[]
            for date in cfg['validation_dates']:
                d=args.reference/'jobs/reference'/date;r=json.loads((d/'result.json').read_text())
                assert digest(d/'predictions.csv.gz')==r['sha256']['predictions.csv.gz']
                reference.append(pd.read_csv(d/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}))
            frames['reference']=pd.concat(reference,ignore_index=True)
            report(args,o,[{'name':'reference'},{'name':cfg['profile']}],frames)


if __name__=='__main__':main()
