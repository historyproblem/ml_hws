"""Two fixed nonlinear Ridge maps; alpha selected on June-August only."""
import argparse
import hashlib
import json
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import atomic_json,digest
from spend_course10.ramp_ridge import RampDesign,RampRidge
from run_monthly_history import prepare,frozen
from run_season_transfer import indices,report


def alpha_name(alpha):return f'alpha{alpha:g}'


def directory(args,profile,alpha,date,final):
    if final:return args.out/'refit'/profile['name']
    return args.out/'jobs'/profile['name']/alpha_name(alpha)/date


def fit_trials(args,cfg,objects,x,names,profile,date,alphas,final=False):
    pending=[]
    for alpha in alphas:
        d=directory(args,profile,alpha,date,final);marker=d/'result.json'
        if marker.exists():
            r=json.loads(marker.read_text());assert r['profile']==profile and r['alpha']==alpha
            for filename,sha in r['sha256'].items():assert digest(d/filename)==sha
        else:pending.append(alpha)
    if not pending:return
    tr=indices(objects,date)
    vi=np.flatnonzero(objects.is_test.to_numpy()) if final else indices(objects,date,True)
    assert objects.iloc[tr].target_end.max()<=pd.Timestamp(date)
    assert not objects.iloc[vi].Id.isin(objects.iloc[tr].Id).any()
    start=time.monotonic()
    design=RampDesign(profile['interactions']).fit(x[tr],names)
    train=design.transform(x[tr]);shared_design_seconds=time.monotonic()-start
    ztarget=np.log1p(objects.iloc[tr].spend_30d.to_numpy(float))
    for alpha in pending:
        d=directory(args,profile,alpha,date,final);d.mkdir(parents=True,exist_ok=True)
        start=time.monotonic();model=RampRidge(design,alpha).fit_design(train,ztarget)
        ridge_seconds=time.monotonic()-start
        start=time.monotonic();z=model.predict(x[vi]);inference_seconds=time.monotonic()-start
        joblib.dump(model,d/'model.joblib',compress=3)
        np.testing.assert_array_equal(z,joblib.load(d/'model.joblib').predict(x[vi]))
        metadata={'profile':profile,'alpha':alpha,'n_train':len(tr),'max_target_end':str(objects.iloc[tr].target_end.max()),
                  'n_raw_features':len(names),'n_regression_features':design.n_output,'shared_design_seconds':shared_design_seconds,
                  'ridge_fit_seconds':ridge_seconds,'fit_seconds':shared_design_seconds+ridge_seconds,'inference_seconds':inference_seconds,
                  'precision':'float64','checkpoint_forward_exact':True,'Kaggle_uploaded':False,'smoke':args.smoke}
        if final:
            sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str})
            if args.smoke:sample=sample[sample.Id.isin(objects.iloc[vi].Id)]
            filename=('smoke_' if args.smoke else 'submission_')+profile['name']+'.csv'
            write_submission(objects.iloc[vi],z,d/filename,sample_ids=sample.Id)
            exported=pd.read_csv(d/filename,dtype={'Id':str})
            assert exported.Id.tolist()==sample.Id.tolist() and exported.columns.tolist()==sample.columns.tolist()
            assert np.isfinite(exported.spend_30d).all() and exported.spend_30d.ge(0).all()
            ordered=pd.Series(z,index=objects.iloc[vi].Id).loc[sample.Id].to_numpy()
            np.testing.assert_allclose(np.log1p(exported.spend_30d),ordered,rtol=0,atol=2e-12)
            np.savez_compressed(d/'inputs_predictions.npz',Id=objects.iloc[vi].Id.to_numpy(str),X=x[vi],z=z)
            metadata.update(n_test=len(vi),feature_names=names,mean_prediction_z=float(z.mean()),upload_ready=not args.smoke)
        else:
            a=objects.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
            a['date']=date;a['profile']=profile['name'];a['alpha']=alpha;a['z']=z
            a['target_z']=np.log1p(objects.iloc[vi].spend_30d.to_numpy(float))
            a.to_csv(d/'predictions.csv.gz',index=False)
            metadata['RMSLE']=float(np.sqrt(np.mean((a.z-a.target_z)**2)))
        metadata['sha256']={p.name:digest(p) for p in d.iterdir() if p.name!='result.json'}
        atomic_json(d/'result.json',metadata)
        if final and not args.smoke:
            args.quick.mkdir(parents=True,exist_ok=True);dest=args.quick/filename
            if dest.exists() and digest(dest)!=digest(d/filename):raise ValueError('Different submission already exists')
            shutil.copyfile(d/filename,dest);atomic_json(args.quick/(filename+'.json'),metadata)
        print('CSV_READY' if final else 'VALID',profile['name'],alpha,date,metadata.get('RMSLE'),flush=True)


def frame(args,profile,alpha,dates):
    return pd.concat([pd.read_csv(directory(args,profile,alpha,d,False)/'predictions.csv.gz',dtype={'Id':str,'customer_id':str})
                      for d in dates],ignore_index=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/course10_nonlinear_ridge.json'))
    parser.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'))
    parser.add_argument('--reference',type=Path,default=Path('outputs/season-transfer'))
    parser.add_argument('--out',type=Path,default=Path('outputs/nonlinear-ridge'))
    parser.add_argument('--quick',type=Path,default=Path('outputs/quick-variants-20261116'))
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--refit-only',action='store_true');parser.add_argument('--profile');parser.add_argument('--alpha',type=float)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    assert cfg['dataset']=='D0' and cfg['task']=='direct' and cfg['max_seminar']==10 and cfg['seed']==42
    if args.refit_only and (args.profile not in {p['name'] for p in cfg['profiles']} or args.alpha not in cfg['alphas']):
        raise ValueError('Refit requires one predefined profile and its earlier selected alpha')
    args.out.mkdir(parents=True,exist_ok=True)
    source=['scripts/run_nonlinear_ridge.py','src/spend_course10/ramp_ridge.py','src/spend_baselines/suite_models.py','scripts/run_monthly_history.py']
    frozen(args.out/'study.json',{'config':cfg,'smoke':args.smoke,'threads':args.threads,
           'libraries':{n:version(n) for n in ['numpy','pandas','scikit-learn','scipy']},
           'source_sha256':{n:digest(n) for n in source}})
    with threadpool_limits(limits=args.threads):
        fcfg={**cfg,'profiles':[{'name':'raw','base':'F0','season':True,'history':'full'}]}
        objects,matrices=prepare(args,fcfg);x,names=matrices['raw']
        if args.smoke:
            keep=sorted(objects.customer_id.unique(),key=lambda v:hashlib.sha256(str(v).encode()).hexdigest())[:64]
            ix=np.flatnonzero(objects.customer_id.isin(keep).to_numpy());objects=objects.iloc[ix].reset_index(drop=True);x=x[ix]
        cutoff=objects.loc[objects.is_test,'as_of_date'].min()
        if args.refit_only:
            p=next(p for p in cfg['profiles'] if p['name']==args.profile)
            fit_trials(args,cfg,objects,x,names,p,cutoff,[args.alpha],True);return
        ranking=[];chosen={}
        for p in cfg['profiles']:
            for date in cfg['tuning_dates']:fit_trials(args,cfg,objects,x,names,p,date,cfg['alphas'])
            rows=[]
            for alpha in cfg['alphas']:
                a=frame(args,p,alpha,cfg['tuning_dates'])
                rows.append({'profile':p['name'],'alpha':alpha,'RMSLE':float(np.sqrt(np.mean((a.z-a.target_z)**2)))})
            rows=sorted(rows,key=lambda r:(r['RMSLE'],r['alpha']))
            ranking.extend(rows);chosen[p['name']]=rows[0]['alpha']
        frozen(args.out/'selection.json',{'ranking':ranking,'selected_alphas':chosen,'uses_confirmation':False})
        pd.DataFrame(ranking).to_csv(args.out/'early_ranking.csv',index=False)
        frames={}
        for p in cfg['profiles']:
            alpha=chosen[p['name']]
            for date in cfg['confirmation_dates']:fit_trials(args,cfg,objects,x,names,p,date,[alpha])
            frames[p['name']]=frame(args,p,alpha,cfg['tuning_dates']+cfg['confirmation_dates'])
            fit_trials(args,cfg,objects,x,names,p,cutoff,[alpha],True)
        if not args.smoke:
            chunks=[]
            for date in cfg['tuning_dates']+cfg['confirmation_dates']:
                d=args.reference/'jobs/reference'/date;r=json.loads((d/'result.json').read_text())
                assert digest(d/'predictions.csv.gz')==r['sha256']['predictions.csv.gz']
                chunks.append(pd.read_csv(d/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}))
            frames['reference']=pd.concat(chunks,ignore_index=True)
            report(args,objects,[{'name':'reference'}]+cfg['profiles'],frames)


if __name__=='__main__':main()
