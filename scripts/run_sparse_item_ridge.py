"""Direct sparse SKU regression on proven monthly inputs, one early alpha choice."""
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
from scipy import sparse
from threadpoolctl import threadpool_limits
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import atomic_json,digest
from spend_behavior.data import positive_lines
from spend_course10.sparse_item_ridge import SparseItemDesign,SparseItemRidge
from run_monthly_history import prepare,frozen
from run_customer_svd import build_matrix
from run_season_transfer import indices,report


def path(args,alpha,date,final=False):
    return args.out/'refit' if final else args.out/'jobs'/f'alpha{alpha:g}'/date


def execute(args,cfg,o,x,hist,names,date,alphas,final=False):
    pending=[]
    for alpha in alphas:
        d=path(args,alpha,date,final);marker=d/'result.json'
        if marker.exists():
            r=json.loads(marker.read_text());assert r['alpha']==alpha
            for filename,sha in r['sha256'].items():assert digest(d/filename)==sha
        else:pending.append(alpha)
    if not pending:return
    tr=indices(o,date);vi=np.flatnonzero(o.is_test.to_numpy()) if final else indices(o,date,True)
    assert o.iloc[tr].target_end.max()<=pd.Timestamp(date)
    assert not o.iloc[vi].Id.isin(o.iloc[tr].Id).any()
    start=time.monotonic()
    design=SparseItemDesign(cfg['min_customers'],cfg['bag_scale']).fit(x[tr],hist[tr],o.iloc[tr].customer_id,names)
    a=design.transform(x[tr],hist[tr]);design_seconds=time.monotonic()-start
    y=np.log1p(o.iloc[tr].spend_30d.to_numpy(float))
    for alpha in pending:
        d=path(args,alpha,date,final);d.mkdir(parents=True,exist_ok=True)
        start=time.monotonic();m=SparseItemRidge(design,alpha).fit_design(a,y);ridge_seconds=time.monotonic()-start
        start=time.monotonic();z=m.predict(x[vi],hist[vi]);inference_seconds=time.monotonic()-start
        joblib.dump(m,d/'model.joblib',compress=3)
        np.testing.assert_array_equal(z,joblib.load(d/'model.joblib').predict(x[vi],hist[vi]))
        r={'alpha':alpha,'profile':cfg['profile'],'n_train':len(tr),'max_target_end':str(o.iloc[tr].target_end.max()),
           'n_items':len(design.columns),'n_design_features':a.shape[1],'iterations':m.estimator.n_iter_.tolist(),
           'shared_design_seconds':design_seconds,'ridge_fit_seconds':ridge_seconds,'fit_seconds':design_seconds+ridge_seconds,
           'inference_seconds':inference_seconds,'checkpoint_forward_exact':True,'smoke':args.smoke,'Kaggle_uploaded':False}
        if final:
            sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str})
            if args.smoke:sample=sample[sample.Id.isin(o.iloc[vi].Id)]
            filename=('smoke_' if args.smoke else 'submission_')+cfg['profile']+'.csv'
            write_submission(o.iloc[vi],z,d/filename,sample_ids=sample.Id)
            frame=pd.read_csv(d/filename,dtype={'Id':str})
            assert frame.Id.tolist()==sample.Id.tolist() and frame.columns.tolist()==sample.columns.tolist()
            assert np.isfinite(frame.spend_30d).all() and frame.spend_30d.ge(0).all()
            ordered=pd.Series(z,index=o.iloc[vi].Id).loc[sample.Id].to_numpy()
            np.testing.assert_allclose(np.log1p(frame.spend_30d),ordered,rtol=0,atol=2e-12)
            sparse.save_npz(d/'test_histories.npz',hist[vi])
            np.savez_compressed(d/'inputs_predictions.npz',Id=o.iloc[vi].Id.to_numpy(str),X=x[vi],z=z)
            r.update(n_test=len(vi),mean_prediction_z=float(z.mean()),upload_ready=not args.smoke)
        else:
            frame=o.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
            frame['date']=date;frame['profile']=cfg['profile'];frame['z']=z
            frame['target_z']=np.log1p(o.iloc[vi].spend_30d.to_numpy(float))
            frame.to_csv(d/'predictions.csv.gz',index=False)
            r['RMSLE']=float(np.sqrt(np.mean((frame.z-frame.target_z)**2)))
        r['sha256']={p.name:digest(p) for p in d.iterdir() if p.name!='result.json'}
        atomic_json(d/'result.json',r)
        if final and not args.smoke:
            args.quick.mkdir(parents=True,exist_ok=True);dest=args.quick/filename
            if dest.exists() and digest(dest)!=digest(d/filename):raise ValueError('Different CSV already exists')
            shutil.copyfile(d/filename,dest);atomic_json(args.quick/(filename+'.json'),r)
        print('CSV_READY' if final else 'VALID',alpha,date,r.get('RMSLE'),flush=True)


def frame(args,alpha,dates):
    return pd.concat([pd.read_csv(path(args,alpha,d)/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}) for d in dates],ignore_index=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/course10_sparse_item_ridge.json'))
    p.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    p.add_argument('--data',type=Path,default=Path('data/raw'));p.add_argument('--out',type=Path,default=Path('outputs/sparse-item-ridge'))
    p.add_argument('--reference',type=Path,default=Path('outputs/season-transfer'))
    p.add_argument('--quick',type=Path,default=Path('outputs/quick-variants-20261116'))
    p.add_argument('--threads',type=int,default=2);p.add_argument('--smoke',action='store_true')
    p.add_argument('--refit-only',action='store_true');p.add_argument('--alpha',type=float)
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    assert cfg['dataset']=='D0' and cfg['task']=='direct' and cfg['seed']==42 and cfg['max_seminar']==10
    if args.refit_only and args.alpha not in cfg['alphas']:raise ValueError('Use the earlier selected alpha for a final refit')
    args.out.mkdir(parents=True,exist_ok=True)
    source=['scripts/run_sparse_item_ridge.py','src/spend_course10/sparse_item_ridge.py','scripts/run_customer_svd.py','src/spend_behavior/data.py','src/spend_baselines/suite_models.py']
    frozen(args.out/'study.json',{'config':cfg,'smoke':args.smoke,'threads':args.threads,
           'libraries':{n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']},'source_sha256':{n:digest(n) for n in source}})
    with threadpool_limits(limits=args.threads):
        fcfg={**cfg,'profiles':[{'name':'raw','base':'F0','season':True,'history':'full'}]}
        o,matrices=prepare(args,fcfg);x=matrices['raw'][0]
        if args.smoke:
            keep=sorted(o.customer_id.unique(),key=lambda v:hashlib.sha256(str(v).encode()).hexdigest())[:64]
            ix=np.flatnonzero(o.customer_id.isin(keep).to_numpy());o=o.iloc[ix].reset_index(drop=True);x=x[ix]
        marker=args.out/'histories.ready.json'
        if marker.exists():
            r=json.loads(marker.read_text())
            for filename,sha in r['sha256'].items():assert digest(args.out/filename)==sha
            hist=sparse.load_npz(args.out/'histories.npz');names=json.loads((args.out/'item_names.json').read_text())
            assert np.array_equal(np.load(args.out/'history_Ids.npy'),o.Id.to_numpy(str))
        else:
            lines,_=positive_lines(args.data);lines=lines[lines.customer_id.isin(o.customer_id.unique())]
            hist,names=build_matrix(o,lines,window=cfg['window_days'],half_life=cfg['half_life_days'])
            sparse.save_npz(args.out/'histories.npz',hist)
            atomic_json(args.out/'item_names.json',names.tolist());np.save(args.out/'history_Ids.npy',o.Id.to_numpy(str))
            atomic_json(marker,{'rows':len(o),'items':len(names),'nnz':hist.nnz,'sha256':{n:digest(args.out/n) for n in ['histories.npz','item_names.json','history_Ids.npy']}})
        cutoff=o.loc[o.is_test,'as_of_date'].min()
        if args.refit_only:
            execute(args,cfg,o,x,hist,names,cutoff,[args.alpha],True);return
        for date in cfg['tuning_dates']:execute(args,cfg,o,x,hist,names,date,cfg['alphas'])
        ranking=[]
        for alpha in cfg['alphas']:
            a=frame(args,alpha,cfg['tuning_dates']);ranking.append({'alpha':alpha,'RMSLE':float(np.sqrt(np.mean((a.z-a.target_z)**2)))})
        ranking=sorted(ranking,key=lambda r:(r['RMSLE'],r['alpha']));chosen=ranking[0]['alpha']
        frozen(args.out/'selection.json',{'ranking':ranking,'alpha':chosen,'uses_confirmation':False})
        for date in cfg['confirmation_dates']:execute(args,cfg,o,x,hist,names,date,[chosen])
        execute(args,cfg,o,x,hist,names,cutoff,[chosen],True)
        if not args.smoke:
            frames={cfg['profile']:frame(args,chosen,cfg['tuning_dates']+cfg['confirmation_dates'])}
            reference=[]
            for date in cfg['tuning_dates']+cfg['confirmation_dates']:
                d=args.reference/'jobs/reference'/date;r=json.loads((d/'result.json').read_text())
                assert digest(d/'predictions.csv.gz')==r['sha256']['predictions.csv.gz']
                reference.append(pd.read_csv(d/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}))
            frames['reference']=pd.concat(reference,ignore_index=True)
            report(args,o,[{'name':'reference'},{'name':cfg['profile']}],frames)


if __name__=='__main__':main()
