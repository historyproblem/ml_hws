"""Three plain MLP settings on proven monthly features; one final seed, no averaging."""
import argparse,copy,hashlib,json,platform,tarfile,time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import atomic_json,digest
from spend_baselines.suite_models import Neural
from spend_behavior.report import comparison
from run_monthly_history import frozen,prepare,fit
from tune_course10_fast import Worker


class DenseCache:
    def __init__(self,x):self.x=x
    def matrix(self,representation,indices):return self.x[indices],None


class MonthlyWorker(Worker):
    def __init__(self,args,cfg,objects,matrices,deadline):
        self.args,self.cfg,self.objects,self.deadline=args,cfg,objects,deadline
        self.active=np.ones(len(objects),bool)
        self.profiles={p['name']:p for p in cfg['profiles'] if p['model']=='mlp'}
        self.cache=DenseCache(matrices[next(iter(self.profiles))][0])


def frame(args,name,dates,seeds=(42,),control=False):
    paths=([args.out/'controls'/name/d/'predictions.csv.gz' for d in dates] if control else
           [args.out/'jobs'/name/d/str(s)/'predictions.csv.gz' for d in dates for s in seeds])
    return pd.concat([pd.read_csv(p,dtype={'Id':str,'customer_id':str}) for p in paths],ignore_index=True)


def controls(args,cfg,objects,matrices,date):
    for p in cfg['profiles']:
        if p['model']=='mlp':continue
        d=args.out/'controls'/p['name']/date;d.mkdir(parents=True,exist_ok=True)
        marker=d/'result.json'
        if marker.exists():
            r=json.loads(marker.read_text());assert digest(d/'predictions.csv.gz')==r['sha256'];continue
        vi=np.flatnonzero((objects.official&objects.as_of_date.eq(date)).to_numpy())
        z,_,_,meta=fit(objects,matrices[p['name']][0],p,cfg,date,vi,args.threads)
        a=objects.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
        a['date'],a['z'],a['target_z']=date,z,np.log1p(objects.iloc[vi].spend_30d.to_numpy(float))
        a.to_csv(d/'predictions.csv.gz',index=False)
        score=float(np.sqrt(np.mean((a.z-a.target_z)**2)))
        atomic_json(marker,{'profile':p,'date':date,'RMSLE':score,**meta,'sha256':digest(d/'predictions.csv.gz')})
        print(p['name'],date,f'{score:.6f}',flush=True)


def export(root):
    archive=root/'monthly-mlp-results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        candidates=[p for p in root.rglob('*') if p.is_file() and p!=archive and p.name!='runner.lock']
        for p in sorted(candidates):tar.add(p,arcname=str(p.relative_to(root)))
    print('EXPORT',archive,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/course10_monthly_mlp.json'))
    p.add_argument('--cache',type=Path,default=Path('outputs/v100-focused/cache'))
    p.add_argument('--data',type=Path,default=Path('data/raw'))
    p.add_argument('--out',type=Path,default=Path('outputs/monthly-mlp-v100'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-minutes',type=float,default=45);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);args.out_dir=args.out
    cfg=json.loads(args.config.read_text())
    if any(cfg[k]!=v for k,v in {'seed':42,'dataset':'D0','task':'direct','max_seminar':10,'final_seed':42,'diagnostic_seeds':[42,123,2026]}.items()):raise ValueError('Not the fixed single-model recipe')
    import torch
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu or run on V100')
    if args.max_minutes<=0:raise ValueError('Positive budget required')
    with threadpool_limits(limits=args.threads):
        objects,matrices=prepare(args,cfg)
        if args.smoke:
            keep=sorted(objects.customer_id.unique(),key=lambda x:hashlib.sha256(str(x).encode()).hexdigest())[:64]
            ix=np.flatnonzero(objects.customer_id.isin(keep).to_numpy());objects=objects.iloc[ix].reset_index(drop=True)
            matrices={n:(x[ix],names) for n,(x,names) in matrices.items()}
            cfg=copy.deepcopy(cfg);cfg['forest']['n_estimators']=3
        identity={'config':cfg,'device':args.device,'threads':args.threads,'smoke':args.smoke,'python':platform.python_version(),
            'libraries':{n:version(n) for n in ['numpy','pandas','scikit-learn','scipy','torch']},
            'source_sha256':{n:digest(n) for n in ['scripts/run_monthly_mlp.py','scripts/tune_course10_fast.py','src/spend_baselines/suite_models.py']},
            'feature_names':matrices['mlp_base'][1],'forecast_averaging':False,'final_seed':42}
        frozen(args.out/'model_manifest.json',identity)
        deadline=time.monotonic()+60*args.max_minutes;worker=MonthlyWorker(args,cfg,objects,matrices,deadline)
        def execute(name,date,seed):
            r=worker.execute((name,date,seed))
            if r.get('time_limit'):raise TimeoutError('Budget ended between complete jobs; partial artifacts retained')
        try:
            for d in cfg['tuning_dates']:
                controls(args,cfg,objects,matrices,d)
                for n in worker.profiles:execute(n,d,42)
            ranking=[]
            for n in worker.profiles:
                a=frame(args,n,cfg['tuning_dates']);ranking.append({'profile':n,'RMSLE':float(np.sqrt(np.mean((a.z-a.target_z)**2)))})
            ranking=sorted(ranking,key=lambda r:(r['RMSLE'],r['profile']));winner=ranking[0]['profile']
            frozen(args.out/'selection.json',{'ranking':ranking,'winner':winner,'uses_confirmation':False,'final_seed':42})
            for s in [123,2026]:
                for d in cfg['tuning_dates']:execute(winner,d,s)
            for d in cfg['confirmation_dates']:
                controls(args,cfg,objects,matrices,d)
                for s in cfg['diagnostic_seeds']:execute(winner,d,s)
            rows=[]
            for seed in cfg['diagnostic_seeds']:
                a=frame(args,winner,cfg['confirmation_dates'],[seed])
                for control in ['ridge_control','forest_control']:
                    b=frame(args,control,cfg['confirmation_dates'],control=True).set_index('Id').loc[a.Id]
                    a['reference_z']=b.z.to_numpy();rows.append({'profile':winner,'seed':seed,'reference':control,**comparison(a,20 if args.smoke else 2000)})
            pd.DataFrame(rows).to_csv(args.out/'confirmation_scores.csv',index=False)
            # Fixed seed42 final predictor. Diagnostic forecasts are never averaged.
            cutoff=objects.loc[objects.is_test,'as_of_date'].min();tr=worker.indices(cutoff)
            inner=objects.loc[objects.official&objects.target_end.le(cutoff),'as_of_date'].max()
            ti,iv=worker.indices(inner),worker.indices(inner,True);vi=np.flatnonzero(objects.is_test.to_numpy())
            settings=worker.settings(worker.profiles[winner]);x=worker.cache.x
            d=args.out/'refit';d.mkdir(exist_ok=True)
            if (d/'result.json').exists():
                previous=json.loads((d/'result.json').read_text())
                if previous['profile']!=winner or previous['settings']!=settings or previous['smoke']!=args.smoke:
                    raise ValueError('Completed refit definition differs')
                for name,sha in previous['sha256'].items():
                    if digest(d/name)!=sha:raise ValueError('Corrupt completed final refit')
                atomic_json(args.out/'status.json',{'complete':True,'smoke':args.smoke,'forecast_averaging':False,'Kaggle_uploaded':False})
                print('Verified completed final predictor; no repeated fit',flush=True)
                return
            selector=Neural('mlp','direct',settings,42,args.device,args.threads).fit(x[ti],objects.iloc[ti].spend_30d.to_numpy(float),np.ones(len(ti)),validation=(x[iv],objects.iloc[iv].spend_30d.to_numpy(float)))
            epoch=selector.best_epoch
            pd.DataFrame(selector.history).to_csv(d/'selection.csv',index=False);del selector
            model=Neural('mlp','direct',settings,42,args.device,args.threads).fit(x[tr],objects.iloc[tr].spend_30d.to_numpy(float),np.ones(len(tr)),epochs=epoch)
            assert model.network.output.out_features==1
            z=model.predict(x[vi]);model.save(d/'model.pt');pd.DataFrame(model.history).to_csv(d/'refit.csv',index=False)
            sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str})
            if args.smoke:sample=sample[sample.Id.isin(objects.iloc[vi].Id)]
            filename='smoke_predictions.csv' if args.smoke else 'submission_monthly_mlp_seed42.csv'
            write_submission(objects.iloc[vi],z,d/filename,sample_ids=sample.Id)
            np.savez_compressed(d/'inputs_predictions.npz',Id=objects.iloc[vi].Id.to_numpy(str),customer_id=objects.iloc[vi].customer_id.to_numpy(str),X=x[vi],z=z)
            atomic_json(d/'result.json',{'profile':winner,'settings':settings,'seed':42,'one_predictor':True,'task':'direct',
                'smoke':args.smoke,'upload_ready':not args.smoke,'n_train':len(tr),'max_target_end':str(objects.iloc[tr].target_end.max()),
                'inner_date':str(inner),'selected_epoch':epoch,'feature_names':identity['feature_names'],'Kaggle_uploaded':False,
                'sha256':{p.name:digest(p) for p in d.iterdir() if p.name!='result.json'}})
            atomic_json(args.out/'status.json',{'complete':True,'smoke':args.smoke,'forecast_averaging':False,'Kaggle_uploaded':False})
            print(pd.DataFrame(rows).to_string(index=False),flush=True)
        finally:export(args.out)


if __name__=='__main__':main()
