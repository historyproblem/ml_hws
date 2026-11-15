"""Joint regularized customer indicators inside one Ridge; chronological labels only."""
import argparse,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import atomic_json,digest
from spend_baselines.suite_models import Preprocessor
from spend_behavior.report import comparison
from spend_course10.customer_ridge import CustomerRidge
from run_monthly_history import prepare,frozen


def frame(args,name,dates):
    return pd.concat([pd.read_csv(args.out/'jobs'/name/d/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}) for d in dates],ignore_index=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/course10_customer_ridge.json'))
    parser.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'));parser.add_argument('--out',type=Path,default=Path('outputs/customer-ridge-variants'))
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);cfg=json.loads(args.config.read_text())
    assert cfg['dataset']=='D0' and cfg['task']=='direct' and cfg['seed']==42 and cfg['max_seminar']==10
    with threadpool_limits(limits=args.threads):
        objects,matrices=prepare(args,cfg);x,names=matrices[cfg['profiles'][0]['name']]
        frozen(args.out/'model_manifest.json',{'config':cfg,'source_sha256':{n:digest(n) for n in ['scripts/run_customer_ridge.py','src/spend_course10/customer_ridge.py']},'feature_names':names,'one_estimator':True})
        def fit(p,date,vi):
            tr=np.flatnonzero((objects.official&objects.target_end.le(pd.Timestamp(date))).to_numpy());assert objects.iloc[tr].target_end.max()<=pd.Timestamp(date)
            start=time.monotonic();pre=Preprocessor().fit(x[tr],np.ones(len(tr)))
            model=CustomerRidge(p['alpha'],p['customer_alpha']).fit(pre.transform(x[tr]),np.log1p(objects.iloc[tr].spend_30d.to_numpy(float)),objects.iloc[tr].customer_id.to_numpy(str))
            seconds=time.monotonic()-start;z=model.predict(pre.transform(x[vi]),objects.iloc[vi].customer_id.to_numpy(str));assert np.isfinite(z).all()
            return z,pre,model,{'n_train':len(tr),'n_customers':len(model.effects_),'max_target_end':str(objects.iloc[tr].target_end.max()),'fit_seconds':seconds}
        def trial(p,date):
            d=args.out/'jobs'/p['name']/date;d.mkdir(parents=True,exist_ok=True);path=d/'predictions.csv.gz';marker=d/'result.json'
            if marker.exists():
                r=json.loads(marker.read_text());assert r['profile']==p and r['sha256']==digest(path);return
            vi=np.flatnonzero((objects.official&objects.as_of_date.eq(date)).to_numpy());z,_,_,meta=fit(p,date,vi)
            a=objects.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True);a['date'],a['z'],a['target_z']=date,z,np.log1p(objects.iloc[vi].spend_30d.to_numpy(float))
            a.to_csv(path,index=False);score=float(np.sqrt(np.mean((a.z-a.target_z)**2)))
            atomic_json(marker,{'profile':p,'date':date,'RMSLE':score,**meta,'sha256':digest(path)})
            print(p['name'],date,f'{score:.6f}',flush=True)
        for p in cfg['profiles']:
            for date in cfg['tuning_dates']:trial(p,date)
        ranking=[]
        for p in cfg['profiles']:
            a=frame(args,p['name'],cfg['tuning_dates']);ranking.append({'profile':p['name'],'RMSLE':float(np.sqrt(np.mean((a.z-a.target_z)**2)))})
        ranking=sorted(ranking,key=lambda r:(r['RMSLE'],r['profile']));chosen=[r['profile'] for r in ranking[:2]]
        frozen(args.out/'selection.json',{'ranking':ranking,'confirmation_profiles':chosen,'uses_confirmation':False});rows=[]
        for p in cfg['profiles']:
            if p['name'] not in chosen:continue
            for date in cfg['confirmation_dates']:trial(p,date)
            a=frame(args,p['name'],cfg['confirmation_dates'])
            ref=pd.concat([pd.read_csv(Path('outputs/course10-item-season/jobs/ridge64_control')/date/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}) for date in cfg['confirmation_dates']],ignore_index=True).set_index('Id').loc[a.Id]
            a['reference_z']=ref.z.to_numpy();rows.append({'profile':p['name'],**comparison(a,2000)})
            vi=np.flatnonzero(objects.is_test.to_numpy());date=objects.iloc[vi].as_of_date.min();d=args.out/'refit'/p['name'];d.mkdir(parents=True,exist_ok=True)
            z,pre,model,meta=fit(p,date,vi);joblib.dump({'preprocessor':pre,'model':model},d/'model.joblib',compress=3)
            filename=f"submission_{p['name']}.csv";sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str});write_submission(objects.iloc[vi],z,d/filename,sample_ids=sample.Id)
            np.savez_compressed(d/'inputs_predictions.npz',Id=objects.iloc[vi].Id.to_numpy(str),customer_id=objects.iloc[vi].customer_id.to_numpy(str),X=x[vi],z=z)
            atomic_json(d/'result.json',{'profile':p,'one_estimator':True,'task':'direct','dataset':'D0','precision':'float64',**meta,'Kaggle_uploaded':False,
                'sha256':{q.name:digest(q) for q in d.iterdir() if q.name!='result.json'}})
            print('READY',d/filename,flush=True)
        pd.DataFrame(rows).to_csv(args.out/'confirmation_scores.csv',index=False);print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
