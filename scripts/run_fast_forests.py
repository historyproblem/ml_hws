"""Three fixed random-forest variants; emit a checked CSV as each fit completes."""
import argparse,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import atomic_json,digest
from run_monthly_history import prepare,frozen,fit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/course10_fast_forests.json'))
    parser.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'));parser.add_argument('--out',type=Path,default=Path('outputs/fast-forest-variants'))
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);cfg=json.loads(args.config.read_text())
    assert cfg['dataset']=='D0' and cfg['task']=='direct' and cfg['seed']==42 and cfg['max_seminar']==10
    with threadpool_limits(limits=args.threads):
        objects,matrices=prepare(args,cfg)
        frozen(args.out/'model_manifest.json',{'config':cfg,'source_sha256':{'scripts/run_fast_forests.py':digest('scripts/run_fast_forests.py')}})
        summaries=[]
        for p in cfg['profiles']:
            x,names=matrices[p['name']];settings={**cfg,'forest':p['params']};frames=[]
            for date in cfg['confirmation_dates']:
                d=args.out/'jobs'/p['name']/date;d.mkdir(parents=True,exist_ok=True);path=d/'predictions.csv.gz'
                if (d/'result.json').exists():
                    r=json.loads((d/'result.json').read_text());assert digest(path)==r['sha256']
                    frames.append(pd.read_csv(path,dtype={'Id':str,'customer_id':str}));continue
                vi=np.flatnonzero((objects.official&objects.as_of_date.eq(date)).to_numpy())
                z,_,_,meta=fit(objects,x,p,settings,date,vi,args.threads)
                a=objects.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
                a['date'],a['z'],a['target_z']=date,z,np.log1p(objects.iloc[vi].spend_30d.to_numpy(float));a.to_csv(path,index=False);frames.append(a)
                score=float(np.sqrt(np.mean((a.z-a.target_z)**2)))
                atomic_json(d/'result.json',{'profile':p,'date':date,'RMSLE':score,**meta,'sha256':digest(path)})
                print(p['name'],date,f'{score:.6f}',flush=True)
            a=pd.concat(frames,ignore_index=True);score=float(np.sqrt(np.mean((a.z-a.target_z)**2)))
            d=args.out/'refit'/p['name'];d.mkdir(parents=True,exist_ok=True)
            if (d/'result.json').exists():
                r=json.loads((d/'result.json').read_text())
                for n,s in r['sha256'].items():assert digest(d/n)==s
            else:
                vi=np.flatnonzero(objects.is_test.to_numpy());date=objects.iloc[vi].as_of_date.min();z,pre,model,meta=fit(objects,x,p,settings,date,vi,args.threads)
                joblib.dump({'preprocessor':pre,'model':model},d/'model.joblib',compress=1)
                sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str});write_submission(objects.iloc[vi],z,d/f"submission_{p['name']}.csv",sample_ids=sample.Id)
                np.savez_compressed(d/'inputs_predictions.npz',Id=objects.iloc[vi].Id.to_numpy(str),customer_id=objects.iloc[vi].customer_id.to_numpy(str),X=x[vi],z=z)
                atomic_json(d/'result.json',{'profile':p,'feature_names':names,'one_predictor':True,'task':'direct','dataset':'D0','seed':42,**meta,'RMSLE_SepOct':score,'Kaggle_uploaded':False,'sha256':{q.name:digest(q) for q in d.iterdir() if q.name!='result.json'}})
            summaries.append({'profile':p['name'],'RMSLE_SepOct':score,'features':len(names)})
            pd.DataFrame(summaries).to_csv(args.out/'summary.csv',index=False);print('READY',d,flush=True)


if __name__=='__main__':main()
