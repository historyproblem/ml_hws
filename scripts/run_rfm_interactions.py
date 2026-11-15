"""One Ridge on fixed interpretable RFM/calendar interactions, with temporal selection."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import atomic_json,digest
from spend_behavior.report import comparison
from spend_course10.rfm_interactions import rfm_interactions
from run_monthly_history import prepare,frozen
from run_item_season import fit,trial,frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/course10_rfm_interactions.json'))
    p.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    p.add_argument('--data',type=Path,default=Path('data/raw'));p.add_argument('--out',type=Path,default=Path('outputs/course10-rfm-interactions'))
    p.add_argument('--reference',type=Path,default=Path('outputs/course10-item-season'))
    p.add_argument('--threads',type=int,default=2);p.add_argument('--refit')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);cfg=json.loads(args.config.read_text())
    assert cfg['dataset']=='D0' and cfg['task']=='direct' and cfg['max_seminar']==10 and cfg['seed']==42
    with threadpool_limits(limits=args.threads):
        objects,matrices=prepare(args,cfg)
        frozen(args.out/'interaction_manifest.json',{'source_sha256':{n:digest(n) for n in ['scripts/run_rfm_interactions.py','src/spend_course10/rfm_interactions.py','scripts/run_item_season.py']}})
        profiles={p['name']:p for p in cfg['profiles']}
        for p in profiles.values():
            x,names=matrices[p['name']];extra,extra_names=rfm_interactions(x,names,p['interaction']);matrices[p['name']]=(np.c_[x,extra],names+extra_names)
        if args.refit:
            p=profiles[args.refit];x,names=matrices[args.refit];vi=np.flatnonzero(objects.is_test.to_numpy());date=objects.iloc[vi].as_of_date.min()
            z,pre,model,meta=fit(objects,x,p,cfg,date,vi,args.threads)
            d=args.out/'refit'/args.refit;d.mkdir(parents=True,exist_ok=True)
            joblib.dump({'preprocessor':pre,'model':model},d/'model.joblib',compress=3)
            sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str});write_submission(objects.iloc[vi],z,d/'submission.csv',sample_ids=sample.Id)
            np.savez_compressed(d/'inputs_predictions.npz',Id=objects.iloc[vi].Id.to_numpy(str),customer_id=objects.iloc[vi].customer_id.to_numpy(str),X=x[vi],z=z)
            atomic_json(d/'result.json',{'profile':p,'feature_names':names,'single_predictor':True,'seed':42,'task':'direct','dataset':'D0','max_seminar':10,**meta,
                'Kaggle_uploaded':False,'sha256':{q.name:digest(q) for q in d.iterdir() if q.name!='result.json'}})
            print('READY',d/'submission.csv',flush=True);return
        for p in profiles.values():
            for date in cfg['tuning_dates']:trial(args,cfg,objects,matrices,p,date)
        ranking=[]
        for name,p in profiles.items():
            a=frame(args,name,cfg['tuning_dates']);ranking.append({'profile':name,'control':p['control'],'RMSLE':float(np.sqrt(np.mean((a.z-a.target_z)**2)))})
        ranking=sorted(ranking,key=lambda r:(r['RMSLE'],r['profile']));chosen=['ridge64_control']+[r['profile'] for r in ranking if not r['control']][:2]
        frozen(args.out/'selection.json',{'ranking':ranking,'confirmation_profiles':chosen,'uses_confirmation':False})
        pd.DataFrame(ranking).to_csv(args.out/'early_scores.csv',index=False)
        for name in chosen:
            for date in cfg['confirmation_dates']:trial(args,cfg,objects,matrices,profiles[name],date)
        rows=[];dates=[]
        for name in chosen:
            a=frame(args,name,cfg['confirmation_dates']);b=frame(args,'ridge64_control',cfg['confirmation_dates']).set_index('Id').loc[a.Id];a['reference_z']=b.z.to_numpy()
            rows.append({'profile':name,**comparison(a,2000)})
            for date,g in a.groupby('date'):dates.append({'profile':name,'date':date,'RMSLE':float(np.sqrt(np.mean((g.z-g.target_z)**2)))})
        pd.DataFrame(rows).to_csv(args.out/'confirmation_scores.csv',index=False);pd.DataFrame(dates).to_csv(args.out/'per_date_scores.csv',index=False)
        print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
