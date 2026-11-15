"""Emit checked CSVs for four fixed single-MLP variants; no notebook generation."""
import argparse,hashlib,json,platform,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import atomic_json,digest
from spend_baselines.suite_models import Neural
from spend_behavior.data import positive_lines
from spend_course10.item_season_features import CURRENT_COLUMNS,ITEM_SEASON_COLUMNS,item_season_features
from run_monthly_history import prepare,frozen
from run_monthly_mlp import MonthlyWorker,DenseCache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/course10_fast_mlp_variants.json'))
    parser.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'));parser.add_argument('--out',type=Path,default=Path('outputs/fast-mlp-variants'))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--max-minutes',type=float,default=10);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);args.out_dir=args.out
    cfg=json.loads(args.config.read_text());assert cfg['dataset']=='D0' and cfg['task']=='direct' and cfg['seed']==42 and cfg['max_seminar']==10
    with threadpool_limits(limits=args.threads):
        objects,matrices=prepare(args,cfg)
        itemcache=args.out/'items.npz'
        if itemcache.exists():
            r=json.loads((args.out/'items.ready.json').read_text());assert digest(itemcache)==r['sha256']
            with np.load(itemcache,allow_pickle=False) as a:
                assert np.array_equal(a['Id'],objects.Id.to_numpy(str));items=a['X']
        else:
            lines,start=positive_lines(args.data);items=item_season_features(objects,lines,start)
            np.savez_compressed(itemcache,Id=objects.Id.to_numpy(str),X=items);atomic_json(args.out/'items.ready.json',{'sha256':digest(itemcache)})
        for p in cfg['profiles']:
            if p['input']=='monthly_items':
                x,names=matrices[p['name']];cols=[ITEM_SEASON_COLUMNS.index(n) for n in CURRENT_COLUMNS]
                matrices[p['name']]=(np.c_[x,items[:,cols]],names+CURRENT_COLUMNS)
        if args.smoke:
            keep=sorted(objects.customer_id.unique(),key=lambda s:hashlib.sha256(str(s).encode()).hexdigest())[:64]
            ix=np.flatnonzero(objects.customer_id.isin(keep).to_numpy());objects=objects.iloc[ix].reset_index(drop=True)
            matrices={n:(x[ix],cols) for n,(x,cols) in matrices.items()}
        identity={'config':cfg,'device':args.device,'smoke':args.smoke,'python':platform.python_version(),
            'libraries':{n:version(n) for n in ['numpy','pandas','scikit-learn','scipy','torch']},
            'source_sha256':{n:digest(n) for n in ['scripts/run_fast_mlp_variants.py','src/spend_baselines/suite_models.py','scripts/tune_course10_fast.py','src/spend_course10/item_season_features.py']}}
        frozen(args.out/'model_manifest.json',identity)
        deadline=time.monotonic()+60*args.max_minutes;worker=MonthlyWorker(args,cfg,objects,matrices,deadline)
        summary=[]
        for p in cfg['profiles']:
            name=p['name'];x,names=matrices[name];worker.cache=DenseCache(x)
            for date in cfg['confirmation_dates']:
                done=worker.execute((name,date,42))
                if done.get('time_limit'):raise TimeoutError('Time budget ended between jobs; completed CSVs preserved')
            frames=[pd.read_csv(worker.path(name,d,42)/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}) for d in cfg['confirmation_dates']]
            a=pd.concat(frames,ignore_index=True);score=float(np.sqrt(np.mean((a.z-a.target_z)**2)))
            d=args.out/'refit'/name;d.mkdir(parents=True,exist_ok=True);marker=d/'result.json'
            settings=worker.settings(p)
            if marker.exists():
                result=json.loads(marker.read_text());assert result['settings']==settings
                for filename,sha in result['sha256'].items():assert digest(d/filename)==sha
            else:
                vi=np.flatnonzero(objects.is_test.to_numpy());cutoff=objects.iloc[vi].as_of_date.min()
                tr=worker.indices(cutoff);inner=objects.loc[objects.official&objects.target_end.le(cutoff),'as_of_date'].max();ti,iv=worker.indices(inner),worker.indices(inner,True)
                selector=Neural('mlp','direct',settings,42,args.device,args.threads).fit(x[ti],objects.iloc[ti].spend_30d.to_numpy(float),np.ones(len(ti)),validation=(x[iv],objects.iloc[iv].spend_30d.to_numpy(float)))
                epochs=selector.best_epoch;pd.DataFrame(selector.history).to_csv(d/'selection.csv',index=False);del selector
                model=Neural('mlp','direct',settings,42,args.device,args.threads).fit(x[tr],objects.iloc[tr].spend_30d.to_numpy(float),np.ones(len(tr)),epochs=epochs)
                assert model.network.output.out_features==1
                z=model.predict(x[vi]);model.save(d/'model.pt');pd.DataFrame(model.history).to_csv(d/'refit.csv',index=False)
                sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str})
                if args.smoke:sample=sample[sample.Id.isin(objects.iloc[vi].Id)]
                filename='smoke_predictions.csv' if args.smoke else f'submission_{name}_{args.device}.csv'
                write_submission(objects.iloc[vi],z,d/filename,sample_ids=sample.Id)
                np.savez_compressed(d/'inputs_predictions.npz',Id=objects.iloc[vi].Id.to_numpy(str),customer_id=objects.iloc[vi].customer_id.to_numpy(str),X=x[vi],z=z)
                result={'profile':p,'settings':settings,'seed':42,'one_predictor':True,'task':'direct','dataset':'D0','smoke':args.smoke,'upload_ready':not args.smoke,
                    'n_train':len(tr),'max_target_end':str(objects.iloc[tr].target_end.max()),'inner_date':str(inner),'selected_epoch':epochs,'feature_names':names,
                    'RMSLE_SepOct':score,'Kaggle_uploaded':False,'sha256':{q.name:digest(q) for q in d.iterdir() if q.name!='result.json'}}
                atomic_json(marker,result)
            summary.append({'profile':name,'RMSLE_SepOct':score,'epoch':result['selected_epoch'],'features':len(names),'device':args.device,'smoke':args.smoke})
            pd.DataFrame(summary).to_csv(args.out/'summary.csv',index=False)
            print('READY',name,'RMSLE',f'{score:.6f}',d,flush=True)


if __name__=='__main__':main()
