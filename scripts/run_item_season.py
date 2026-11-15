"""Past global product seasonality transferred through current customer preferences."""
import argparse,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import atomic_json,digest
from spend_baselines.suite_models import Preprocessor
from spend_behavior.data import positive_lines
from spend_behavior.report import comparison
from spend_course10.item_season_features import ITEM_SEASON_COLUMNS,CURRENT_COLUMNS,YEAR_COLUMNS,item_season_features
from run_monthly_history import prepare,frozen


def fit(objects,x,p,cfg,date,vi,threads):
    tr=np.flatnonzero((objects.official&objects.target_end.le(pd.Timestamp(date))).to_numpy())
    assert objects.iloc[tr].target_end.max()<=pd.Timestamp(date)
    start=time.monotonic();pre=Preprocessor().fit(x[tr],np.ones(len(tr)))
    dtype=np.float64 if p.get('precision')=='float64' else np.float32
    a,b=pre.transform(x[tr]).astype(dtype),pre.transform(x[vi]).astype(dtype)
    y=np.log1p(objects.iloc[tr].spend_30d.to_numpy(float))
    model=Ridge(alpha=p['alpha']) if p['model']=='ridge' else RandomForestRegressor(**cfg['forest'],random_state=42,n_jobs=threads)
    model.fit(a,y,sample_weight=np.ones(len(tr),dtype=dtype));elapsed=time.monotonic()-start;start=time.monotonic()
    z=np.maximum(np.asarray(model.predict(b),dtype=float),0);assert np.isfinite(z).all()
    return z,pre,model,{'n_train':len(tr),'max_target_end':str(objects.iloc[tr].target_end.max()),
                       'fit_seconds':elapsed,'inference_seconds':time.monotonic()-start,'precision':str(np.dtype(dtype))}


def frame(args,name,dates):
    return pd.concat([pd.read_csv(args.out/'jobs'/name/d/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}) for d in dates],ignore_index=True)


def trial(args,cfg,objects,matrices,p,date):
    d=args.out/'jobs'/p['name']/date;d.mkdir(parents=True,exist_ok=True);marker=d/'result.json';path=d/'predictions.csv.gz'
    if marker.exists():
        r=json.loads(marker.read_text());assert r['profile']==p and digest(path)==r['sha256'];return
    vi=np.flatnonzero((objects.official&objects.as_of_date.eq(date)).to_numpy())
    if p.get('reuse'):
        original=args.reference/'jobs'/p['reuse']/date;r=json.loads((original/'result.json').read_text())
        assert digest(original/'predictions.csv.gz')==r['sha256']
        a=pd.read_csv(original/'predictions.csv.gz',dtype={'Id':str,'customer_id':str})
        meta={k:r[k] for k in ['n_train','max_target_end','fit_seconds','inference_seconds']};meta['reused']=True
    else:
        z,_,_,meta=fit(objects,matrices[p['name']][0],p,cfg,date,vi,args.threads)
        a=objects.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True);a['z']=z
        a['target_z']=np.log1p(objects.iloc[vi].spend_30d.to_numpy(float))
    assert a.Id.is_unique and set(a.Id)==set(objects.iloc[vi].Id.astype(str))
    a['date'],a['profile']=date,p['name'];a.to_csv(path,index=False)
    score=float(np.sqrt(np.mean((a.z-a.target_z)**2)))
    atomic_json(marker,{'profile':p,'date':date,'RMSLE':score,**meta,'sha256':digest(path)})
    print(p['name'],date,f'{score:.6f}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/course10_item_season.json'))
    parser.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'))
    parser.add_argument('--out',type=Path,default=Path('outputs/course10-item-season'))
    parser.add_argument('--reference',type=Path,default=Path('outputs/course10-monthly-history'))
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--refit')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);cfg=json.loads(args.config.read_text())
    assert cfg['dataset']=='D0' and cfg['task']=='direct' and cfg['seed']==42 and cfg['max_seminar']==10
    with threadpool_limits(limits=args.threads):
        objects,matrices=prepare(args,cfg)
        identity={'source_sha256':{n:digest(n) for n in ['scripts/run_item_season.py','src/spend_course10/item_season_features.py','src/spend_embeddings/data.py','src/spend_behavior/data.py']},
                  'input_sha256':json.loads((args.out/'manifest.json').read_text())['input_sha256'],
                  'prior':cfg['item_prior'],'max_items':cfg['item_max_items']}
        frozen(args.out/'item_manifest.json',identity)
        ready=args.out/'items.ready.json';payload=args.out/'item_features.npz'
        if ready.exists():
            r=json.loads(ready.read_text());assert r['identity']==identity and digest(payload)==r['sha256']
            with np.load(payload,allow_pickle=False) as a:
                assert np.array_equal(a['Id'],objects.Id.to_numpy(str));items=a['X']
        else:
            lines,start=positive_lines(args.data)
            items=item_season_features(objects,lines,start,prior=cfg['item_prior'],max_items=cfg['item_max_items'])
            np.savez_compressed(payload,Id=objects.Id.to_numpy(str),X=items)
            atomic_json(ready,{'identity':identity,'sha256':digest(payload),'columns':ITEM_SEASON_COLUMNS})
        profiles={p['name']:p for p in cfg['profiles']}
        for p in profiles.values():
            if p['items']=='none':continue
            names=ITEM_SEASON_COLUMNS if p['items']=='all' else CURRENT_COLUMNS if p['items']=='current' else YEAR_COLUMNS
            x,original_names=matrices[p['name']];cols=[ITEM_SEASON_COLUMNS.index(n) for n in names]
            matrices[p['name']]=(np.c_[x,items[:,cols]],original_names+names)
        if args.refit:
            p=profiles[args.refit];x,names=matrices[args.refit];vi=np.flatnonzero(objects.is_test.to_numpy());date=objects.iloc[vi].as_of_date.min()
            z,pre,model,meta=fit(objects,x,p,cfg,date,vi,args.threads)
            d=args.out/'refit'/args.refit;d.mkdir(parents=True,exist_ok=True)
            joblib.dump({'preprocessor':pre,'model':model},d/'model.joblib',compress=3)
            sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str});write_submission(objects.iloc[vi],z,d/'submission.csv',sample_ids=sample.Id)
            np.savez_compressed(d/'inputs_predictions.npz',Id=objects.iloc[vi].Id.to_numpy(str),customer_id=objects.iloc[vi].customer_id.to_numpy(str),X=x[vi],z=z)
            atomic_json(d/'result.json',{'profile':p,'feature_names':names,'single_predictor':True,'seed':42,'task':'direct',
                'dataset':'D0','max_seminar':10,**meta,'Kaggle_uploaded':False,'sha256':{q.name:digest(q) for q in d.iterdir() if q.name!='result.json'}})
            print('READY',d/'submission.csv',flush=True);return
        for p in profiles.values():
            for date in cfg['tuning_dates']:trial(args,cfg,objects,matrices,p,date)
        ranking=[]
        for name,p in profiles.items():
            a=frame(args,name,cfg['tuning_dates']);ranking.append({'profile':name,'control':p['control'],'RMSLE':float(np.sqrt(np.mean((a.z-a.target_z)**2)))})
        ranking=sorted(ranking,key=lambda r:(r['RMSLE'],r['profile']))
        chosen=[p['name'] for p in profiles.values() if p['control']]+[r['profile'] for r in ranking if not r['control']][:2]
        frozen(args.out/'selection.json',{'ranking':ranking,'confirmation_profiles':chosen,'uses_confirmation':False})
        pd.DataFrame(ranking).to_csv(args.out/'early_scores.csv',index=False)
        for name in chosen:
            for date in cfg['confirmation_dates']:trial(args,cfg,objects,matrices,profiles[name],date)
        rows=[];dates=[]
        for name in chosen:
            a=frame(args,name,cfg['confirmation_dates'])
            for ref in ['measured_ridge32','ridge64_control','forest_history_control']:
                b=frame(args,ref,cfg['confirmation_dates']).set_index('Id').loc[a.Id];a['reference_z']=b.z.to_numpy()
                rows.append({'profile':name,'reference':ref,**comparison(a,2000)})
            for date,g in a.groupby('date'):dates.append({'profile':name,'date':date,'RMSLE':float(np.sqrt(np.mean((g.z-g.target_z)**2)))})
        pd.DataFrame(rows).to_csv(args.out/'confirmation_scores.csv',index=False);pd.DataFrame(dates).to_csv(args.out/'per_date_scores.csv',index=False)
        print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
