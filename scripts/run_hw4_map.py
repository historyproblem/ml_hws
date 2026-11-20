"""MAP-oriented CatBoost screen on old dev only, with frozen75/25 blends."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.metadata,json,platform,sys,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hw4_inspections.features import build_features
from hw4_inspections.extensions import build_context_features
from hw4_inspections.classical_models import SPECS,fit_model,predict_model,choose_iterations
from hw4_inspections.map_ranking import fit_ranker,predict_ranker,choose_ranker_iterations,PARAMS,LOSS
from hw4_inspections.next_round import rank_scores,evaluate_candidates
from hw4_inspections.delivery import validate_submission

QUARTERS=['2023-07-01','2023-10-01','2024-01-01']
RAW=['train.csv','test_student.csv','inspection_history.csv','sample_submission.csv']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)

def main():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument('--raw',type=Path,default=ROOT/'data')
 a.add_argument('--out',type=Path,default=ROOT/'analysis/hw4-map-20261121')
 a.add_argument('--baseline-cv',type=Path,required=True)
 a.add_argument('--context-test',type=Path,required=True)
 a.add_argument('--threads',type=int,default=4)
 a.add_argument('--max-iterations',type=int,default=1000)
 a.add_argument('--bootstrap',type=int,default=2000)
 a.add_argument('--resume',action='store_true')
 a.add_argument('--smoke',action='store_true')
 args=a.parse_args()
 if min(args.threads,args.max_iterations,args.bootstrap)<1:raise ValueError('Positive limits required')
 args.out.mkdir(parents=True,exist_ok=True)
 started=time.monotonic()
 def log(event,**details):print(json.dumps(dict(event=event,elapsed_seconds=round(time.monotonic()-started,3),**details)),flush=True)
 sources=[Path(__file__),*(ROOT/'src/hw4_inspections'/n for n in ['features.py','extensions.py','classical_models.py','map_ranking.py','next_round.py','delivery.py','classical_analysis.py'])]
 config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
 provenance={'config':config,'input_sha256':{n:sha(args.raw/n) for n in RAW},'baseline_sha256':sha(args.baseline_cv),'test_anchor_sha256':sha(args.context_test),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'versions':{n:importlib.metadata.version(n) for n in ['catboost','numpy','pandas','scikit-learn','joblib']},'machine':platform.machine(),'recipe':{'features':346,'loss':LOSS,'params':PARAMS,'seed':42,'max_iterations':5 if args.smoke else args.max_iterations,'grid':50,'final_iterations':'median of three selected earlier-inner counts','blend':'0.75 context percentile rank +0.25 MAP percentile rank','gate':'mean delta>0 and nonnegative in>=2/3 old quarters, independently per export','Q2Q3_used_for_new_selection':False}}
 signature=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest();proto=args.out/'protocol.json'
 if proto.exists():
  old=json.loads(proto.read_text())
  if not args.resume or old['signature']!=signature:raise ValueError('New --out or unchanged --resume required')
 write_json(proto,dict(provenance,signature=signature,status='running',kaggle_uploads=0,new_closed_quarter_metrics=False))
 train=pd.read_csv(args.raw/'train.csv');test=pd.read_csv(args.raw/'test_student.csv');sample=pd.read_csv(args.raw/'sample_submission.csv')
 if args.smoke:train=train.groupby('forecast_period',group_keys=False).head(32).reset_index(drop=True)
 rows=pd.concat([train.drop(columns='target'),test],ignore_index=True);history=pd.read_csv(args.raw/'inspection_history.csv')
 log('features_start');x,cats=build_features(rows,history);x=pd.concat([x,build_context_features(rows,history)],axis=1);del history
 if len(x.columns)!=346:raise ValueError('Frozen full feature view changed')
 meta=train.loc[train.forecast_period.isin(QUARTERS),['row_id','site_id','forecast_period','target']].reset_index(drop=True)
 base=pd.read_csv(args.baseline_cv,float_precision='round_trip')
 if base.row_id.duplicated().any():raise ValueError('Duplicate baseline IDs')
 base=base.set_index('row_id').reindex(meta.row_id)
 for c in ['site_id','forecast_period','target']:
  if not np.array_equal(base[c].to_numpy(),meta[c].to_numpy()):raise ValueError('Baseline official metadata mismatch')
 base_p=base.prediction.to_numpy(dtype=float)
 if not np.isfinite(base_p).all() or ((base_p<0)|(base_p>1)).any():raise ValueError('Bad baseline')
 control=copy.deepcopy(SPECS['cb_reference']);control.update(max_iterations=5 if args.smoke else args.max_iterations,tune_iterations=True)
 predictions=[];records=[];chosen={'map':[],'logloss_matched':[]};reused=0
 for method in chosen:
  for q in QUARTERS:
   fit_idx=np.flatnonzero(train.forecast_period.lt(q));valid_idx=np.flatnonzero(train.forecast_period.eq(q))
   innerq=train.iloc[fit_idx].forecast_period.max();inner_fit=np.flatnonzero(train.forecast_period.lt(innerq));inner_eval=np.flatnonzero(train.forecast_period.eq(innerq))
   job=args.out/f'{method}_{q}';job.mkdir(exist_ok=True);mark=job/'complete.json'
   if args.resume and mark.exists():
    rec=json.loads(mark.read_text())
    if rec['signature']!=signature or any(sha(job/n)!=h for n,h in rec['artifacts'].items()):raise ValueError('Changed completed MAP job')
    p=np.load(job/'prediction.npy',allow_pickle=False);reused+=1;log('job_reused',method=method,quarter=q)
   else:
    log('fit_start',method=method,quarter=q)
    with threadpool_limits(limits=args.threads):
     if method=='map':
      n,curve=choose_ranker_iterations(x.iloc[inner_fit],train.iloc[inner_fit].target,train.iloc[inner_fit].forecast_period,x.iloc[inner_eval],train.iloc[inner_eval].target,cats,max_iterations=control['max_iterations'],seed=42,threads=args.threads)
      model=fit_ranker(x.iloc[fit_idx],train.iloc[fit_idx].target,train.iloc[fit_idx].forecast_period,cats,seed=42,threads=args.threads,iterations=n)
      p=predict_ranker(model,x.iloc[valid_idx])
     else:
      n,curve=choose_iterations(control,x.iloc[inner_fit],train.iloc[inner_fit].target,x.iloc[inner_eval],train.iloc[inner_eval].target,cats,42,args.threads)
      model=fit_model(control,x.iloc[fit_idx],train.iloc[fit_idx].target,cats,42,args.threads,iterations=n)
      p=predict_model(model,x.iloc[valid_idx])
     joblib.dump(model,job/'model.joblib');loaded=joblib.load(job/'model.joblib')
     check=predict_ranker(loaded,x.iloc[valid_idx]) if method=='map' else predict_model(loaded,x.iloc[valid_idx])
    if not np.array_equal(p,check):raise AssertionError('MAP/control reload mismatch')
    np.save(job/'prediction.npy',p,allow_pickle=False);write_json(job/'inner_curve.json',curve)
    rec={'signature':signature,'method':method,'quarter':q,'inner_quarter':innerq,'iterations':int(n),'ap':float(average_precision_score(train.iloc[valid_idx].target,p)),'loaded_forward_exact':True,'artifacts':{n:sha(job/n) for n in ['model.joblib','prediction.npy','inner_curve.json']}}
    write_json(mark,rec);log('fit_complete',method=method,quarter=q,ap=rec['ap'],iterations=n)
   chosen[method].append(rec['iterations']);records.append(rec)
   predictions.append(train.iloc[valid_idx][['row_id','site_id','forecast_period','target']].assign(model=method,prediction=np.asarray(p,dtype=np.float64)))
 long=pd.concat(predictions,ignore_index=True);long.to_csv(args.out/'predictions.csv',index=False)
 score={name:long.loc[long.model.eq(name)].set_index('row_id').loc[meta.row_id].prediction.to_numpy() for name in chosen}
 score['context_3seed']=base_p;blend=np.empty(len(meta))
 for q in QUARTERS:
  mask=meta.forecast_period.eq(q).to_numpy();blend[mask]=.75*rank_scores(base_p[mask])+.25*rank_scores(score['map'][mask])
 score['context_map_75_25']=blend
 report=evaluate_candidates(meta,score,bootstrap=args.bootstrap)
 for n in ['summary','quarter_scores']:report[n].to_csv(args.out/f'{n}.csv',index=False)
 write_json(args.out/'eligibility.json',{'gate':report['protocol'],'eligible':report['eligibility'],'full_map_refit_permitted':any(report['eligibility'][n] for n in ['map','context_map_75_25'])})
 print(report['summary'].to_string(index=False),flush=True)
 exports={};nfinal=int(np.median(chosen['map']))
 if any(report['eligibility'][n] for n in ['map','context_map_75_25']):
  log('full_refit_start',iterations=nfinal)
  model=fit_ranker(x.iloc[:len(train)],train.target,train.forecast_period,cats,seed=42,threads=args.threads,iterations=nfinal)
  p=predict_ranker(model,x.iloc[len(train):]);joblib.dump(model,args.out/'full_model.joblib');check=predict_ranker(joblib.load(args.out/'full_model.joblib'),x.iloc[len(train):])
  if not np.array_equal(p,check):raise AssertionError('Full MAP reload differs')
  anchor=pd.read_csv(args.context_test,float_precision='round_trip')
  if anchor.row_id.duplicated().any() or set(anchor.row_id)!=set(test.row_id):raise ValueError('Test anchor IDs differ')
  anchor_p=anchor.set_index('row_id').loc[test.row_id].target.to_numpy()
  candidate={'map':rank_scores(p),'context_map_75_25':.75*rank_scores(anchor_p)+.25*rank_scores(p)}
  for name,values in candidate.items():
   directory=args.out/('recommended' if report['eligibility'][name] else 'diagnostic');directory.mkdir(exist_ok=True)
   f=directory/f'submission_{name}.csv';validate_submission(test,sample,values,expected_rows=2246).to_csv(f,index=False)
   exports[name]={'file':str(f),'sha256':sha(f),'eligible':report['eligibility'][name]}
 write_json(proto,dict(provenance,signature=signature,status='complete',kaggle_uploads=0,new_closed_quarter_metrics=False,reused_jobs=reused,final_iterations=nfinal,exports=exports))
 log('complete',eligible=report['eligibility'],exports=exports,kaggle_uploads=0)

if __name__=='__main__':main()
