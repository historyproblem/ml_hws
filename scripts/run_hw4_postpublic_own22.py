"""One new, fixed own22 CatBoost view; exploratory post-Public OLD-only gate."""
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from importlib import metadata
import hashlib,json,os,sys,time

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_hw4_final_research as primary
from hw4_inspections.classical_models import fit_model,predict_model
from hw4_inspections.next_round import evaluate_candidates

OLD=('2023-07-01','2023-10-01','2024-01-01')
SEEDS=(42,17,2026)
META=['row_id','site_id','forecast_period','target']
RAW={'train.csv':'30b6acd8c9f2eec4259a775b098017934ba5497cb37689eb00b639433472f0d9',
 'test_student.csv':'1014585cdbdb45e6b1ce7bcb5723700aa025826975a6c5c6910ed5ce39f6f643',
 'inspection_history.csv':'9637ec0279d9cf521795268a0f44099d7d51a795b5155a0d78405a770966afc7',
 'sample_submission.csv':'2d9b5084b510302cd027d46e0c5f9b8e986d48f417d9a582a8b5ec2b3c953b42'}
SPEC=dict(family='catboost',feature_set='own22',add_site=False,recency=False,max_iterations=500,tune_iterations=False,
 params=dict(depth=5,learning_rate=.04,l2_leaf_reg=10.,boosting_type='Plain',bootstrap_type='MVS',subsample=.8,rsm=1.,border_count=254,grow_policy='SymmetricTree'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def csv(path):return pd.read_csv(path,float_precision='round_trip')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def align(frame,meta):
 if not frame.row_id.is_unique or set(frame.row_id)!=set(meta.row_id) or len(frame)!=len(meta):raise ValueError('Wrong OOF identities')
 frame=frame.set_index('row_id').loc[meta.row_id].reset_index()
 pd.testing.assert_frame_equal(meta,frame[META],check_dtype=False,check_exact=True)
 return frame


def run(a):
 if not 1<=a.threads<=2:raise ValueError('CPU thread budget is two')
 a.out.mkdir(parents=True,exist_ok=True)
 start=time.perf_counter();proto=read(a.mac_protocol);cachemeta=read(a.feature_cache.with_name('features-meta.json'))
 raw={n:sha(a.raw/n) for n in RAW}
 if raw!=RAW or raw!=cachemeta['raw_sha256'] or raw!=proto['input_sha256'] or sha(a.feature_cache)!=cachemeta['cache_sha256']:raise ValueError('Raw/cache provenance differs')
 if sha(a.baseline_cv)!='0096456a838264be615680be4eead0ed8b449a4182d9f20acd519937e7cc1f7c':raise ValueError('Exact baseline OOF required')
 for name,h in cachemeta['builder_sha256'].items():
  if sha(ROOT/'src/hw4_inspections'/name)!=h:raise ValueError('Feature cache builder differs')
 paths=[Path(__file__),ROOT/'src/hw4_inspections/classical_models.py',ROOT/'src/hw4_inspections/classical_features.py',ROOT/'src/hw4_inspections/next_round.py',ROOT/'src/hw4_inspections/classical_analysis.py',ROOT/'scripts/run_hw4_final_research.py',a.mac_protocol,a.baseline_cv,a.provider_oof,a.provider_test]
 plan=dict(format='hw4-post-public-own22-1',spec=SPEC,columns=proto['fields'],categories=proto['categories'],seeds=list(SEEDS),old_quarters=list(OLD),threads=a.threads,
  cache_sha256=sha(a.feature_cache),cache_metadata_sha256=sha(a.feature_cache.with_name('features-meta.json')),raw_sha256=raw,source_and_parent_sha256={str(p.resolve()):sha(p) for p in paths},
  versions={n:metadata.version(n) for n in ['catboost','joblib','numpy','pandas','scikit-learn']},
  hypothesis='Exact own22 context from MacICL removes individual codes, all cross-site context groups and absolute calendar; this view has not been evaluated with CB. Tree/seed changes are not searched.',
  formulas=['cb_own22_raw','icl50_cb_own22_50_probability'],
  gate='Mean AP strictly above raw ICL94; >=2 nonnegative OLD quarter deltas vs ICL94; worst >=-.005. Same gate for screen/confirmation.',
  schedule='Seed42 OLD3 first; only if either fixed formula passes, seeds17+2026 OLD6 and confirm fixed mean3. Only if confirmed gate passes, full3. Maximum12 real fits; fixed500, no early stopping.',
  final='At most one confirmed gate-passing formula, greatest mean AP with raw first tie-break. No outer target enters fitting; every OLD train period is strictly before query.',
  bootstrap='0 at screen;2000 common-site paired draws(seed20261122) only for confirmed candidates passing AP gate',
  interpretation='Exploratory after Public11=.34642/12=.34242; reused OLD development, no fresh-holdout or unbiased confirmation claim.',
  new_recent_AP=0,uploads=0)
 plan['signature']=digest(plan);frozen=a.out/'freeze.json'
 if frozen.exists():
  if read(frozen)!=plan:raise ValueError('Changed research freeze')
  if (a.out/'complete.json').exists():
   done=read(a.out/'complete.json')
   if any(sha(a.out/n)!=h for n,h in done['artifacts'].items()):raise ValueError('Changed completed artifacts')
   return done
 else:
  if any(a.out.iterdir()):raise ValueError('New experiment requires empty out')
  write(frozen,plan)
 # Freeze precedes both training and scores.
 train,test,sample,views,cats=primary.validate_cache(joblib.load(a.feature_cache),a)
 columns=proto['fields'];categories=proto['categories'];frame=views['base'][columns]
 if len(columns)!=22 or len(categories)!=4 or set(columns)&{'target','row_id','site_id','history_before','forecast_period','forecast_year','cutoff_gap_days'}:raise ValueError('Wrong own22 schema')
 meta=pd.concat([train.loc[train.forecast_period.eq(q),META] for q in OLD],ignore_index=True)
 baseline=align(csv(a.baseline_cv),meta);provider=align(csv(a.provider_oof),meta)
 context=baseline.prediction.to_numpy(dtype=np.float64);icl=provider.tabicl_v2_compact94__raw.to_numpy(dtype=np.float64)
 if not np.array_equal(context,provider.context04):raise ValueError('OOF anchor differs')
 fits=[]
 def job(seed,quarter):
  fit_idx=np.arange(len(train)) if quarter=='full' else np.flatnonzero(train.forecast_period.lt(quarter))
  query=np.arange(len(train),len(frame)) if quarter=='full' else np.flatnonzero(train.forecast_period.eq(quarter))
  ids=test.row_id if quarter=='full' else train.iloc[query].row_id
  identity=dict(signature=plan['signature'],seed=seed,quarter=quarter,train_rows=len(fit_idx),query_rows=len(query),columns=columns,categories=categories,
   train_ids_sha256=digest(train.iloc[fit_idx].row_id.tolist()),query_ids_sha256=digest(ids.tolist()))
  path=a.out/'jobs'/f'{quarter}_s{seed}'
  if path.exists():
   saved=read(path/'complete.json')
   if saved['identity']!=identity or any(sha(path/n)!=h for n,h in saved['artifacts'].items()):raise ValueError('Changed saved model job')
   p=np.load(path/'prediction.npy',allow_pickle=False)
   fits.append(dict(seed=seed,quarter=quarter,reused=True));return p
  model=fit_model(SPEC,frame.iloc[fit_idx],train.iloc[fit_idx].target,categories,seed=seed,threads=a.threads)
  p=predict_model(model,frame.iloc[query])
  path.parent.mkdir(exist_ok=True)
  with TemporaryDirectory(dir=path.parent,prefix='.pending-') as temporary:
   d=Path(temporary)/'job';d.mkdir();joblib.dump(model,d/'model.joblib',compress=1)
   if not np.array_equal(p,predict_model(joblib.load(d/'model.joblib'),frame.iloc[query])):raise ValueError('Reload prediction changed')
   if p.dtype!=np.float64 or p.shape!=(len(query),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Bad prediction')
   np.save(d/'prediction.npy',p,allow_pickle=False)
   write(d/'complete.json',dict(identity=identity,reload_exact=True,tree_count=model['model'].tree_count_,artifacts={n:sha(d/n) for n in ['model.joblib','prediction.npy']}))
   os.rename(d,path)
  fits.append(dict(seed=seed,quarter=quarter,reused=False));print(json.dumps(dict(event='fit_complete',seed=seed,quarter=quarter,rows=len(fit_idx))),flush=True)
  return p
 def assess(stage,own):
  values=dict(context04=context,icl94_raw=icl,cb_own22_raw=own,icl50_cb_own22_50_probability=.5*icl+.5*own)
  result=evaluate_candidates(meta,values,reference='icl94_raw',bootstrap=0,seed=20261122)
  summary=result['summary'].set_index('model')
  eligible=[n for n in plan['formulas'] if summary.loc[n,'mean_ap']>summary.loc['icl94_raw','mean_ap'] and summary.loc[n,'nonnegative_quarters']>=2 and summary.loc[n,'worst_quarter_delta']>=-.005]
  if stage=='confirmed' and eligible:result=evaluate_candidates(meta,values,reference='icl94_raw',bootstrap=2000,seed=20261122)
  result['summary'].to_csv(a.out/f'{stage}-summary.csv',index=False);result['quarter_scores'].to_csv(a.out/f'{stage}-quarter-metrics.csv',index=False)
  pd.concat([meta,pd.DataFrame(values)],axis=1).to_csv(a.out/f'{stage}-old-oof.csv',index=False)
  return eligible,summary
 seed42=np.concatenate([job(42,q) for q in OLD]);eligible,summary=assess('screen',seed42)
 exports=[]
 if eligible:
  arrays=[seed42,*[np.concatenate([job(seed,q) for q in OLD]) for seed in SEEDS[1:]]]
  eligible,summary=assess('confirmed',np.mean(arrays,axis=0))
  if eligible:
   winner=min(eligible,key=lambda n:(-summary.loc[n,'mean_ap'],plan['formulas'].index(n)))
   p=np.mean([job(seed,'full') for seed in SEEDS],axis=0)
   if winner!='cb_own22_raw':
    parent=csv(a.provider_test).set_index('row_id').loc[test.row_id].prediction.to_numpy(dtype=np.float64);p=.5*p+.5*parent
   output=primary.validate_submission(test,sample,p,2246);path=a.out/f'submission_{winner}.csv';output.to_csv(path,index=False)
   pd.testing.assert_frame_equal(output,csv(path),check_exact=True);exports=[dict(file=path.name,model=winner,sha256=sha(path),rows=2246,status='READY_NOT_SUBMITTED',public_ap=None)]
 done=dict(status='COMPLETE',signature=plan['signature'],fits=fits,real_fit_count=len(fits),eligible=eligible,files=exports,elapsed_seconds=time.perf_counter()-start,new_recent_AP=0,uploads=0,
  artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()})
 write(a.out/'complete.json',done);print(json.dumps(done,indent=2));return done


if __name__=='__main__':
 p=ArgumentParser(description=__doc__)
 for n in ['raw','out','feature-cache','baseline-cv','provider-oof','provider-test','mac-protocol']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--threads',type=int,default=2);p.set_defaults(smoke=False)
 run(p.parse_args())
