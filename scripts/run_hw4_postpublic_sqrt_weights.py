"""Fixed mean-one sqrt class weighting, two seeds, OLD-only exploratory gate."""
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from importlib import metadata
from copy import deepcopy
import hashlib,json,os,sys,time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_hw4_postpublic_own22 as base
from hw4_inspections.classical_models import fit_model,predict_model
from hw4_inspections.next_round import evaluate_candidates

OLD=base.OLD;META=base.META;SEEDS=(42,17)
NAMES=('weighted_raw','icl50_weighted_native50','icl50_weighted_corrected50')
sha,read,csv,digest,write,align=base.sha,base.read,base.csv,base.digest,base.write,base.align


def weights(y):
 y=np.asarray(y,dtype=int);n1=int(y.sum());n0=len(y)-n1
 if not 0<n1<n0:raise ValueError('Expected a nonempty minority positive class')
 ratio=float(np.sqrt(n0/n1));raw=np.where(y==1,ratio,1.);normalizer=float(raw.mean());sample=raw/normalizer
 if abs(sample.mean()-1)>1e-12:raise ValueError('Weight mean differs from one')
 return sample,dict(negative=n0,positive=n1,positive_to_negative_ratio=ratio,normalizer=normalizer,weight_mean=float(sample.mean()))


def correct(p,ratio):
 return expit(logit(np.clip(np.asarray(p,dtype=np.float64),1e-15,1-1e-15))-np.log(ratio))


def run(a):
 if not 1<=a.threads<=2:raise ValueError('Two-thread CPU budget')
 a.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 raw={n:sha(a.raw/n) for n in base.RAW};cm=read(a.feature_cache.with_name('features-meta.json'))
 if raw!=base.RAW or raw!=cm['raw_sha256'] or sha(a.feature_cache)!=cm['cache_sha256']:raise ValueError('Different data/cache')
 for name,h in cm['builder_sha256'].items():
  if sha(ROOT/'src/hw4_inspections'/name)!=h:raise ValueError('Different cache builder')
 if sha(a.baseline_cv)!='0096456a838264be615680be4eead0ed8b449a4182d9f20acd519937e7cc1f7c' or sha(a.baseline_audit)!='6e302488907e7e06b348f8142fa9c0ee815db59003a8f23e3be9585ffabd8749':raise ValueError('Different original baseline evidence')
 spec=deepcopy(base.SPEC);spec['feature_set']='context346'
 files=[Path(__file__),Path(base.__file__),ROOT/'src/hw4_inspections/classical_models.py',ROOT/'src/hw4_inspections/next_round.py',ROOT/'src/hw4_inspections/classical_analysis.py',ROOT/'scripts/run_hw4_final_research.py',a.baseline_cv,a.baseline_audit,a.seed42_oof,a.other_seed_oof,a.provider_oof,a.provider_test]
 plan=dict(format='hw4-post-public-sqrt-weights-1',spec=spec,seeds=list(SEEDS),old_quarters=list(OLD),threads=a.threads,
  raw_sha256=raw,cache_sha256=sha(a.feature_cache),cache_metadata_sha256=sha(a.feature_cache.with_name('features-meta.json')),source_and_inputs_sha256={str(p.resolve()):sha(p) for p in files},
  versions={n:metadata.version(n) for n in ['catboost','joblib','numpy','pandas','scipy','scikit-learn']},
  weight_recipe='FIT-only ratio=sqrt(n_negative/n_positive). Per-row weights1 orratio, then divide by FIT mean; positive/negative multiplier ratio unchanged and total weight equals n. One fixed strength, no weight search.',
  correction='After averaging the two native positive probabilities, within each query quarter apply expit(logit(p)-log(FIT ratio)); ideal class-prior correction, not a guarantee of calibration. This monotonic map preserves within-quarter raw AP.',
  formulas=list(NAMES),formula_details={'weighted_raw':'mean2 native model probabilities; corrected raw has identical OLD AP','icl50_weighted_native50':'.5*ICL94+.5*mean2_native_weighted_CB','icl50_weighted_corrected50':'.5*ICL94+.5*prior_correct(mean2_native_weighted_CB)'},
  references=['context04','matched_unweighted_mean2','icl94_raw'],
  gate='Raw:mean above both unweighted C3/C2, >=2 nonnegative quarter deltas and worst>=-.005 vs both. Mixtures:mean above rawICL94, >=2 nonnegative and worst>=-.005 vsICL94.',
  schedule='Always six OLD fits(2seeds*3quarters). If any formula gate passes, two full fits only; max8. At most one exported formula by greatest OLD mean, ties raw/native50/corrected50 order.',
  bootstrap='2000 paired-site(seed20261122) only if a candidate passes AP gate; otherwise skip',
  primary_source={'url':'https://catboost.ai/docs/en/references/training-parameters/common#auto_class_weights','checked_utc':'2026-11-22','fact':'Class weights multiply object weights; SqrtBalanced is a supported fixed rule. We implement fit-only explicit mean-one sample weights for controlled overall loss/L2 scale.'},
  interpretation='Exploratory new objective after Public04/11/12; OLD heavily reused and nominal intervals. Public feedback motivated this test; no new recent-quarter score.',new_recent_AP=0,uploads=0)
 plan['signature']=digest(plan)
 if (a.out/'freeze.json').exists():
  if read(a.out/'freeze.json')!=plan:raise ValueError('Changed fixed-weight freeze')
  if (a.out/'complete.json').exists():
   done=read(a.out/'complete.json')
   if any(sha(a.out/n)!=h for n,h in {**done['artifacts'],**done['job_artifacts']}.items()):raise ValueError('Changed results/checkpoints')
   return done
 else:
  if any(a.out.iterdir()):raise ValueError('Use empty output directory')
  write(a.out/'freeze.json',plan)
 train,test,sample,views,cats=base.primary.validate_cache(joblib.load(a.feature_cache),a);frame=views['base']
 meta=pd.concat([train.loc[train.forecast_period.eq(q),META] for q in OLD],ignore_index=True)
 context=align(csv(a.baseline_cv),meta).prediction.to_numpy(dtype=np.float64)
 provider=align(csv(a.provider_oof),meta);icl=provider.tabicl_v2_compact94__raw.to_numpy(dtype=np.float64)
 if not np.array_equal(context,provider.context04):raise ValueError('Context anchor differs')
 legacy={};f42=csv(a.seed42_oof);others=csv(a.other_seed_oof)
 for seed,f in [(42,f42),(17,others),(2026,others)]:
  selected=f.loc[f.case.eq('plus_context')&f.seed.eq(seed)]
  legacy[seed]=align(selected,meta).prediction.to_numpy(dtype=np.float64)
 if not np.array_equal(np.mean([legacy[s] for s in [42,17,2026]],axis=0),context):raise ValueError('Audited parent mean3 does not reconstruct exactly')
 control=np.mean([legacy[s] for s in SEEDS],axis=0);fit_log=[];weight_stats={}
 def job(seed,q):
  fit_idx=np.arange(len(train)) if q=='full' else np.flatnonzero(train.forecast_period.lt(q))
  query_idx=np.arange(len(train),len(frame)) if q=='full' else np.flatnonzero(train.forecast_period.eq(q))
  sw,stats=weights(train.target.iloc[fit_idx]);weight_stats[q]=stats
  ids=test.row_id if q=='full' else train.row_id.iloc[query_idx]
  identity=dict(signature=plan['signature'],seed=seed,quarter=q,fit_rows=len(fit_idx),query_rows=len(query_idx),fit_ids_sha256=digest(train.row_id.iloc[fit_idx].tolist()),query_ids_sha256=digest(ids.tolist()),weights=stats)
  path=a.out/'jobs'/f'{q}_s{seed}'
  if path.exists():
   saved=read(path/'complete.json')
   if saved['identity']!=identity or any(sha(path/n)!=h for n,h in saved['artifacts'].items()):raise ValueError('Changed checkpoint')
   p=np.load(path/'prediction.npy',allow_pickle=False);fit_log.append(dict(seed=seed,quarter=q,reused=True));return p
  model=fit_model(spec,frame.iloc[fit_idx],train.target.iloc[fit_idx],cats,seed=seed,threads=a.threads,sample_weight=sw)
  p=predict_model(model,frame.iloc[query_idx]);path.parent.mkdir(exist_ok=True)
  with TemporaryDirectory(dir=path.parent,prefix='.pending-') as temporary:
   d=Path(temporary)/'job';d.mkdir();model['objective_weight_recipe']=stats;joblib.dump(model,d/'model.joblib',compress=1)
   if not np.array_equal(p,predict_model(joblib.load(d/'model.joblib'),frame.iloc[query_idx])):raise ValueError('Reload differs')
   if p.shape!=(len(query_idx),) or p.dtype!=np.float64 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid model probabilities')
   np.save(d/'prediction.npy',p,allow_pickle=False);write(d/'complete.json',dict(identity=identity,reload_exact=True,trees=model['model'].tree_count_,artifacts={n:sha(d/n) for n in ['model.joblib','prediction.npy']}));os.rename(d,path)
  fit_log.append(dict(seed=seed,quarter=q,reused=False));print(json.dumps(dict(event='fit_complete',seed=seed,quarter=q,positive_ratio=stats['positive_to_negative_ratio'])),flush=True);return p
 arrays=[np.concatenate([job(seed,q) for q in OLD]) for seed in SEEDS];native=np.mean(arrays,axis=0);corrected=np.empty_like(native)
 for q in OLD:
  mask=meta.forecast_period.eq(q).to_numpy();corrected[mask]=correct(native[mask],weight_stats[q]['positive_to_negative_ratio'])
  if not np.array_equal(rankdata(native[mask]),rankdata(corrected[mask])):raise ValueError('Correction changed raw within-quarter ranking')
 values=dict(context04=context,matched_unweighted_mean2=control,icl94_raw=icl,weighted_raw=native,
  icl50_weighted_native50=.5*icl+.5*native,icl50_weighted_corrected50=.5*icl+.5*corrected)
 reports={ref:evaluate_candidates(meta,values,reference=ref,bootstrap=0,seed=20261122) for ref in plan['references']}
 summary=reports['icl94_raw']['summary'].set_index('model');eligible=[]
 for name in NAMES:
  refs=['context04','matched_unweighted_mean2'] if name=='weighted_raw' else ['icl94_raw']
  if all((lambda row:row.mean_ap>summary.loc[ref,'mean_ap'] and row.nonnegative_quarters>=2 and row.worst_quarter_delta>=-.005)(reports[ref]['summary'].set_index('model').loc[name]) for ref in refs):eligible.append(name)
 if eligible:reports={ref:evaluate_candidates(meta,values,reference=ref,bootstrap=2000,seed=20261122) for ref in plan['references']}
 for ref,result in reports.items():
  result['summary'].to_csv(a.out/f'summary-vs-{ref}.csv',index=False);result['quarter_scores'].to_csv(a.out/f'quarter-metrics-vs-{ref}.csv',index=False)
 pd.concat([meta,pd.DataFrame(values).assign(weighted_prior_corrected=corrected)],axis=1).to_csv(a.out/'old-oof.csv',index=False)
 exports=[]
 if eligible:
  winner=min(eligible,key=lambda n:(-float(summary.loc[n,'mean_ap']),NAMES.index(n)));native_full=np.mean([job(s,'full') for s in SEEDS],axis=0)
  p=native_full
  if winner!='weighted_raw':
   icl_full=csv(a.provider_test).set_index('row_id').loc[test.row_id].prediction.to_numpy(dtype=np.float64)
   cb=native_full if winner=='icl50_weighted_native50' else correct(native_full,weight_stats['full']['positive_to_negative_ratio']);p=.5*icl_full+.5*cb
  output=base.primary.validate_submission(test,sample,p,2246);path=a.out/f'submission_{winner}.csv';output.to_csv(path,index=False);pd.testing.assert_frame_equal(output,csv(path),check_exact=True)
  exports=[dict(file=path.name,model=winner,sha256=sha(path),rows=2246,status='READY_NOT_SUBMITTED',public_ap=None)]
 done=dict(status='COMPLETE',signature=plan['signature'],real_fit_count=len(fit_log),new_fits_this_invocation=sum(not x['reused'] for x in fit_log),fits=fit_log,weights=weight_stats,eligible=eligible,files=exports,elapsed_seconds=time.perf_counter()-start,
  class_correction_raw_ranking_exact=True,parent_mean3_reconstructed_exact=True,new_recent_AP=0,uploads=0,bootstrap_computed=bool(eligible),artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},
  job_artifacts={str(p.relative_to(a.out)):sha(p) for p in (a.out/'jobs').glob('*/*') if p.is_file()})
 write(a.out/'complete.json',done);print(json.dumps(done,indent=2));return done


if __name__=='__main__':
 p=ArgumentParser(description=__doc__)
 for n in ['raw','out','feature-cache','baseline-cv','baseline-audit','seed42-oof','other-seed-oof','provider-oof','provider-test']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--threads',type=int,default=2);p.set_defaults(smoke=False);run(p.parse_args())
