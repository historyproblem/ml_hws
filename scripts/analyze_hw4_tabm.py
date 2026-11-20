"""Audit a returned TabM archive and apply the predeclared local75/25 gate.

No training, weight search, reserved-quarter metrics or automatic submissions.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,tarfile
from pathlib import Path,PurePosixPath
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hw4_inspections.next_round import QUARTERS,rank_scores,evaluate_candidates
from hw4_inspections.delivery import validate_submission
FILES={'protocol.json','predictions.csv','fold_metrics.csv','summary.csv','test_predictions.csv','submission_tabm32.csv'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument('--incoming',type=Path,required=True);a.add_argument('--raw',type=Path,default=ROOT/'data')
 a.add_argument('--baseline-cv',type=Path,required=True);a.add_argument('--context-test',type=Path,required=True)
 a.add_argument('--out',type=Path,required=True);a.add_argument('--bootstrap',type=int,default=2000)
 a.add_argument('--allow-synthetic-qa',action='store_true',help=argparse.SUPPRESS)
 args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 incoming=args.out/'incoming';incoming.mkdir(exist_ok=True)
 with tarfile.open(args.incoming,'r:gz') as archive:
  members=archive.getmembers()
  if len(members)!=6 or {m.name for m in members}!=FILES:raise ValueError('Unexpected archive member set')
  for m in members:
   path=PurePosixPath(m.name)
   if not m.isfile() or path.is_absolute() or '..' in path.parts or m.size>25_000_000:raise ValueError('Unsafe archive member')
   (incoming/m.name).write_bytes(archive.extractfile(m).read())
 p=json.loads((incoming/'protocol.json').read_text())
 if p['status']!='complete' or p['config']['smoke'] and not args.allow_synthetic_qa:raise ValueError('Completed real run required')
 keys=['config','input_sha256','source_sha256','versions','machine','device_name','optimizer','recipe']
 expected=hashlib.sha256(json.dumps({k:p[k] for k in keys},sort_keys=True).encode()).hexdigest()
 if p['signature']!=expected:raise ValueError('TabM protocol signature mismatch')
 if set(p['input_sha256'])!={'train.csv','test_student.csv','inspection_history.csv','sample_submission.csv'}:raise ValueError('Unexpected official input keys')
 for f,digest in p['input_sha256'].items():
  if sha(args.raw/f)!=digest:raise ValueError('Official input SHA mismatch')
 expected_sources={'scripts/run_hw4_tabm.py',*[f'src/hw4_inspections/{n}' for n in ['features.py','extensions.py','neural.py','delivery.py','tabm_models.py','tabm_training.py']]}
 if set(p['source_sha256'])!=expected_sources:raise ValueError('Unexpected source-file keys')
 for f,digest in p['source_sha256'].items():
  if sha(ROOT/f)!=digest:raise ValueError(f'Published source mismatch: {f}')
 if p.get('new_closed_quarter_metrics') is not False:raise ValueError('Unexpected new reserved-quarter metrics')
 train=pd.read_csv(args.raw/'train.csv');test=pd.read_csv(args.raw/'test_student.csv');sample=pd.read_csv(args.raw/'sample_submission.csv')
 if args.allow_synthetic_qa:train=train.groupby('forecast_period',group_keys=False).head(32).reset_index(drop=True)
 meta=train.loc[train.forecast_period.isin(QUARTERS),['row_id','site_id','forecast_period','target']].reset_index(drop=True)
 base=pd.read_csv(args.baseline_cv,float_precision='round_trip')
 if base.row_id.duplicated().any():raise ValueError('Duplicate baseline identities')
 base=base.set_index('row_id').reindex(meta.row_id)
 for c in ['site_id','forecast_period','target']:
  if not np.array_equal(base[c].to_numpy(),meta[c].to_numpy()):raise ValueError('Baseline metadata mismatch')
 baseline=base.prediction.to_numpy(dtype=float)
 if not np.isfinite(baseline).all() or ((baseline<0)|(baseline>1)).any():raise ValueError('Bad context baseline probabilities')
 long=pd.read_csv(incoming/'predictions.csv',float_precision='round_trip')
 if long.duplicated(['model','row_id']).any():raise ValueError('Duplicate model/row predictions')
 names={f'tabm{k}' for k in p['config']['members']}
 if set(long.model)!=names:raise ValueError('Model coverage mismatch')
 scores={'context_3seed':baseline}
 recomputed=[]
 for name in sorted(names):
  part=long.loc[long.model.eq(name)]
  if set(part.row_id)!=set(meta.row_id):raise ValueError('Missing/extra old-dev IDs')
  part=part.set_index('row_id').loc[meta.row_id]
  for c in ['site_id','forecast_period','target']:
   if not np.array_equal(part[c].to_numpy(),meta[c].to_numpy()):raise ValueError('Official prediction metadata mismatch')
  values=part.prediction.to_numpy(dtype=float)
  if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('Bad TabM probabilities')
  scores[name]=values
  for q in QUARTERS:
   mask=meta.forecast_period.eq(q).to_numpy();recomputed.append({'model':name,'quarter':q,'ap':float(average_precision_score(meta.loc[mask,'target'],values[mask]))})
 actual=pd.DataFrame(recomputed).set_index(['model','quarter']).sort_index()
 reported=pd.read_csv(incoming/'fold_metrics.csv',float_precision='round_trip').set_index(['model','quarter']).sort_index()
 np.testing.assert_allclose(actual.ap,reported.ap,atol=1e-14,rtol=0)
 means=actual.groupby('model').ap.mean();reportedmean=pd.read_csv(incoming/'summary.csv',float_precision='round_trip').set_index('model')
 np.testing.assert_allclose(means,reportedmean.loc[means.index,'mean_ap'],atol=1e-14,rtol=0)
 mix=np.empty(len(meta))
 for q in QUARTERS:
  mask=meta.forecast_period.eq(q).to_numpy();mix[mask]=.75*rank_scores(baseline[mask])+.25*rank_scores(scores['tabm32'][mask])
 scores['context_tabm_75_25']=mix
 result=evaluate_candidates(meta,scores,bootstrap=args.bootstrap)
 for n in ['summary','quarter_scores']:result[n].to_csv(args.out/f'{n}.csv',index=False)
 testp=pd.read_csv(incoming/'test_predictions.csv',float_precision='round_trip')
 if testp.row_id.duplicated().any() or set(testp.row_id)!=set(test.row_id):raise ValueError('Test prediction ID mismatch')
 testp=testp.set_index('row_id').loc[test.row_id]
 if not np.array_equal(testp.forecast_period.to_numpy(),test.forecast_period.to_numpy()):raise ValueError('Test periods differ')
 probability=testp.prediction.to_numpy(dtype=float)
 if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():raise ValueError('Bad test probabilities')
 original=pd.read_csv(incoming/'submission_tabm32.csv',float_precision='round_trip').set_index('row_id')
 np.testing.assert_array_equal(probability,original.loc[test.row_id,'target'].to_numpy())
 anchor=pd.read_csv(args.context_test,float_precision='round_trip')
 if anchor.row_id.duplicated().any() or set(anchor.row_id)!=set(test.row_id):raise ValueError('Anchor IDs differ')
 anchor_p=anchor.set_index('row_id').loc[test.row_id].target.to_numpy(dtype=float)
 if not np.isfinite(anchor_p).all() or ((anchor_p<0)|(anchor_p>1)).any():raise ValueError('Bad test anchor probabilities')
 values={'tabm32':probability,'context_tabm_75_25':.75*rank_scores(anchor_p)+.25*rank_scores(probability)}
 exports={}
 for name,v in values.items():
  directory=args.out/('recommended' if result['eligibility'][name] else 'diagnostic');directory.mkdir(exist_ok=True)
  file=directory/f'submission_{name}.csv';validate_submission(test,sample,v,expected_rows=2246).to_csv(file,index=False)
  exports[name]={'file':str(file),'sha256':sha(file),'eligible':result['eligibility'][name]}
 write_json(args.out/'verification.json',{'archive_sha256':sha(args.incoming),'analyzer_source_sha256':sha(Path(__file__)),'protocol_signature_exact':True,'source_and_data_SHA_exact':True,'official_dev_rows':len(meta),'all_AP_and_means_reproduced':True,'test_CSV_exact':True,'baseline_SHA':sha(args.baseline_cv),'anchor_SHA':sha(args.context_test),'gate':result['protocol'],'eligible':result['eligibility'],'exports':exports,'new_fits':0,'new_closed_quarter_metrics':False,'kaggle_uploads':0,'synthetic_qa_only':args.allow_synthetic_qa,'independent_checkpoint_reload':False})
 print(result['summary'].to_string(index=False));print(json.dumps(exports))

if __name__=='__main__':main()
