"""Official TabM k1/k32 screen on the original dev quarters, then fixed full refit.

No Q2/Q3 selection. Returned predictions are blended locally with the exact
already-submitted context model; no reference-file transfer to GPU is required.
"""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os,platform,sys,tarfile,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hw4_inspections.features import build_features
from hw4_inspections.extensions import build_context_features
from hw4_inspections.delivery import validate_submission
try:
 from hw4_inspections.tabm_training import fit_tabm,predict_bundle,OPTIMIZER
except ImportError as error:
 raise RuntimeError('Install pinned official wheels: python -m pip install --no-deps -r configs/hw4-tabm-requirements.txt') from error

QUARTERS=['2023-07-01','2023-10-01','2024-01-01']
RAW=['train.csv','test_student.csv','inspection_history.csv','sample_submission.csv']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)

def main():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument('--raw',type=Path,default=ROOT/'data');a.add_argument('--out',type=Path,default=ROOT/'analysis/hw4-tabm-v100-01')
 a.add_argument('--device',choices=['cpu','cuda'],default='cuda');a.add_argument('--threads',type=int,default=4)
 a.add_argument('--members',nargs='+',type=int,choices=[1,32],default=[1,32])
 a.add_argument('--max-epochs',type=int,default=60);a.add_argument('--patience',type=int,default=10)
 a.add_argument('--width',type=int,default=256);a.add_argument('--blocks',type=int,default=3)
 a.add_argument('--bins',type=int,default=16);a.add_argument('--embedding-dim',type=int,default=8)
 a.add_argument('--batch-size',type=int,default=256);a.add_argument('--resume',action='store_true');a.add_argument('--smoke',action='store_true')
 args=a.parse_args()
 if min(args.threads,args.max_epochs,args.patience,args.width,args.blocks,args.bins,args.embedding_dim,args.batch_size)<1 or len(args.members)!=len(set(args.members)) or 32 not in args.members:raise ValueError('Positive limits and unique members including32 required')
 if any(not (args.raw/f).is_file() for f in RAW):raise FileNotFoundError(f'Put four official CSV directly into {args.raw.resolve()}')
 if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; no fallback')
 torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False
 started=time.monotonic()
 def log(event,**detail):print(json.dumps(dict(event=event,elapsed_seconds=round(time.monotonic()-started,3),**detail)),flush=True)
 args.out.mkdir(parents=True,exist_ok=True)
 sources=[Path(__file__),*(ROOT/'src/hw4_inspections'/f for f in ['features.py','extensions.py','neural.py','delivery.py','tabm_models.py','tabm_training.py'])]
 config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
 provenance={'config':config,'input_sha256':{f:sha(args.raw/f) for f in RAW},'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'versions':{n:importlib.metadata.version(n) for n in ['torch','tabm','rtdl-num-embeddings','numpy','pandas','scikit-learn','joblib']},'machine':platform.machine(),'device_name':torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU','optimizer':OPTIMIZER,'recipe':{'features':346,'primary_k':32,'control_k':1,'seed':42,'selection':'previous-quarter inner AP of mean probabilities','final_epochs':'median of three earlier-inner selections for k32','blending':'deferred exact75% current context rank +25% TabM rank after return','Q2Q3_selection':False,'external_pretrained_weights':False}}
 signature=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest();proto=args.out/'protocol.json'
 if proto.exists():
  old=json.loads(proto.read_text())
  if not args.resume or old['signature']!=signature:raise ValueError('New --out or unchanged --resume required')
 write_json(proto,dict(provenance,signature=signature,status='running',kaggle_uploads=0,new_closed_quarter_metrics=False))
 if args.device=='cuda':
  probe=torch.ones(1,device='cuda');assert (probe+1).item()==2
  log('cuda_ready',gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
 log('features_start',device='cpu')
 train=pd.read_csv(args.raw/'train.csv');test=pd.read_csv(args.raw/'test_student.csv');sample=pd.read_csv(args.raw/'sample_submission.csv')
 if args.smoke:train=train.groupby('forecast_period',group_keys=False).head(32).reset_index(drop=True)
 rows=pd.concat([train.drop(columns='target'),test],ignore_index=True);history=pd.read_csv(args.raw/'inspection_history.csv')
 x,cats=build_features(rows,history);x=pd.concat([x,build_context_features(rows,history)],axis=1);del history
 if len(x.columns)!=346:raise ValueError('Frozen feature view changed')
 log('features_complete',rows=len(x),columns=len(x.columns),visit_sequences='not used')
 common={'seed':42,'device':args.device,'patience':args.patience,'batch_size':args.batch_size,'width':args.width,'blocks':args.blocks,'bins':args.bins,'embedding_dim':args.embedding_dim,'threads':args.threads}
 records=[];predictions=[];selected=[];reused=0
 for k in args.members:
  for q in QUARTERS:
   job=args.out/f'tabm{k}_{q}';job.mkdir(exist_ok=True);marker=job/'complete.json'
   fit_idx=np.flatnonzero(train.forecast_period.lt(q));valid_idx=np.flatnonzero(train.forecast_period.eq(q))
   innerq=train.iloc[fit_idx].forecast_period.max();innerfit=np.flatnonzero(train.forecast_period.lt(innerq));innereval=np.flatnonzero(train.forecast_period.eq(innerq))
   if args.resume and marker.exists():
    rec=json.loads(marker.read_text())
    if rec['signature']!=signature or any(sha(job/f)!=h for f,h in rec['artifacts'].items()):raise ValueError('Changed completed TabM job')
    p=np.load(job/'prediction.npy',allow_pickle=False);reused+=1;log('job_reused',k=k,quarter=q)
   else:
    def progress(event,**detail):log(event,k=k,quarter=q,**detail)
    log('inner_fit_start',k=k,quarter=q,inner_quarter=innerq)
    inner=fit_tabm(x.iloc[innerfit],train.iloc[innerfit].target,cats,k=k,epochs=min(2,args.max_epochs) if args.smoke else args.max_epochs,eval_frame=x.iloc[innereval],eval_target=train.iloc[innereval].target,log=progress,**common)
    epochs=inner['epochs'];innercurve=inner['curve'];del inner
    log('outer_fit_start',k=k,quarter=q,epochs=epochs)
    bundle=fit_tabm(x.iloc[fit_idx],train.iloc[fit_idx].target,cats,k=k,epochs=epochs,log=progress,**common)
    p=predict_bundle(bundle,x.iloc[valid_idx]);joblib.dump(bundle,job/'model.joblib');loaded=joblib.load(job/'model.joblib')
    if not np.array_equal(p,predict_bundle(loaded,x.iloc[valid_idx])):raise AssertionError('Saved TabM reload differs')
    np.save(job/'prediction.npy',p,allow_pickle=False);write_json(job/'curves.json',{'inner':innercurve,'outer':bundle['curve']})
    rec={'signature':signature,'model':f'tabm{k}','k':k,'quarter':q,'inner_quarter':innerq,'epochs':int(epochs),'ap':float(average_precision_score(train.iloc[valid_idx].target,p)),'parameters':sum(v.numel() for v in bundle['model'].parameters()),'loaded_forward_exact':True,'official_kwargs':bundle['official_kwargs'],'model_kwargs':bundle['model_kwargs'],'artifacts':{f:sha(job/f) for f in ['model.joblib','prediction.npy','curves.json']}}
    write_json(marker,rec);log('fit_complete',k=k,quarter=q,ap=rec['ap'],epochs=epochs,parameters=rec['parameters'],loaded_forward_exact=True)
    del bundle,loaded
   records.append(rec)
   if k==32:selected.append(rec['epochs'])
   predictions.append(train.iloc[valid_idx][['row_id','site_id','forecast_period','target']].assign(model=f'tabm{k}',prediction=np.asarray(p,dtype=np.float64)))
 pd.concat(predictions,ignore_index=True).to_csv(args.out/'predictions.csv',index=False)
 metrics=pd.DataFrame([{k:r[k] for k in ['model','quarter','epochs','ap','parameters','loaded_forward_exact']} for r in records]);metrics.to_csv(args.out/'fold_metrics.csv',index=False)
 summary=metrics.groupby('model',as_index=False).agg(mean_ap=('ap','mean'));summary.to_csv(args.out/'summary.csv',index=False)
 epochs=int(np.median(selected));job=args.out/'full_tabm32';job.mkdir(exist_ok=True);marker=job/'complete.json'
 if args.resume and marker.exists():
  rec=json.loads(marker.read_text())
  if rec['signature']!=signature or any(sha(job/f)!=h for f,h in rec['artifacts'].items()):raise ValueError('Changed full TabM artifact')
  p=np.load(job/'prediction.npy',allow_pickle=False);reused+=1
 else:
  def progress(event,**detail):log(event,phase='full_refit',k=32,**detail)
  log('full_refit_start',epochs=epochs,rows=len(train))
  bundle=fit_tabm(x.iloc[:len(train)],train.target,cats,k=32,epochs=epochs,log=progress,**common)
  p=predict_bundle(bundle,x.iloc[len(train):]);joblib.dump(bundle,job/'model.joblib')
  if not np.array_equal(p,predict_bundle(joblib.load(job/'model.joblib'),x.iloc[len(train):])):raise AssertionError('Full TabM reload differs')
  np.save(job/'prediction.npy',p,allow_pickle=False);write_json(job/'training_curve.json',bundle['curve'])
  write_json(marker,{'signature':signature,'epochs':epochs,'loaded_forward_exact':True,'official_kwargs':bundle['official_kwargs'],'artifacts':{f:sha(job/f) for f in ['model.joblib','prediction.npy','training_curve.json']}})
 test_predictions=test[['row_id','forecast_period']].assign(prediction=np.asarray(p,dtype=np.float64));test_predictions.to_csv(args.out/'test_predictions.csv',index=False)
 submission=validate_submission(test,sample,p,expected_rows=2246);submission.to_csv(args.out/'submission_tabm32.csv',index=False)
 write_json(proto,dict(provenance,signature=signature,status='complete',kaggle_uploads=0,new_closed_quarter_metrics=False,final_epochs=epochs,reused_jobs=reused,full_train_rows=len(train),test_rows=len(test),exports_status='diagnostic pending matched current-context evaluation locally'))
 files=['protocol.json','predictions.csv','fold_metrics.csv','summary.csv','test_predictions.csv','submission_tabm32.csv']
 with tarfile.open(args.out/'return_results.tar.gz','w:gz') as archive:
  for f in files:archive.add(args.out/f,arcname=f)
 print(summary.to_string(index=False),flush=True);log('complete',results=str(args.out/'return_results.tar.gz'),kaggle_uploads=0)

if __name__=='__main__':main()
