"""Fixed compact94 retrieval/control MLP; timing gate before OLD-only training."""
from pathlib import Path
from datetime import datetime,timezone
from importlib import metadata
import copy,hashlib,json,os,random,resource,sys,time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1];WORK=Path('/Users/khlystov/Documents/ChatGPT/ML ДЗ');BASE=WORK/'analysis/hw4-post-public-20261122/retrieval-mlp94'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from hw4_inspections import retrieval_mlp as models
import run_hw4_final_research as trusted
from hw4_inspections.classical_features import select_feature_columns

OLD=['2023-07-01','2023-10-01','2024-01-01'];META=['row_id','site_id','forecast_period','target']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
csv=lambda p:pd.read_csv(p,float_precision='round_trip')


def write(p,j):p.write_text(json.dumps(j,sort_keys=True,indent=2,allow_nan=False)+'\n')
def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)


def main():
    BASE.mkdir(parents=True,exist_ok=True);started=time.time();deadline=min(started+600,datetime(2026,11,22,18,29,tzinfo=timezone.utc).timestamp())
    raw=WORK/'data/hw4/raw';cachefile=WORK/'analysis/hw4-deep-research-20261122/features.joblib';protocol=WORK/'analysis/hw4-six-hour-20261122/tabicl-mac-own22/protocol.json'
    baseline=WORK/'analysis/hw4-next-round-20261121/baseline_oof.csv';iclfile=WORK/'analysis/hw4-six-hour-20261122/gpu-analysis/tabicl/old-oof.csv';icltest=WORK/'analysis/hw4-six-hour-20261122/gpu-analysis/tabicl/raw-selected-test.csv'
    source=[Path(__file__),Path(models.__file__),ROOT/'scripts/run_hw4_final_research.py',ROOT/'src/hw4_inspections/features.py',ROOT/'src/hw4_inspections/extensions.py',ROOT/'src/hw4_inspections/classical_features.py',protocol,baseline,iclfile,icltest]
    architecture=dict(numeric='fit-only median; nonconstant train-only normalquantile<=1000 clipped[-5,5]; missing flags for90 numeric',categorical='four fit-only vocabularies, unknown0,embeddings4',encoder='Linear196x64,GELU,drop.1,Linear64x64,GELU,drop.1,LN64',head='Linear64x1',retrieval='keyLinear64normalized; exacttorch k16 detachedFITcache each epoch; chosen neighbors reencoded with gradients; attentionsoftmax(-8sqdist); value=labelEmb2x64+Linear64,GELU,drop.1,Linear64(featuredifference); residual gaininit.1',mask='exclude candidateperiod>=queryperiod and samephysicalsite for train/validation/test; no eligible candidates => zero aggregate')
    plan=dict(format='hw4-simplified-causal-retrieval-mlp-compact94-1',view='frozen compact96 selector minus forecast_year/cutoff_gap_days =94 inputs:90numeric+4categorical',architecture=architecture,algorithm_names=['control','retrieval'],seed=42,threads=2,optimizer=dict(name='AdamW',lr=.001,weight_decay=.0003,batch=256,clip_norm=1,max_epochs=30,inner_patience=6),k=16,old_quarters=OLD,epoch_selection='Eachalgorithm independently selects earliestmaxAP on previousquarter with earlierfit; outerfreshfixedselectedN; fullmedian3innerN. No representation/width/weight search.',formulas=['raw','fixedprobability50ICL94/50new'],gate='RawvsC04 mean>0,>=2nonnegative,worst>=-.005,Q4>=0; mixvsICL94 same. At most one final by highestOLDmean among admissible predefined outputs.',resource_gate='Two epochs each algorithm on earliestinner, noAP; conservative projected12fits+2full<=600sec or stopbeforequality. Actual walltime<=10min and18:29UTC.',source_and_inputs_sha256={str(p):sha(p) for p in source},raw_sha256={n:sha(raw/n) for n in trusted.RAW_FILES},cache_sha256=sha(cachefile),cache_meta_sha256=sha(cachefile.with_name('features-meta.json')),packages={n:metadata.version(n) for n in ['torch','numpy','pandas','scikit-learn','joblib']},primary_sources=['https://github.com/yandex-research/tabular-dl-tabr/blob/main/README.md','https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html'],interpretation='Simplified retrievalMLP inspired by TabR, not official implementation. Exploratory postPublic follow-up after observing own22 negative; same preset, only compact94 input support changes. OLD reused; not independent evidence, no new recentAP.',new_recent_AP=0,uploads=0)
    if (BASE/'freeze.json').exists():raise ValueError('Existing run is immutable; inspect results rather than launch again')
    seed();test_result=models.targeted_tests();write(BASE/'targeted-tests.json',test_result)
    # Everything above is synthetic; this freezes before any real model/metric.
    write(BASE/'freeze.json',plan)
    from argparse import Namespace
    train,test,sample,views,cats=trusted.validate_cache(joblib.load(cachefile),Namespace(raw=raw,smoke=False))
    fields=[c for c in select_feature_columns(views['base'],'compact_context') if c not in ['forecast_year','cutoff_gap_days']];categories=read(protocol)['categories'];frame=views['base'][fields]
    if len(fields)!=94 or len(categories)!=4 or len([c for c in fields if c not in categories])!=90:raise ValueError('Expected fixed compact94')
    rowmeta=pd.concat([train.drop(columns='target'),test],ignore_index=True)
    sites=pd.factorize(rowmeta.site_id,sort=True)[0]
    periods=pd.PeriodIndex(rowmeta.forecast_period,freq='Q').asi8
    meta=pd.concat([train.loc[train.forecast_period.eq(q),META] for q in OLD],ignore_index=True)
    def align(f):
        if not f.row_id.is_unique or set(f.row_id)!=set(meta.row_id):raise ValueError('Different parentIDs')
        f=f.set_index('row_id').loc[meta.row_id].reset_index();pd.testing.assert_frame_equal(meta,f[META],check_dtype=False,check_exact=True);return f
    context=align(csv(baseline)).prediction.to_numpy();icl=align(csv(iclfile)).tabicl_v2_compact94__raw.to_numpy()
    counts={'probe_fits':0,'inner_fits':0,'outer_fits':0,'full_fits':0};job_records=[]
    def fit_job(kind,fit_idx,query_idx,epochs,jobid,eval_target=None,probe=False):
        if time.time()>deadline:raise TimeoutError('Fixed10min/deadline resource gate')
        t0=time.perf_counter();seed();prep=models.Preprocessor().fit(frame.iloc[fit_idx],categories)
        x,c=prep.transform(frame.iloc[fit_idx]);qx,qc=prep.transform(frame.iloc[query_idx])
        candidate=dict(numeric=x,categories=c,site=torch.tensor(sites[fit_idx]),period=torch.tensor(periods[fit_idx]),target=torch.tensor(train.target.iloc[fit_idx].to_numpy(),dtype=torch.float32))
        qs=torch.tensor(sites[query_idx]);qp=torch.tensor(periods[query_idx]);target=candidate['target']
        model=models.RetrievalMLP(x.shape[1],prep.cardinalities,retrieval=kind=='retrieval');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
        rng=np.random.default_rng(42);curve=[];best=-1.;best_epoch=epochs;best_state=None;wait=0;timing=[]
        def predict():
            model.eval();cache=model.cache(candidate) if model.retrieval else None;parts=[]
            with torch.no_grad():
                for i in range(0,len(qx),256):parts.append(torch.sigmoid(model(qx[i:i+256],qc[i:i+256],qs[i:i+256],qp[i:i+256],candidate,cache)).numpy())
            return np.concatenate(parts).astype(np.float64)
        for epoch in range(1,epochs+1):
            if time.time()>deadline:raise TimeoutError('Fixed10min/deadline resource gate')
            e0=time.perf_counter();model.train();cache=model.cache(candidate) if model.retrieval else None
            order=rng.permutation(len(x))
            for start in range(0,len(x),256):
                idx=torch.tensor(order[start:start+256]);optimizer.zero_grad(set_to_none=True)
                logits=model(x[idx],c[idx],candidate['site'][idx],candidate['period'][idx],candidate,cache)
                loss=nn.functional.binary_cross_entropy_with_logits(logits,target[idx]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            train_seconds=time.perf_counter()-e0;p0=time.perf_counter();p=predict();prediction_seconds=time.perf_counter()-p0
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid neural probabilities')
            timing.append(dict(epoch=epoch,train_seconds=train_seconds,predict_seconds=prediction_seconds))
            if eval_target is not None and not probe:
                ap=float(average_precision_score(eval_target,p));curve.append(dict(epoch=epoch,ap=ap))
                if ap>best:best=ap;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());wait=0
                else:wait+=1
                if wait>=6:break
        if best_state is not None:model.load_state_dict(best_state)
        prediction=predict();directory=BASE/'jobs'/jobid;directory.mkdir(parents=True)
        torch.save(model.state_dict(),directory/'state.pt');joblib.dump(prep,directory/'preprocessor.joblib')
        clone=models.RetrievalMLP(x.shape[1],prep.cardinalities,retrieval=kind=='retrieval');clone.load_state_dict(torch.load(directory/'state.pt',weights_only=True));clone.eval()
        original=model;model=clone;reloaded=predict();model=original
        if not np.array_equal(prediction,reloaded):raise ValueError('Neural saved forward differs')
        np.save(directory/'prediction.npy',prediction,allow_pickle=False)
        rec=dict(kind=kind,fit_rows=len(fit_idx),query_rows=len(query_idx),fit_ids_sha256=hashlib.sha256(json.dumps(rowmeta.row_id.iloc[fit_idx].tolist()).encode()).hexdigest(),query_ids_sha256=hashlib.sha256(json.dumps(rowmeta.row_id.iloc[query_idx].tolist()).encode()).hexdigest(),selected_epochs=best_epoch,actual_epochs=len(timing),curve=curve,timing=timing,seconds=time.perf_counter()-t0,reload_exact=True,query_labels_used_only_for_innerAP=eval_target is not None and not probe,artifacts={p.name:sha(p) for p in directory.iterdir() if p.is_file()})
        write(directory/'complete.json',rec);job_records.append(dict(job=jobid,**rec));print(json.dumps({'event':'job_complete','job':jobid,'epochs':best_epoch,'seconds':rec['seconds']}),flush=True)
        return prediction,rec
    # Real timings without quality information; both models are discarded.
    earliest=(pd.Period(OLD[0],freq='Q')-1).start_time.strftime('%Y-%m-%d');probe_fit=np.flatnonzero(train.forecast_period.lt(earliest));probe_query=np.flatnonzero(train.forecast_period.eq(earliest));probe={}
    for kind in ['control','retrieval']:
        _,probe[kind]=fit_job(kind,probe_fit,probe_query,2,'probe-'+kind,probe=True);counts['probe_fits']+=1
    projected=0.;projection={}
    for kind in ['control','retrieval']:
        timing=probe[kind]['timing'];tr=float(np.mean([r['train_seconds'] for r in timing]));pr=float(np.mean([r['predict_seconds'] for r in timing]));seconds=0.
        for q in OLD:
            iq=(pd.Period(q,freq='Q')-1).start_time.strftime('%Y-%m-%d')
            for point in [iq,q]:
                n=int(train.forecast_period.lt(point).sum());m=int(train.forecast_period.eq(point).sum());seconds+=30*(tr*(n/len(probe_fit))**(2 if kind=='retrieval' else 1)+pr*(n/len(probe_fit) if kind=='retrieval' else 1)*(m/len(probe_query)))
        seconds+=30*(tr*(len(train)/len(probe_fit))**(2 if kind=='retrieval' else 1)+pr*(len(train)/len(probe_fit) if kind=='retrieval' else 1)*(len(test)/len(probe_query)))
        projection[kind]=seconds;projected+=seconds
    projection.update(projected_total_seconds_with15pct_buffer=1.15*projected,available_seconds=deadline-time.time(),peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),probe_AP_calls=0)
    write(BASE/'resource-probe.json',projection);print(json.dumps({'event':'resource_projection',**projection}),flush=True)
    if 1.15*projected>min(600,deadline-time.time()):
        write(BASE/'complete.json',dict(status='RESOURCE_GATE_CLOSED',counts=counts,projection=projection,new_recent_AP=0,uploads=0));return
    predictions={};inner={};epochs={}
    for kind in ['control','retrieval']:
        inner[kind]={};epochs[kind]={}
        # All three previous-inner choices are fixed before this kind's outer fits.
        for q in OLD:
            iq=(pd.Period(q,freq='Q')-1).start_time.strftime('%Y-%m-%d');fit=np.flatnonzero(train.forecast_period.lt(iq));query=np.flatnonzero(train.forecast_period.eq(iq))
            _,rec=fit_job(kind,fit,query,30,f'{kind}-inner-{q}',eval_target=train.target.iloc[query].to_numpy());counts['inner_fits']+=1;inner[kind][q]=rec;epochs[kind][q]=rec['selected_epochs']
        parts=[]
        for q in OLD:
            fit=np.flatnonzero(train.forecast_period.lt(q));query=np.flatnonzero(train.forecast_period.eq(q));p,_=fit_job(kind,fit,query,epochs[kind][q],f'{kind}-outer-{q}');counts['outer_fits']+=1;parts.append(p)
        predictions[kind]=np.concatenate(parts)
    write(BASE/'inner-selection.json',epochs)
    values={'context04':context,'icl94':icl}
    for kind,p in predictions.items():values[kind]=p;values[kind+'_icl50']=.5*p+.5*icl
    metrics=[]
    for q in OLD:
        mask=meta.forecast_period.eq(q).to_numpy()
        for name,p in values.items():metrics.append(dict(quarter=q,model=name,ap=float(average_precision_score(meta.target[mask],p[mask]))))
    metrics=pd.DataFrame(metrics);table=metrics.pivot(index='quarter',columns='model',values='ap');summary=[]
    for name in table:
        ref='icl94' if name.endswith('_icl50') else 'context04';d=table[name]-table[ref]
        eligible=name not in ['context04','icl94'] and d.mean()>0 and (d>=0).sum()>=2 and d.min()>=-.005 and d.loc['2023-10-01']>=0
        summary.append(dict(model=name,mean_ap=float(table[name].mean()),reference=ref,delta=float(d.mean()),nonnegative_quarters=int((d>=0).sum()),worst=float(d.min()),q4_delta=float(d.loc['2023-10-01']),eligible=bool(eligible)))
    summary=pd.DataFrame(summary).sort_values(['mean_ap','model'],ascending=[False,True]);summary.to_csv(BASE/'summary.csv',index=False);metrics.to_csv(BASE/'quarter-metrics.csv',index=False);pd.concat([meta,pd.DataFrame(values)],axis=1).to_csv(BASE/'old-oof.csv',index=False)
    exports=[];eligible=summary.loc[summary.eligible]
    if len(eligible):
        winner=eligible.iloc[0].model;kind=winner.split('_')[0];n=int(np.median(list(epochs[kind].values())))
        p,_=fit_job(kind,np.arange(len(train)),np.arange(len(train),len(frame)),n,kind+'-full');counts['full_fits']+=1
        if winner.endswith('_icl50'):parent=csv(icltest).set_index('row_id').loc[test.row_id].prediction.to_numpy();p=.5*p+.5*parent
        output=trusted.validate_submission(test,sample,p,len(test));path=BASE/f'submission_{winner}.csv';output.to_csv(path,index=False);pd.testing.assert_frame_equal(output,csv(path),check_exact=True);exports=[dict(file=path.name,sha256=sha(path),model=winner,rows=2246,status='READY_NOT_SUBMITTED',public_ap=None)]
    write(BASE/'complete.json',dict(status='COMPLETE',counts=counts,exports=exports,elapsed_seconds=time.time()-started,new_recent_AP=0,uploads=0,projection=projection,source_sha256=plan['source_and_inputs_sha256'],job_records=job_records));print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
