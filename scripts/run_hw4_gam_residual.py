"""Causal OOF GAM baseline -> fixed shallow CatBoost binary-loss correction."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_hw4_final_research as primary
from hw4_inspections.additive_search import fit_recipe,recipe_by_id
from hw4_inspections.gam_residual import (
    GRID,PARAMS,TEACHER_RECIPE,choose_count,fit_residual,predict_residual,residual_raw,teacher_raw,
)
from hw4_inspections.next_round import evaluate_candidates

DEFAULT_PLAN=ROOT/'docs/hw4-gam-residual-plan-20261122.json'
sha,read_json,write_json=primary.sha,primary.read_json,primary.write_json
read_csv,write_csv=primary.read_csv,primary.write_csv


def make_freeze(args):
    plan=read_json(args.plan)
    if plan['teacher']['recipe']!=TEACHER_RECIPE or tuple(plan['selection']['grid'])!=GRID or plan['residual']['params']!=PARAMS:
        raise ValueError('Frozen residual teacher/grid/params differ')
    frozen=primary.provenance(args,{'residual':dict(params=PARAMS,grid=list(GRID),teacher=TEACHER_RECIPE)})
    module=ROOT/'src/hw4_inspections'
    frozen['source_sha256'].update({n:sha(module/n) for n in ['additive_logit.py','classical_features.py','additive_search.py','gam_residual.py']})
    frozen['source_sha256'][Path(__file__).name]=sha(Path(__file__))
    manifest=read_json(args.algorithm_control/'manifest.json');control=read_json(args.algorithm_control/'freeze.json');claimed=control.pop('signature')
    if hashlib.sha256(json.dumps(control,sort_keys=True).encode()).hexdigest()!=claimed or manifest['signature']!=claimed:
        raise ValueError('Previous algorithm signature differs')
    if not manifest['complete'] or manifest['new_q2_q3_metrics'] is not False:raise ValueError('Previous algorithm is not completed old-dev only')
    for n,h in manifest['output_sha256'].items():
        if sha(args.algorithm_control/n)!=h:raise ValueError('Previous algorithm output changed')
    for key in ['raw_sha256','cache_sha256']:
        if control[key]!=frozen[key]:raise ValueError('Teacher/control input data differ')
    for name in ['additive_logit.py','classical_features.py','additive_search.py']:
        if control['source_sha256'][name]!=frozen['source_sha256'][name]:raise ValueError('Teacher scientific source differs')
    control_hash=dict(oof_sha256=sha(args.algorithm_control/'outer-oof.csv'),manifest_sha256=sha(args.algorithm_control/'manifest.json'))
    if not args.smoke and (control_hash['oof_sha256']!=plan['controls']['oof_sha256'] or control_hash['manifest_sha256']!=plan['controls']['manifest_sha256']):
        raise ValueError('Previous algorithm differs from pre-results plan')
    sources={}
    for q in [*pd.read_csv(args.raw/'train.csv').forecast_period.unique(),'full']:
        path=args.algorithm_control/'jobs'/f'{q}__{TEACHER_RECIPE}'
        if not path.exists():continue
        rec=read_json(path/'complete.json')
        if (rec['identity']['signature']!=claimed or rec['status']!='complete' or rec['reload_exact'] is not True
                or rec['identity']['recipe']!=TEACHER_RECIPE or rec['identity']['period']!=q):
            raise ValueError('Reusable teacher identity differs')
        if any(sha(path/n)!=h for n,h in rec['sha256'].items()):raise ValueError('Reusable teacher artifact changed')
        sources[q]=dict(path=str(path.resolve()),complete_sha256=sha(path/'complete.json'),artifacts=rec['sha256'],identity=rec['identity'])
    if 'full' not in sources:raise ValueError('The verified21017-row full fixed teacher is required, not a new initializer')
    frozen.update(plan_sha256=sha(args.plan),pre_results_plan=plan,algorithm_control=control_hash,
                  teacher_sources=sources,seeds=[42,17,2026],tree_selection='One max400 inner model, previous-quarter AP grid0/50/100/200/400, earliest tie0; final median')
    return frozen


class Teachers:
    def __init__(self,args,signature,sources,train,test,frame):
        self.args,self.signature,self.sources=args,signature,sources
        self.train,self.test,self.frame=train,test,frame
        self.first=sorted(train.forecast_period.unique())[0]
        self.oof=np.full(len(train),np.nan,dtype=np.float64)
        self.records={}

    def ensure(self,q):
        full=q=='full'
        if q==self.first:raise ValueError('Earliest-quarter teacher cannot be trained without past labels')
        fit=np.arange(len(self.train)) if full else np.flatnonzero(self.train.forecast_period.lt(q))
        pred=np.arange(len(self.train),len(self.frame)) if full else np.flatnonzero(self.train.forecast_period.eq(q))
        if not len(fit) or not len(pred):raise ValueError('Empty teacher fold')
        source=self.sources.get(q);identity=dict(signature=self.signature,kind='teacher',quarter=q,
            train_rows=len(fit),predict_rows=len(pred),recipe=TEACHER_RECIPE,source_sha256=source['complete_sha256'] if source else None)
        path=self.args.out/'teacher'/q
        if path.exists():
            rec=read_json(path/'complete.json')
            if rec['identity']!=identity or set(rec['sha256'])!={'model.joblib','raw.npy','row-ids.npy'} or any(sha(path/n)!=h for n,h in rec['sha256'].items()):raise ValueError('Changed teacher checkpoint')
            if rec['reload_raw_exact'] is not True:raise ValueError('Teacher raw reload failed')
            raw=np.load(path/'raw.npy',allow_pickle=False)
        else:
            print(json.dumps(dict(event='teacher_start',**identity,reused=bool(source))),flush=True)
            if source:
                source_path=Path(source['path']);source_marker=read_json(source_path/'complete.json')
                if sha(source_path/'complete.json')!=source['complete_sha256'] or sha(source_path/'model.joblib')!=source['artifacts']['model.joblib']:
                    raise ValueError('Teacher source changed since freeze')
                if source['identity']['train_rows']!=len(fit) or source['identity']['predict_rows']!=len(pred):raise ValueError('Teacher source train/predict partition differs')
                bundle=joblib.load(source_path/'model.joblib')
            else:
                if full:raise ValueError('Fullteacher must be reused')
                bundle=fit_recipe(self.frame.iloc[fit],self.train.target.iloc[fit],recipe_by_id(TEACHER_RECIPE))
            if bundle['recipe']!=recipe_by_id(TEACHER_RECIPE) or bundle['sample_weights'] is not None:
                raise ValueError('Teacher must be the fixed unweighted E/C.01')
            raw=teacher_raw(bundle,self.frame.iloc[pred])
            path.parent.mkdir(exist_ok=True)
            with TemporaryDirectory(dir=path.parent,prefix='.pending-teacher-') as temp:
                temp=Path(temp)/'job';temp.mkdir()
                if source:shutil.copyfile(Path(source['path'])/'model.joblib',temp/'model.joblib')
                else:joblib.dump(bundle,temp/'model.joblib',compress=1)
                if not np.array_equal(raw,teacher_raw(joblib.load(temp/'model.joblib'),self.frame.iloc[pred])):raise AssertionError('Teacher raw checkpoint replay differs')
                np.save(temp/'raw.npy',raw,allow_pickle=False)
                ids=self.test.row_id.to_numpy(dtype=str) if full else self.train.row_id.iloc[pred].to_numpy(dtype=str)
                np.save(temp/'row-ids.npy',ids,allow_pickle=False)
                rec=dict(identity=identity,reused_checkpoint=bool(source),new_fit=not bool(source),
                    train_max_period=self.train.forecast_period.iloc[fit].max(),predict_period=q,
                    evaluation_metrics_computed=False,reload_raw_exact=True,source_checkpoint=source)
                rec['sha256']={p.name:sha(p) for p in temp.iterdir()};write_json(temp/'complete.json',rec);os.rename(temp,path)
            print(json.dumps(dict(event='teacher_complete',quarter=q,fit_rows=len(fit),predict_rows=len(pred),new_fit=not bool(source),AP_computed=False)),flush=True)
        if raw.dtype!=np.float64 or raw.shape!=(len(pred),) or not np.isfinite(raw).all():raise ValueError('Invalid teacher raw logits')
        expected=self.test.row_id.to_numpy(dtype=str) if full else self.train.row_id.iloc[pred].to_numpy(dtype=str)
        if not np.array_equal(expected,np.load(path/'row-ids.npy',allow_pickle=False)):raise ValueError('Teacher row identity/order differs')
        if not full:
            if not rec['train_max_period']<q:raise ValueError('Teacher used its own/future quarter')
            self.oof[pred]=raw
        self.records[q]=rec
        return raw


def meta_job(args,sig,quarter,count,seed,teachers,train,frame,cats):
    full=quarter=='full';available=np.isfinite(teachers.oof)
    fit=np.flatnonzero(available if full else available&train.forecast_period.lt(quarter).to_numpy())
    pred=np.arange(len(train),len(frame)) if full else np.flatnonzero(train.forecast_period.eq(quarter))
    if not len(fit) or not len(pred):raise ValueError('Empty meta fold')
    if (train.forecast_period.iloc[fit]==teachers.first).any():raise AssertionError('Earliest un-OOF quarter entered meta fitting')
    baseline=teachers.oof[fit];identity=dict(signature=sig,kind='residual',quarter=quarter,trees=int(count),seed=int(seed),
        train_rows=len(fit),predict_rows=len(pred),training_baseline_sha256=hashlib.sha256(baseline.tobytes()).hexdigest())
    path=args.out/'meta'/f'{quarter}__n{count}__s{seed}'
    if path.exists():
        rec=read_json(path/'complete.json')
        if rec['identity']!=identity or rec['reload_raw_exact'] is not True or set(rec['sha256'])!={'model.joblib','raw.npy'} or any(sha(path/n)!=h for n,h in rec['sha256'].items()):raise ValueError('Meta checkpoint changed')
        raw=np.load(path/'raw.npy',allow_pickle=False)
        if raw.dtype!=np.float64 or raw.shape!=(len(pred),) or not np.isfinite(raw).all():raise ValueError('Meta raw forecast invalid')
        return joblib.load(path/'model.joblib'),raw,rec
    print(json.dumps(dict(event='meta_start',**identity)),flush=True)
    bundle=fit_residual(frame.iloc[fit],train.target.iloc[fit],cats,baseline,iterations=count,seed=seed)
    raw=residual_raw(bundle,frame.iloc[pred])
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent,prefix='.pending-residual-') as temp:
        temp=Path(temp)/'job';temp.mkdir();joblib.dump(bundle,temp/'model.joblib',compress=1)
        if not np.array_equal(raw,residual_raw(joblib.load(temp/'model.joblib'),frame.iloc[pred])):raise AssertionError('Residual raw replay differs')
        np.save(temp/'raw.npy',raw,allow_pickle=False)
        teacher_dependencies={q:sha(args.out/'teacher'/q/'complete.json') for q in train.forecast_period.iloc[fit].unique()}
        rec=dict(identity=identity,native_model_fit=bundle['native_model_fit'],reload_raw_exact=True,
            teacher_dependencies=teacher_dependencies,train_max_period=train.forecast_period.iloc[fit].max(),
            no_in_sample_teacher_baselines=True,original_binary_targets=True,AP_computed=False,
            native_baseline_max_cast_error=bundle.get('native_baseline_max_cast_error',0.))
        rec['sha256']={p.name:sha(p) for p in temp.iterdir()};write_json(temp/'complete.json',rec);os.rename(temp,path)
    print(json.dumps(dict(event='meta_complete',quarter=quarter,trees=count,seed=seed,native_model_fit=bundle['native_model_fit'])),flush=True)
    return bundle,raw,rec


def assess(args,meta,anchor,previous,teacher,residual):
    values={primary.ANCHOR:anchor,'previous_algorithm':previous,'fixed_teacher':teacher,
        'residual_raw':residual,'teacher50_residual50':.5*teacher+.5*residual}
    result=evaluate_candidates(meta,values,reference=primary.ANCHOR,bootstrap=args.bootstrap,seed=20261122)
    gates={};comparisons=[]
    for name in ['residual_raw','teacher50_residual50']:
        teacher_cmp=evaluate_candidates(meta,{'fixed_teacher':teacher,name:values[name]},reference='fixed_teacher',bootstrap=args.bootstrap,seed=20261122)
        previous_cmp=evaluate_candidates(meta,{'previous_algorithm':previous,name:values[name]},reference='previous_algorithm',bootstrap=args.bootstrap,seed=20261122)
        comparisons.extend([teacher_cmp['summary'].assign(reference='fixed_teacher'),previous_cmp['summary'].assign(reference='previous_algorithm')])
        c=result['summary'].loc[lambda f:f.model.eq(name)].iloc[0];t=teacher_cmp['summary'].loc[lambda f:f.model.eq(name)].iloc[0];p=previous_cmp['summary'].loc[lambda f:f.model.eq(name)].iloc[0]
        gates[name]=dict(eligible=bool(c.eligible and c.worst_quarter_delta>=-.005 and t.delta_vs_reference>0 and t.nonnegative_quarters>=2 and p.delta_vs_reference>0),mean_ap=float(c.mean_ap),delta_context=float(c.delta_vs_reference),delta_matched_teacher=float(t.delta_vs_reference),nonnegative_matched_teacher=int(t.nonnegative_quarters),delta_previous_algorithm=float(p.delta_vs_reference))
    write_csv(args.out/'outer-quarter-metrics.csv',result['quarter_scores']);write_csv(args.out/'outer-summary.csv',result['summary']);write_csv(args.out/'matched-comparisons.csv',pd.concat(comparisons,ignore_index=True));write_csv(args.out/'outer-oof.csv',pd.concat([meta,pd.DataFrame(values)],axis=1));write_json(args.out/'gates.json',gates)
    return gates


def run(args):
    args.threads=2
    if args.bootstrap<0 or not args.smoke and not args.bootstrap:raise ValueError('Positive bootstrap required')
    frozen=make_freeze(args);sig=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest();frozen['signature']=sig;freeze=args.out/'freeze.json'
    if freeze.exists():
        if not args.resume or read_json(freeze)!=frozen:raise ValueError('Changed sources/data/cache/teacher/control/plan/packages')
    else:
        if any(p.name not in {'.lock','plan.json','run.log'} for p in args.out.iterdir()):raise ValueError('New residual run requires empty output')
        if(args.out/'plan.json').exists() and sha(args.out/'plan.json')!=sha(args.plan):raise ValueError('Frozen plan copy differs')
        write_json(freeze,frozen)
    manifest_path=args.out/'manifest.json'
    if manifest_path.exists():
        previous_manifest=read_json(manifest_path)
        if previous_manifest['signature']!=sig or any(sha(args.out/n)!=h for n,h in previous_manifest['output_sha256'].items()):raise ValueError('Completed outputs changed')
    train,test,sample,views,cats=primary.validate_cache(joblib.load(args.feature_cache),args);frame=views['base'];teachers=Teachers(args,sig,frozen['teacher_sources'],train,test,frame)
    # Ordinary causal OOF meta-training values: no per-quarter target scores.
    periods=sorted(train.forecast_period.unique())
    for q in periods[1:]:teachers.ensure(q)
    if not np.isnan(teachers.oof[train.forecast_period.eq(periods[0])]).all() or not np.isfinite(teachers.oof[train.forecast_period.ne(periods[0])]).all():raise AssertionError('OOF teacher coverage incorrect')
    np.save(args.out/'teacher-oof-raw.npy',teachers.oof,allow_pickle=False)
    meta=pd.concat([train.loc[train.forecast_period.eq(q),list(primary.REQUIRED)] for q in primary.QUARTERS],ignore_index=True);anchor,anchor_test=primary.align_anchors(meta,test,args)
    control=read_csv(args.algorithm_control/'outer-oof.csv').set_index('row_id').loc[meta.row_id].reset_index();pd.testing.assert_frame_equal(meta,control[list(primary.REQUIRED)],check_dtype=False)
    if not np.array_equal(anchor,control.context_3seed.to_numpy()):raise ValueError('Algorithm context anchor differs')
    previous=primary.probabilities(control.algorithm_raw,len(meta));teacher_p=[];residual_p=[];counts=[];(args.out/'decisions').mkdir(exist_ok=True)
    for outer in primary.QUARTERS:
        inner=(pd.Period(outer,freq='Q')-1).start_time.strftime('%Y-%m-%d');inner_idx=np.flatnonzero(train.forecast_period.eq(inner));inner_bundle,_,inner_rec=meta_job(args,sig,inner,400,42,teachers,train,frame,cats)
        count,curve=choose_count(inner_bundle,frame.iloc[inner_idx],teachers.oof[inner_idx],train.target.iloc[inner_idx]);counts.append(count)
        decision=dict(outer=outer,inner=inner,selected_trees=int(count),curve=curve,meta_inner_sha256=sha(args.out/'meta'/f'{inner}__n400__s42'/'complete.json'))
        path=args.out/'decisions'/f'{outer}.json'
        if path.exists():
            if read_json(path)!=decision:raise ValueError('Earlier count choice changed')
        else:write_json(path,decision)
        bundle,raw,rec=meta_job(args,sig,outer,count,42,teachers,train,frame,cats);idx=np.flatnonzero(train.forecast_period.eq(outer));t=expit(teachers.oof[idx]);p=expit(teachers.oof[idx]+raw)
        if count==0 and not np.array_equal(t,p):raise AssertionError('Zero correction is not exactly teacher')
        teacher_p.append(t);residual_p.append(p)
        print(json.dumps(dict(event='outer_complete',quarter=outer,selected_trees=int(count),zero_is_exact_teacher=count==0)),flush=True)
    gates=assess(args,meta,anchor,previous,np.concatenate(teacher_p),np.concatenate(residual_p));final_count=int(np.median(counts));files=[];full_counts=[]
    if any(g['eligible'] for g in gates.values()) and final_count>0:
        teacher_test=teachers.ensure('full');seed_probabilities=[]
        for seed in [42,17,2026]:
            _,raw,rec=meta_job(args,sig,'full',final_count,seed,teachers,train,frame,cats);seed_probabilities.append(expit(teacher_test+raw));full_counts.append(seed)
        p=np.mean(seed_probabilities,axis=0,dtype=np.float64);t=expit(teacher_test)
        for variant,g in gates.items():
            if not g['eligible']:continue
            val=p if variant=='residual_raw' else .5*t+.5*p;submission=primary.validate_submission(test,sample,val,len(test) if args.smoke else 2246);file=args.out/f'submission_gam_residual_{variant}.csv';write_csv(file,submission);pd.testing.assert_frame_equal(submission,read_csv(file),check_exact=True);files.append(dict(variant=variant,csv=file.name,sha256=sha(file),rows=len(submission),final_trees=final_count,seeds=[42,17,2026]))
    outputs=[p for p in args.out.iterdir() if p.suffix in {'.json','.csv','.npy'} and p!=manifest_path]
    manifest=dict(signature=sig,complete=True,teacher_oof_quarters=periods[1:],earliest_meta_drop=periods[0],new_teacher_fits=sum(r['new_fit'] for r in teachers.records.values()),reused_teacher_checkpoints=sum(r['reused_checkpoint'] for r in teachers.records.values()),selected_outer_counts=counts,final_median_trees=final_count,final_seeds=full_counts,full_teacher_reused='full' in teachers.records,zero_median_duplicate_teacher=final_count==0,gates=gates,files=files,new_q2_q3_metrics=False,uploads=0,output_sha256={p.name:sha(p) for p in outputs});write_json(manifest_path,manifest);print(json.dumps(dict(event='complete',counts=counts,final_trees=final_count,gates=gates,files=files)),flush=True);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['raw','out','feature-cache','baseline-cv','context-test','algorithm-control']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan',type=Path,default=DEFAULT_PLAN);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with(args.out/'.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)


if __name__=='__main__':main()
