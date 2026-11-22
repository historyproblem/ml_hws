"""One prequential positive2logit stacker; reuse bases and never score recent q."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_hw4_final_research as primary
from hw4_inspections.classical_models import SPECS
from hw4_inspections.gam_residual import TEACHER_RECIPE,teacher_raw
from hw4_inspections.next_round import evaluate_candidates
from hw4_inspections.stacking import (
    EPSILON,LAMBDA,OPTIMIZER,REFERENCE_SEEDS,fit_reference,fit_stacker,
    meta_features,predict_reference,predict_stacker,
)

DEFAULT_PLAN=ROOT/'docs/hw4-prequential-stacking-plan-20261122.json'
EARLY=('2022-04-01','2022-07-01','2022-10-01','2023-01-01','2023-04-01')
RECENT=('2024-04-01','2024-07-01')
sha,read_json,write_json=primary.sha,primary.read_json,primary.write_json
read_csv,write_csv=primary.read_csv,primary.write_csv


def make_freeze(args):
    plan=read_json(args.plan)
    if (plan['meta']['lambda_']!=LAMBDA or plan['inputs']['CB_clip_epsilon']!=EPSILON
            or plan['new_base_fits']['quarters']!=list(EARLY)):
        raise ValueError('Frozen stacker algorithm differs')
    frozen=primary.provenance(args,{'stacker':dict(lambda_=LAMBDA,epsilon=EPSILON,optimizer=OPTIMIZER)})
    module=ROOT/'src/hw4_inspections'
    scientific=['additive_logit.py','classical_features.py','classical_models.py','additive_search.py','gam_residual.py','stacking.py']
    frozen['source_sha256'].update({n:sha(module/n) for n in scientific});frozen['source_sha256'][Path(__file__).name]=sha(Path(__file__))
    library_manifest=read_json(args.teacher_library/'manifest.json');library_freeze=read_json(args.teacher_library/'freeze.json')
    claimed=library_freeze.pop('signature')
    if hashlib.sha256(json.dumps(library_freeze,sort_keys=True).encode()).hexdigest()!=claimed or library_manifest['signature']!=claimed or not library_manifest['complete']:
        raise ValueError('GAM OOF library signature differs')
    if library_freeze['raw_sha256']!=frozen['raw_sha256'] or library_freeze['cache_sha256']!=frozen['cache_sha256']:
        raise ValueError('GAM library inputs differ')
    for n in ['additive_logit.py','classical_features.py','additive_search.py','gam_residual.py']:
        if library_freeze['source_sha256'][n]!=frozen['source_sha256'][n]:raise ValueError('GAM scientific source differs')
    if not args.smoke and sha(args.teacher_library/'manifest.json')!=plan['provenance']['GAM_teacher_manifest_sha256']:
        raise ValueError('GAM library differs from frozen plan')
    for n,h in library_manifest['output_sha256'].items():
        if sha(args.teacher_library/n)!=h:raise ValueError('GAM library output changed')
    teacher_sources={}
    for q in sorted(pd.read_csv(args.raw/'train.csv').forecast_period.unique())[1:]:
        path=args.teacher_library/'teacher'/q;rec=read_json(path/'complete.json')
        if (rec['identity']['signature']!=claimed or rec['identity']['quarter']!=q
                or rec['evaluation_metrics_computed'] is not False or not rec['train_max_period']<q):
            raise ValueError('Teacher OOF is not strictpast/unscored')
        if any(sha(path/n)!=h for n,h in rec['sha256'].items()):raise ValueError('Teacher OOF artifact changed')
        teacher_sources[q]=dict(path=str(path.resolve()),marker_sha256=sha(path/'complete.json'),artifacts=rec['sha256'])
    full_teacher=library_freeze['teacher_sources']['full']
    if sha(Path(full_teacher['path'])/'complete.json')!=full_teacher['complete_sha256'] or sha(Path(full_teacher['path'])/'model.joblib')!=full_teacher['artifacts']['model.joblib']:
        raise ValueError('Fixed full GAM teacher source changed')
    delivered=read_json(args.recent_components/'frozen_recipe.json');delivery_signature=delivered.pop('signature');delivered.pop('frozen_before_holdout_scoring',None)
    if hashlib.sha256(json.dumps(delivered,sort_keys=True).encode()).hexdigest()!=delivery_signature:
        raise ValueError('Existing recent reference delivery signature differs')
    if (delivered['input_sha256']!=frozen['raw_sha256'] or delivered['catboost_specs']['cb_reference']!=SPECS['cb_reference']):
        raise ValueError('Existing recent reference data/spec differs')
    for n in ['features.py','extensions.py','classical_features.py','classical_models.py']:
        if delivered['source_sha256'][f'src/hw4_inspections/{n}']!=sha(module/n):raise ValueError('Existing reference feature/model source differs')
    if not args.smoke and delivery_signature!=plan['base_reuse']['recent_delivery_signature']:
        raise ValueError('Existing recent source differs from frozen plan')
    recent_sources={}
    for q in RECENT:
        recent_sources[q]={}
        for seed in REFERENCE_SEEDS:
            path=args.recent_components/f'holdout_{q}'/f'cb_reference_s{seed}';rec=read_json(path/'complete.json')
            if (rec['signature']!=delivery_signature or rec['stage']!=f'holdout_{q}' or rec['model']!='cb_reference'
                    or rec['seed']!=seed or rec['loaded_forward_exact'] is not True):raise ValueError('Recent reference job identity differs')
            if any(sha(path/n)!=h for n,h in rec['artifacts'].items()):raise ValueError('Recent reference artifact changed')
            recent_sources[q][str(seed)]=dict(path=str(path.resolve()),marker_sha256=sha(path/'complete.json'),artifacts=rec['artifacts'],train_rows=rec['train_rows'])
    control_manifest=read_json(args.algorithm_control/'manifest.json')
    if not control_manifest['complete']:raise ValueError('Previous algorithm incomplete')
    if not args.smoke and sha(args.algorithm_control/'outer-oof.csv')!=plan['provenance']['previous_algorithm_oof_sha256']:
        raise ValueError('Previous algorithm OOF differs')
    frozen.update(plan_sha256=sha(args.plan),pre_results_plan=plan,seeds=list(REFERENCE_SEEDS),
        tree_selection='None. Fixed500-tree original reference only; ONE fixed3parameter convex stacker.',
        selection='No HPO/outer choice; fit meta only on strictly past chronological OOF inputs',
        teacher_library_signature=claimed,teacher_sources=teacher_sources,full_teacher_source=full_teacher,
        recent_delivery_signature=delivery_signature,recent_reference_sources=recent_sources,
        previous_algorithm_oof_sha256=sha(args.algorithm_control/'outer-oof.csv'))
    return frozen


def cb_early_job(args,sig,q,seed,train,frame,cats):
    fit=np.flatnonzero(train.forecast_period.lt(q));pred=np.flatnonzero(train.forecast_period.eq(q));trees=5 if args.smoke else 500
    identity=dict(signature=sig,quarter=q,seed=seed,iterations=trees,train_rows=len(fit),predict_rows=len(pred),family='cb_reference_context346')
    path=args.out/'early-cb'/f'{q}__s{seed}'
    if path.exists():
        rec=read_json(path/'complete.json')
        if rec['identity']!=identity or rec['reload_exact'] is not True or any(sha(path/n)!=h for n,h in rec['sha256'].items()):raise ValueError('Early reference checkpoint changed')
        return primary.probabilities(np.load(path/'prediction.npy',allow_pickle=False),len(pred))
    print(json.dumps(dict(event='early_cb_start',**identity)),flush=True)
    bundle=fit_reference(frame.iloc[fit],train.target.iloc[fit],cats,seed,iterations=trees);p=predict_reference(bundle,frame.iloc[pred])
    actual=bundle['model'].get_all_params()
    if actual['depth']!=5 or abs(actual['learning_rate']-.04)>1e-7 or actual['l2_leaf_reg']!=10 or actual['boosting_type']!='Plain' or actual['bootstrap_type']!='MVS':raise ValueError('Not exact original reference family')
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent,prefix='.pending-reference-') as temp:
        temp=Path(temp)/'job';temp.mkdir();joblib.dump(bundle,temp/'model.joblib',compress=1)
        if not np.array_equal(p,predict_reference(joblib.load(temp/'model.joblib'),frame.iloc[pred])):raise AssertionError('Early CB replay differs')
        np.save(temp/'prediction.npy',p,allow_pickle=False);np.save(temp/'row-ids.npy',train.row_id.iloc[pred].to_numpy(str),allow_pickle=False)
        rec=dict(identity=identity,reload_exact=True,new_fit=True,train_max_period=train.forecast_period.iloc[fit].max(),prediction_metrics_computed=False,actual_params=actual)
        rec['sha256']={p.name:sha(p) for p in temp.iterdir()};write_json(temp/'complete.json',rec);os.rename(temp,path)
    print(json.dumps(dict(event='early_cb_complete',quarter=q,seed=seed,AP_computed=False)),flush=True)
    return p


def components(args,frozen,sig,train,test,frame,cats):
    cb=np.full(len(train),np.nan);gam=np.full(len(train),np.nan);periods=sorted(train.forecast_period.unique())
    for q in periods[1:]:
        idx=np.flatnonzero(train.forecast_period.eq(q));source=frozen['teacher_sources'][q];path=Path(source['path']);rec=read_json(path/'complete.json')
        if (sha(path/'complete.json')!=source['marker_sha256']
                or any(sha(path/n)!=h for n,h in source['artifacts'].items())
                or rec['identity']['train_rows']!=int(train.forecast_period.lt(q).sum()) or rec['identity']['predict_rows']!=len(idx)):
            raise ValueError('GAM strictpast partition or raw source differs')
        ids=np.load(path/'row-ids.npy',allow_pickle=False);value=np.load(path/'raw.npy',allow_pickle=False)
        if not np.array_equal(ids,train.row_id.iloc[idx].to_numpy(str)) or value.dtype!=np.float64 or value.shape!=(len(idx),) or not np.isfinite(value).all():raise ValueError('Teacher raw data/IDs differ')
        gam[idx]=value
        if q in EARLY:
            cb[idx]=np.mean([cb_early_job(args,sig,q,seed,train,frame,cats) for seed in REFERENCE_SEEDS],axis=0,dtype=np.float64)
        elif q in primary.QUARTERS:
            baseline=read_csv(args.baseline_cv).set_index('row_id').loc[train.row_id.iloc[idx]]
            for name in ['site_id','forecast_period','target']:
                if not np.array_equal(baseline[name].to_numpy(),train[name].iloc[idx].to_numpy()):raise ValueError('Old reference OOF identity differs')
            cb[idx]=primary.probabilities(baseline.prediction,len(idx))
        elif q in RECENT:
            values=[]
            for seed in REFERENCE_SEEDS:
                source=frozen['recent_reference_sources'][q][str(seed)];path=Path(source['path'])
                if (sha(path/'complete.json')!=source['marker_sha256']
                        or sha(path/'prediction.npy')!=source['artifacts']['prediction.npy']
                        or source['train_rows']!=int(train.forecast_period.lt(q).sum())):
                    raise ValueError('Recent strictpast source differs')
                value=np.load(path/'prediction.npy',allow_pickle=False)
                values.append(primary.probabilities(value,len(idx)))
            cb[idx]=np.mean(values,axis=0,dtype=np.float64)
        else:raise ValueError('Unexpected OOF quarter')
    if not np.isnan(cb[train.forecast_period.eq(periods[0])]).all() or not np.isnan(gam[train.forecast_period.eq(periods[0])]).all():raise AssertionError('Earliest quarter cannot have causal OOF')
    valid=train.forecast_period.ne(periods[0]).to_numpy()
    if not np.isfinite(cb[valid]).all() or not np.isfinite(gam[valid]).all():raise ValueError('OOF coverage incomplete')
    np.save(args.out/'cb-oof-mean3.npy',cb,allow_pickle=False);np.save(args.out/'gam-oof-raw.npy',gam,allow_pickle=False)
    return cb,gam


def meta_job(args,sig,q,train,test,cb,gam,full_cb=None,full_gam=None):
    valid=np.isfinite(cb)&np.isfinite(gam);full=q=='full';fit=np.flatnonzero(valid if full else valid&train.forecast_period.lt(q).to_numpy());pred=np.arange(len(test)) if full else np.flatnonzero(train.forecast_period.eq(q))
    cb_pred,gam_pred=(full_cb,full_gam) if full else (cb[pred],gam[pred])
    identity=dict(signature=sig,quarter=q,train_rows=len(fit),predict_rows=len(pred),
        meta_input_sha256=hashlib.sha256(meta_features(cb[fit],gam[fit]).tobytes()).hexdigest())
    path=args.out/'meta'/q
    if path.exists():
        rec=read_json(path/'complete.json')
        if rec['identity']!=identity or rec['reload_exact'] is not True or set(rec['sha256'])!={'model.joblib','prediction.npy','row-ids.npy'} or any(sha(path/n)!=h for n,h in rec['sha256'].items()):raise ValueError('Stacker checkpoint changed')
        return primary.probabilities(np.load(path/'prediction.npy',allow_pickle=False),len(pred)),rec
    print(json.dumps(dict(event='meta_start',**identity)),flush=True)
    bundle=fit_stacker(cb[fit],gam[fit],train.target.iloc[fit]);p=predict_stacker(bundle,cb_pred,gam_pred)
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent,prefix='.pending-stacker-') as temp:
        temp=Path(temp)/'job';temp.mkdir();joblib.dump(bundle,temp/'model.joblib')
        if not np.array_equal(p,predict_stacker(joblib.load(temp/'model.joblib'),cb_pred,gam_pred)):raise AssertionError('Stacker replay differs')
        np.save(temp/'prediction.npy',p,allow_pickle=False);np.save(temp/'row-ids.npy',test.row_id.to_numpy(str) if full else train.row_id.iloc[pred].to_numpy(str),allow_pickle=False)
        rec=dict(identity=identity,reload_exact=True,train_max_period=train.forecast_period.iloc[fit].max(),prediction_metrics_computed=False,
            weights=bundle['weights'].tolist(),intercept=bundle['intercept'],active_inputs=bundle['active_inputs'],
            optimizer=bundle['optimizer'],projected_gradient_inf=bundle['projected_gradient_inf'],solver_message=bundle['solver_message'],iterations=bundle['iterations'])
        rec['sha256']={p.name:sha(p) for p in temp.iterdir()};write_json(temp/'complete.json',rec);os.rename(temp,path)
    print(json.dumps(dict(event='meta_complete',quarter=q,weights=rec['weights'],intercept=rec['intercept'],AP_computed=False)),flush=True)
    return p,rec


def assess(args,meta,anchor,previous,teacher,stacker):
    values={primary.ANCHOR:anchor,'previous_algorithm':previous,'matched_teacher':teacher,'positive_logit_stacker':stacker}
    result=evaluate_candidates(meta,values,reference=primary.ANCHOR,bootstrap=args.bootstrap,seed=20261122)
    matched=evaluate_candidates(meta,{'matched_teacher':teacher,'positive_logit_stacker':stacker},reference='matched_teacher',bootstrap=args.bootstrap,seed=20261122)
    old=evaluate_candidates(meta,{'previous_algorithm':previous,'positive_logit_stacker':stacker},reference='previous_algorithm',bootstrap=args.bootstrap,seed=20261122)
    c=result['summary'].loc[lambda f:f.model.eq('positive_logit_stacker')].iloc[0];t=matched['summary'].loc[lambda f:f.model.eq('positive_logit_stacker')].iloc[0];p=old['summary'].loc[lambda f:f.model.eq('positive_logit_stacker')].iloc[0]
    gate=dict(eligible=bool(c.eligible and c.worst_quarter_delta>=-.005 and t.delta_vs_reference>0 and t.nonnegative_quarters>=2 and t.worst_quarter_delta>=-.005 and p.delta_vs_reference>0),mean_ap=float(c.mean_ap),delta_context=float(c.delta_vs_reference),delta_matched_teacher=float(t.delta_vs_reference),nonnegative_teacher=int(t.nonnegative_quarters),worst_teacher=float(t.worst_quarter_delta),delta_previous_algorithm=float(p.delta_vs_reference))
    write_csv(args.out/'outer-quarter-metrics.csv',result['quarter_scores']);write_csv(args.out/'outer-summary.csv',result['summary']);write_csv(args.out/'matched-comparisons.csv',pd.concat([matched['summary'].assign(reference='matched_teacher'),old['summary'].assign(reference='previous_algorithm')],ignore_index=True));write_csv(args.out/'outer-oof.csv',pd.concat([meta,pd.DataFrame(values)],axis=1));write_json(args.out/'gate.json',gate)
    return gate


def run(args):
    args.threads=2
    if args.bootstrap<0 or not args.smoke and not args.bootstrap:raise ValueError('Positive bootstrap required')
    frozen=make_freeze(args);sig=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest();frozen['signature']=sig;freeze=args.out/'freeze.json'
    if freeze.exists():
        if not args.resume or read_json(freeze)!=frozen:raise ValueError('Changed source/plan/data/cache/reused-base/library')
    else:
        if any(p.name not in {'.lock','plan.json','run.log'} for p in args.out.iterdir()):raise ValueError('New run requires empty output')
        if(args.out/'plan.json').exists() and sha(args.out/'plan.json')!=sha(args.plan):raise ValueError('Frozen plan copy differs')
        write_json(freeze,frozen)
    manifest_path=args.out/'manifest.json'
    if manifest_path.exists():
        old=read_json(manifest_path)
        if old['signature']!=sig or any(sha(args.out/n)!=h for n,h in old['output_sha256'].items()):raise ValueError('Completed output changed')
    train,test,sample,views,cats=primary.validate_cache(joblib.load(args.feature_cache),args);frame=views['base'];cb,gam=components(args,frozen,sig,train,test,frame,cats)
    meta=pd.concat([train.loc[train.forecast_period.eq(q),list(primary.REQUIRED)] for q in primary.QUARTERS],ignore_index=True);anchor,anchor_test=primary.align_anchors(meta,test,args);control=read_csv(args.algorithm_control/'outer-oof.csv').set_index('row_id').loc[meta.row_id].reset_index();pd.testing.assert_frame_equal(meta,control[list(primary.REQUIRED)],check_dtype=False)
    prediction=[];teacher=[]
    for q in primary.QUARTERS:
        p,_=meta_job(args,sig,q,train,test,cb,gam);prediction.append(p);teacher.append(expit(gam[train.forecast_period.eq(q).to_numpy()]))
    gate=assess(args,meta,anchor,control.algorithm_raw.to_numpy(),np.concatenate(teacher),np.concatenate(prediction))
    # Only prediction, never AP: these fixed-estimator jobs are for root's later
    # one-shot recent framework and cannot affect this branch's hyperparameters.
    for q in RECENT:meta_job(args,sig,q,train,test,cb,gam)
    files=[]
    if gate['eligible']:
        source=frozen['full_teacher_source'];path=Path(source['path'])
        if sha(path/'model.joblib')!=source['artifacts']['model.joblib']:raise ValueError('Fullteacher changed')
        gam_test=teacher_raw(joblib.load(path/'model.joblib'),frame.iloc[len(train):]);p,rec=meta_job(args,sig,'full',train,test,cb,gam,anchor_test,gam_test)
        csv=primary.validate_submission(test,sample,p,len(test) if args.smoke else 2246);file=args.out/'submission_prequential_positive_stacker.csv';write_csv(file,csv);pd.testing.assert_frame_equal(csv,read_csv(file),check_exact=True);files=[dict(csv=file.name,sha256=sha(file),rows=len(csv),weights=rec['weights'],intercept=rec['intercept'])]
    outputs=[p for p in args.out.iterdir() if p.suffix in {'.json','.csv','.npy'} and p!=manifest_path]
    manifest=dict(signature=sig,complete=True,new_early_CB_fits=15,reused_old_CB_OOF_quarters=list(primary.QUARTERS),reused_recent_CB_components=6,reused_GAM_OOF_quarters=10,new_full_base_fits=0,recent_meta_predictions_without_AP=list(RECENT),new_q2_q3_metrics=False,uploads=0,gate=gate,files=files,output_sha256={p.name:sha(p) for p in outputs});write_json(manifest_path,manifest);print(json.dumps(dict(event='complete',gate=gate,files=files)),flush=True);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['raw','out','feature-cache','baseline-cv','context-test','teacher-library','recent-components','algorithm-control']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan',type=Path,default=DEFAULT_PLAN);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with(args.out/'.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)


if __name__=='__main__':main()
