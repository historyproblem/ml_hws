"""Frozen12 interaction/recency recipes, previous-inner selection and gated CSVs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_hw4_final_research as primary
from hw4_inspections.additive_hybrid import CLASSIFIER,RECIPES,choose_final,choose_inner,fit_recipe,predict_recipe
from hw4_inspections.next_round import evaluate_candidates

DEFAULT_PLAN=ROOT/'docs/hw4-additive-hybrid-plan-20261122.json'
sha,read_json,write_json=primary.sha,primary.read_json,primary.write_json
read_csv,write_csv=primary.read_csv,primary.write_csv


def make_freeze(args):
    plan=read_json(args.plan)
    if plan['recipes']!=list(RECIPES) or plan['recipe_count']!=12:
        raise ValueError('Frozen hybrid recipes differ')
    if plan['optimizer']['threads']!=2 or any(CLASSIFIER[k]!=v for k,v in plan['optimizer'].items() if k!='threads'):
        raise ValueError('Frozen hybrid optimizer differs')
    freeze=primary.provenance(args,{r['id']:r for r in RECIPES})
    module=ROOT/'src/hw4_inspections'
    freeze['source_sha256'].update({n:sha(module/n) for n in [
        'additive_logit.py','classical_features.py','additive_search.py','additive_hybrid.py']})
    freeze['source_sha256'][Path(__file__).name]=sha(Path(__file__))
    control_manifest=read_json(args.algorithm_control/'manifest.json')
    control_freeze=read_json(args.algorithm_control/'freeze.json')
    claimed=control_freeze.pop('signature')
    if hashlib.sha256(json.dumps(control_freeze,sort_keys=True).encode()).hexdigest()!=claimed or control_manifest['signature']!=claimed:
        raise ValueError('Previous algorithm signature invalid')
    if (not control_manifest['complete'] or control_manifest['new_q2_q3_metrics'] is not False
            or control_manifest['outer_selection_of24recipes'] is not False):
        raise ValueError('Previous algorithm is not the frozen completed old-dev algorithm')
    for n,h in control_manifest['output_sha256'].items():
        if sha(args.algorithm_control/n)!=h:raise ValueError('Changed previous algorithm output')
    for key in ['raw_sha256','cache_sha256']:
        if control_freeze[key]!=freeze[key]:raise ValueError('Previous algorithm inputs differ')
    for n in ['additive_logit.py','classical_features.py','additive_search.py']:
        if control_freeze['source_sha256'][n]!=freeze['source_sha256'][n]:raise ValueError('Scientific control source differs')
    control=dict(oof_sha256=sha(args.algorithm_control/'outer-oof.csv'),
                 manifest_sha256=sha(args.algorithm_control/'manifest.json'),freeze_signature=claimed)
    if not args.smoke and (control['oof_sha256']!=plan['control']['previous_algorithm_oof_sha256']
            or control['manifest_sha256']!=plan['control']['previous_algorithm_manifest_sha256']):
        raise ValueError('Previous algorithm differs from pre-results plan')
    freeze.update(plan_sha256=sha(args.plan),pre_results_plan=plan,seeds=[],algorithm_control=control,
        tree_selection='No tree/epoch/seed selection: one previous-inner AP choice among12 fixed H/C recipes',
        selection='Outer predictions only of selected algorithm. Final majority/tie from inner scores, never best12outer')
    return freeze


def load_control(args,meta,anchor):
    control=read_csv(args.algorithm_control/'outer-oof.csv')
    if not control.row_id.is_unique or set(control.row_id)!=set(meta.row_id):raise ValueError('Control OOF IDs differ')
    control=control.set_index('row_id').loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(meta[list(primary.REQUIRED)],control[list(primary.REQUIRED)],check_dtype=False)
    if not np.array_equal(anchor,control.context_3seed.to_numpy()):raise ValueError('Control context anchor differs')
    p=primary.probabilities(control.algorithm_raw,len(meta))
    if not np.array_equal(control.algorithm_blend75.to_numpy(),.75*anchor+.25*p):raise ValueError('Control fixed blend differs')
    return p


def job(args,signature,name,quarter,role,train,frame):
    full=quarter=='full';fit=np.arange(len(train)) if full else np.flatnonzero(train.forecast_period.lt(quarter));pred=np.arange(len(train),len(frame)) if full else np.flatnonzero(train.forecast_period.eq(quarter))
    identity=dict(signature=signature,recipe=name,period=quarter,train_rows=len(fit),predict_rows=len(pred))
    if not len(fit) or not len(pred):raise ValueError('Empty fold cannot be omitted')
    path=args.out/'jobs'/f'{quarter}__{name}'
    if path.exists():
        rec=read_json(path/'complete.json')
        if rec['identity']!=identity or any(sha(path/n)!=h for n,h in rec['sha256'].items()):raise ValueError('Changed hybrid checkpoint')
        if rec['status']=='invalid':
            if set(rec['sha256'])!={'invalid-info.json'}:raise ValueError('Invalid numerical checkpoint schema')
            return rec,None
        if rec['status']!='complete' or rec['reload_exact'] is not True or set(rec['sha256'])!={'model.joblib','prediction.npy','fit-info.json'}:raise ValueError('Invalid completed checkpoint schema')
        p=np.load(path/'prediction.npy',allow_pickle=False)
        if p.dtype!=np.float64:raise ValueError('Prediction dtype differs')
        print(json.dumps(dict(event='job_reused',period=quarter,recipe=name,role=role)),flush=True)
        return rec,primary.probabilities(p,len(pred))
    print(json.dumps(dict(event='fit_start',**identity,role=role)),flush=True);started=time.monotonic();error=None
    try:
        bundle=fit_recipe(frame.iloc[fit],train.target.iloc[fit],{'id':name},periods=train.forecast_period.iloc[fit],reference='2024-10-01' if full else quarter)
        p=primary.probabilities(predict_recipe(bundle,frame.iloc[pred]),len(pred))
    except (ConvergenceWarning,FloatingPointError,np.linalg.LinAlgError) as exc:error=dict(type=type(exc).__name__,message=str(exc))
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent,prefix='.pending-hybrid-') as tmp:
        tmp=Path(tmp)/'job';tmp.mkdir();rec=dict(identity=identity,recipe=name,period=quarter,first_role=role,status='invalid' if error else 'complete',ap=None,seconds=time.monotonic()-started)
        if error:write_json(tmp/'invalid-info.json',error);rec.update(numerical_failure=error,reload_exact=False)
        else:
            joblib.dump(bundle,tmp/'model.joblib',compress=1)
            if not np.array_equal(p,predict_recipe(joblib.load(tmp/'model.joblib'),frame.iloc[pred])):raise AssertionError('Hybrid reload differs')
            np.save(tmp/'prediction.npy',p,allow_pickle=False);write_json(tmp/'fit-info.json',dict(recipe=bundle['recipe'],classifier=bundle['classifier'],weight_summary=bundle['weight_summary'],iterations=bundle['iterations'],features=bundle['feature_names'],threads=2))
            rec.update(reload_exact=True,iterations=bundle['iterations'],weight_summary=bundle['weight_summary'])
            if not full:rec['ap']=float(average_precision_score(train.target.iloc[pred],p))
        rec['sha256']={p.name:sha(p) for p in tmp.iterdir()};write_json(tmp/'complete.json',rec);os.rename(tmp,path)
    print(json.dumps(dict(event='job_complete',period=quarter,recipe=name,status=rec['status'],ap=rec['ap'],seconds=rec['seconds'])),flush=True)
    return rec,None if error else p


def assess(args,meta,anchor,previous,hybrid):
    values={primary.ANCHOR:anchor,'previous_raw':previous,'previous_blend75':.75*anchor+.25*previous,'hybrid_raw':hybrid,'hybrid_blend75':.75*anchor+.25*hybrid}
    result=evaluate_candidates(meta,values,reference=primary.ANCHOR,bootstrap=args.bootstrap,seed=20261122)
    gates={};paired=[]
    for variant in ['raw','blend75']:
        name,control=f'hybrid_{variant}',f'previous_{variant}'
        comparison=evaluate_candidates(meta,{control:values[control],name:values[name]},reference=control,bootstrap=args.bootstrap,seed=20261122)
        paired.append(comparison['summary'].assign(variant=variant));row=result['summary'].loc[lambda f:f.model.eq(name)].iloc[0];old=comparison['summary'].loc[lambda f:f.model.eq(name)].iloc[0]
        gates[variant]=dict(eligible=bool(row.eligible and row.worst_quarter_delta>=-.005 and old.delta_vs_reference>0 and old.nonnegative_quarters>=2),mean_ap=float(row.mean_ap),delta_context=float(row.delta_vs_reference),nonnegative_context=int(row.nonnegative_quarters),worst_context=float(row.worst_quarter_delta),delta_previous_algorithm=float(old.delta_vs_reference),nonnegative_previous_algorithm=int(old.nonnegative_quarters))
    write_csv(args.out/'outer-quarter-metrics.csv',result['quarter_scores']);write_csv(args.out/'outer-summary.csv',result['summary']);write_csv(args.out/'hybrid-vs-previous-algorithm.csv',pd.concat(paired,ignore_index=True));write_csv(args.out/'outer-oof.csv',pd.concat([meta,pd.DataFrame(values)],axis=1));write_json(args.out/'gates.json',gates)
    return gates


def run(args):
    args.threads=2
    if args.bootstrap<0 or not args.smoke and not args.bootstrap:raise ValueError('Positive bootstrap required;0smokeonly')
    freeze=make_freeze(args);sig=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();freeze['signature']=sig;path=args.out/'freeze.json'
    if path.exists():
        if not args.resume or read_json(path)!=freeze:raise ValueError('Changed source/plan/data/cache/control/environment')
    else:
        if any(p.name not in {'.lock','plan.json','run.log'} for p in args.out.iterdir()):raise ValueError('New run requires empty output')
        if (args.out/'plan.json').exists() and sha(args.out/'plan.json')!=sha(args.plan):raise ValueError('Plan copy differs')
        write_json(path,freeze)
    manifest_path=args.out/'manifest.json'
    if manifest_path.exists():
        old=read_json(manifest_path)
        if old['signature']!=sig or any(sha(args.out/n)!=h for n,h in old['output_sha256'].items()):raise ValueError('Completed output changed')
    train,test,sample,views,_=primary.validate_cache(joblib.load(args.feature_cache),args);frame=views['base'];meta=pd.concat([train.loc[train.forecast_period.eq(q),list(primary.REQUIRED)] for q in primary.QUARTERS],ignore_index=True);anchor,anchor_test=primary.align_anchors(meta,test,args);previous=load_control(args,meta,anchor)
    records=[];winners=[];outer=[];(args.out/'decisions').mkdir(exist_ok=True)
    for q in primary.QUARTERS:
        inner=(pd.Period(q,freq='Q')-1).start_time.strftime('%Y-%m-%d');rs=[job(args,sig,r['id'],inner,'inner',train,frame)[0] for r in RECIPES];winner=choose_inner(rs)
        decision=dict(outer=q,inner=inner,selected_recipe=winner,inner_complete_sha256={r['recipe']:sha(args.out/'jobs'/f"{inner}__{r['recipe']}"/'complete.json') for r in rs});dp=args.out/'decisions'/f'{q}.json'
        if dp.exists():
            if read_json(dp)!=decision:raise ValueError('Past inner choice changed')
        else:write_json(dp,decision)
        record,p=job(args,sig,winner,q,'hybrid_outer',train,frame)
        if record['status']!='complete':raise RuntimeError('Selected hybrid outer invalid; no quarter dropped')
        records.extend(rs);winners.append(winner);outer.append(p)
        print(json.dumps(dict(event='outer_complete',quarter=q,inner=inner,recipe=winner,ap=record['ap'])),flush=True)
    write_csv(args.out/'inner-records.csv',pd.DataFrame([dict(recipe=r['recipe'],quarter=r['period'],status=r['status'],ap=r['ap'],numerical_failure=json.dumps(r.get('numerical_failure'),sort_keys=True)) for r in records]))
    gates=assess(args,meta,anchor,previous,np.concatenate(outer));final=choose_final(winners,records);write_json(args.out/'final-recipe.json',dict(inner_winners=winners,selected_recipe=final,uses_outer_recipe_scores=False));files=[]
    if any(g['eligible'] for g in gates.values()):
        rec,p=job(args,sig,final,'full','gated_full',train,frame)
        if rec['status']!='complete':raise RuntimeError('Gated final failed; no test-based replacement')
        for variant,g in gates.items():
            if not g['eligible']:continue
            prediction=p if variant=='raw' else .75*anchor_test+.25*p;csv=primary.validate_submission(test,sample,prediction,len(test) if args.smoke else 2246);file=args.out/f'submission_additive_hybrid_{variant}.csv';write_csv(file,csv);pd.testing.assert_frame_equal(csv,read_csv(file),check_exact=True);files.append(dict(variant=variant,csv=file.name,sha256=sha(file),rows=len(csv),recipe=final))
    outputs=[p for p in args.out.iterdir() if p.suffix in {'.json','.csv'} and p!=manifest_path]
    manifest=dict(signature=sig,complete=True,inner_candidate_jobs=36,inner_winners=winners,final_recipe=final,gates=gates,files=files,new_q2_q3_metrics=False,uploads=0,outer_bestof12=False,output_sha256={p.name:sha(p) for p in outputs});write_json(manifest_path,manifest);print(json.dumps(dict(event='complete',gates=gates,final_recipe=final,files=files)),flush=True);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['raw','out','feature-cache','baseline-cv','context-test','algorithm-control']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan',type=Path,default=DEFAULT_PLAN);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with(args.out/'.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)


if __name__=='__main__':main()
