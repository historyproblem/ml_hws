"""Eight frozen small state-GAM recipes, previous-inner selection and finals."""
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_hw4_final_research as primary
from hw4_inspections.gam_state import CLASSIFIER,RECIPES,choose_final,choose_inner,fit_recipe,predict_recipe
from hw4_inspections.next_round import evaluate_candidates

DEFAULT_PLAN=ROOT/'docs/hw4-compact-state-plan-20261122.json'
sha,read_json,write_json=primary.sha,primary.read_json,primary.write_json
read_csv,write_csv=primary.read_csv,primary.write_csv


def make_freeze(args):
    plan=read_json(args.plan)
    if plan['recipes']!=list(RECIPES) or plan['recipe_count']!=8:raise ValueError('Frozen state recipes differ')
    if plan['optimizer']['threads']!=2 or any(CLASSIFIER[k]!=v for k,v in plan['optimizer'].items() if k!='threads'):raise ValueError('Frozen state optimizer differs')
    f=primary.provenance(args,{r['id']:r for r in RECIPES});module=ROOT/'src/hw4_inspections'
    f['source_sha256'].update({n:sha(module/n) for n in ['additive_logit.py','classical_features.py','additive_search.py','gam_state.py']});f['source_sha256'][Path(__file__).name]=sha(Path(__file__))
    benchmark=read_json(args.benchmark/'manifest.json');bf=read_json(args.benchmark/'freeze.json');claimed=bf.pop('signature')
    if hashlib.sha256(json.dumps(bf,sort_keys=True).encode()).hexdigest()!=claimed or benchmark['signature']!=claimed or not benchmark['complete'] or benchmark['new_q2_q3_metrics'] is not False:raise ValueError('Benchmark signature/status differs')
    for n,h in benchmark['output_sha256'].items():
        if sha(args.benchmark/n)!=h:raise ValueError('Benchmark output changed')
    for key in ['raw_sha256','cache_sha256']:
        if bf[key]!=f[key]:raise ValueError('Benchmark inputs differ')
    for n in ['additive_logit.py','classical_features.py','additive_search.py']:
        if bf['source_sha256'][n]!=f['source_sha256'][n]:raise ValueError('Fixed E scientific source differs')
    source=dict(oof_sha256=sha(args.benchmark/'outer-oof.csv'),manifest_sha256=sha(args.benchmark/'manifest.json'))
    if not args.smoke and (source['oof_sha256']!=plan['controls']['oof_sha256'] or source['manifest_sha256']!=plan['controls']['manifest_sha256']):raise ValueError('Benchmark differs from frozen plan')
    f.update(plan_sha256=sha(args.plan),pre_results_plan=plan,seeds=[],benchmark=source,
        tree_selection='No tree/epoch/seed grid; eight state representations/C recipes previous-inner AP only',
        selection='Only sequentially selected algorithm gets outer AP; final majority/tie of inner outcomes')
    return f


def job(args,sig,name,q,role,train,frame):
    full=q=='full';ti=np.arange(len(train)) if full else np.flatnonzero(train.forecast_period.lt(q));vi=np.arange(len(train),len(frame)) if full else np.flatnonzero(train.forecast_period.eq(q));identity=dict(signature=sig,recipe=name,quarter=q,train_rows=len(ti),predict_rows=len(vi));path=args.out/'jobs'/f'{q}__{name}'
    if not len(ti) or not len(vi):raise ValueError('Empty fold cannot be omitted')
    if path.exists():
        r=read_json(path/'complete.json')
        if r['identity']!=identity or any(sha(path/n)!=h for n,h in r['sha256'].items()):raise ValueError('State checkpoint changed')
        if r['status']=='invalid':
            if set(r['sha256'])!={'invalid-info.json'}:raise ValueError('Invalid numerical schema')
            return r,None
        if r['status']!='complete' or r['reload_exact'] is not True or set(r['sha256'])!={'model.joblib','prediction.npy','row-ids.npy'}:raise ValueError('Invalid completed state checkpoint')
        value=np.load(path/'prediction.npy',allow_pickle=False)
        expected_ids=read_csv(args.raw/'test_student.csv').row_id.to_numpy(str) if full else train.row_id.iloc[vi].to_numpy(str)
        if value.dtype!=np.float64 or not np.array_equal(np.load(path/'row-ids.npy',allow_pickle=False),expected_ids):raise ValueError('State prediction dtype/IDs differ')
        return r,primary.probabilities(value,len(vi))
    print(json.dumps(dict(event='fit_start',**identity,role=role)),flush=True);error=None
    try:bundle=fit_recipe(frame.iloc[ti],train.target.iloc[ti],{'id':name});p=primary.probabilities(predict_recipe(bundle,frame.iloc[vi]),len(vi))
    except (ConvergenceWarning,FloatingPointError,np.linalg.LinAlgError) as exc:error=dict(type=type(exc).__name__,message=str(exc))
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent,prefix='.pending-state-') as temp:
        temp=Path(temp)/'job';temp.mkdir();r=dict(identity=identity,recipe=name,quarter=q,first_role=role,status='invalid' if error else 'complete',ap=None)
        if error:write_json(temp/'invalid-info.json',error);r.update(numerical_failure=error,reload_exact=False)
        else:
            joblib.dump(bundle,temp/'model.joblib',compress=1)
            if not np.array_equal(p,predict_recipe(joblib.load(temp/'model.joblib'),frame.iloc[vi])):raise AssertionError('State reload differs')
            np.save(temp/'prediction.npy',p,allow_pickle=False)
            ids=read_csv(args.raw/'test_student.csv').row_id.to_numpy(str) if full else train.row_id.iloc[vi].to_numpy(str)
            np.save(temp/'row-ids.npy',ids,allow_pickle=False);r.update(reload_exact=True,features=len(bundle['features']),iterations=bundle['iterations'])
            if not full:r['ap']=float(average_precision_score(train.target.iloc[vi],p))
        r['sha256']={p.name:sha(p) for p in temp.iterdir()};write_json(temp/'complete.json',r);os.rename(temp,path)
    print(json.dumps(dict(event='job_complete',quarter=q,recipe=name,status=r['status'],ap=r['ap'])),flush=True);return r,None if error else p


def assess(args,meta,anchor,teacher,previous,new):
    values={primary.ANCHOR:anchor,'matched_E_raw':teacher,'previous_raw':previous,'state_raw':new,
        'matched_E_blend75':.75*anchor+.25*teacher,'previous_blend75':.75*anchor+.25*previous,'state_blend75':.75*anchor+.25*new}
    result=evaluate_candidates(meta,values,reference=primary.ANCHOR,bootstrap=args.bootstrap,seed=20261122);gates={};comparisons=[]
    for variant in ['raw','blend75']:
        name,matched,old=f'state_{variant}',f'matched_E_{variant}',f'previous_{variant}'
        t=evaluate_candidates(meta,{matched:values[matched],name:values[name]},reference=matched,bootstrap=args.bootstrap,seed=20261122)
        p=evaluate_candidates(meta,{old:values[old],name:values[name]},reference=old,bootstrap=args.bootstrap,seed=20261122)
        comparisons.extend([t['summary'].assign(reference=matched),p['summary'].assign(reference=old)]);c=result['summary'].loc[lambda f:f.model.eq(name)].iloc[0];a=t['summary'].loc[lambda f:f.model.eq(name)].iloc[0];b=p['summary'].loc[lambda f:f.model.eq(name)].iloc[0]
        gates[variant]=dict(eligible=bool(c.eligible and c.worst_quarter_delta>=-.005 and a.delta_vs_reference>0 and a.nonnegative_quarters>=2 and a.worst_quarter_delta>=-.005 and b.delta_vs_reference>0),mean_ap=float(c.mean_ap),delta_context=float(c.delta_vs_reference),delta_matched_E=float(a.delta_vs_reference),nonnegative_matched_E=int(a.nonnegative_quarters),worst_matched_E=float(a.worst_quarter_delta),delta_previous_algorithm=float(b.delta_vs_reference))
    write_csv(args.out/'outer-quarter-metrics.csv',result['quarter_scores']);write_csv(args.out/'outer-summary.csv',result['summary']);write_csv(args.out/'matched-comparisons.csv',pd.concat(comparisons,ignore_index=True));write_csv(args.out/'outer-oof.csv',pd.concat([meta,pd.DataFrame(values)],axis=1));write_json(args.out/'gates.json',gates);return gates


def run(args):
    args.threads=2
    if args.bootstrap<0 or not args.smoke and not args.bootstrap:raise ValueError('Positive bootstrap required')
    f=make_freeze(args);sig=hashlib.sha256(json.dumps(f,sort_keys=True).encode()).hexdigest();f['signature']=sig;path=args.out/'freeze.json'
    if path.exists():
        if not args.resume or read_json(path)!=f:raise ValueError('Changed source/data/cache/plan/benchmark/environment')
    else:
        if any(p.name not in {'.lock','plan.json','run.log'} for p in args.out.iterdir()):raise ValueError('New run requires empty output')
        if(args.out/'plan.json').exists() and sha(args.out/'plan.json')!=sha(args.plan):raise ValueError('Plan copy differs')
        write_json(path,f)
    manifest_path=args.out/'manifest.json'
    if manifest_path.exists():
        m=read_json(manifest_path)
        if m['signature']!=sig or any(sha(args.out/n)!=h for n,h in m['output_sha256'].items()):raise ValueError('Completed outputs changed')
    train,test,sample,views,cats=primary.validate_cache(joblib.load(args.feature_cache),args);frame=views['base'];meta=pd.concat([train.loc[train.forecast_period.eq(q),list(primary.REQUIRED)] for q in primary.QUARTERS],ignore_index=True);anchor,anchor_test=primary.align_anchors(meta,test,args)
    source=read_csv(args.benchmark/'outer-oof.csv')
    if not source.row_id.is_unique or set(source.row_id)!=set(meta.row_id):raise ValueError('Benchmark OOF IDs differ')
    source=source.set_index('row_id').loc[meta.row_id].reset_index();pd.testing.assert_frame_equal(meta,source[list(primary.REQUIRED)],check_dtype=False)
    if not np.array_equal(anchor,source.context_3seed.to_numpy()):raise ValueError('Benchmark context anchor differs')
    teacher=primary.probabilities(source.fixed_teacher,len(meta));previous=primary.probabilities(source.previous_algorithm,len(meta));records=[];winners=[];predictions=[];(args.out/'decisions').mkdir(exist_ok=True)
    for outer in primary.QUARTERS:
        inner=(pd.Period(outer,freq='Q')-1).start_time.strftime('%Y-%m-%d');rs=[job(args,sig,r['id'],inner,'inner',train,frame)[0] for r in RECIPES];winner=choose_inner(rs);decision=dict(outer=outer,inner=inner,recipe=winner,inner_complete_sha256={r['recipe']:sha(args.out/'jobs'/f"{inner}__{r['recipe']}"/'complete.json') for r in rs});dp=args.out/'decisions'/f'{outer}.json'
        if dp.exists():
            if read_json(dp)!=decision:raise ValueError('Past decision changed')
        else:write_json(dp,decision)
        rec,p=job(args,sig,winner,outer,'state_outer',train,frame)
        if rec['status']!='complete':raise RuntimeError('Selected outer invalid; no fold omitted')
        records.extend(rs);winners.append(winner);predictions.append(p)
        print(json.dumps(dict(event='outer_complete',quarter=outer,recipe=winner,ap=rec['ap'])),flush=True)
    write_csv(args.out/'inner-records.csv',pd.DataFrame([dict(recipe=r['recipe'],quarter=r['quarter'],status=r['status'],ap=r['ap'],numerical_failure=json.dumps(r.get('numerical_failure'),sort_keys=True)) for r in records]));gates=assess(args,meta,anchor,teacher,previous,np.concatenate(predictions));final=choose_final(winners,records);write_json(args.out/'final-recipe.json',dict(inner_winners=winners,selected_recipe=final,uses_outer_scores=False));files=[]
    if any(g['eligible'] for g in gates.values()):
        rec,p=job(args,sig,final,'full','gated_full',train,frame)
        if rec['status']!='complete':raise RuntimeError('Final state fit invalid; no test-based replacement')
        for variant,g in gates.items():
            if not g['eligible']:continue
            val=p if variant=='raw' else .75*anchor_test+.25*p;csv=primary.validate_submission(test,sample,val,len(test) if args.smoke else 2246);file=args.out/f'submission_compact_state_{variant}.csv';write_csv(file,csv);pd.testing.assert_frame_equal(csv,read_csv(file),check_exact=True);files.append(dict(variant=variant,csv=file.name,sha256=sha(file),rows=len(csv),recipe=final))
    outputs=[p for p in args.out.iterdir() if p.suffix in {'.json','.csv'} and p!=manifest_path];manifest=dict(signature=sig,complete=True,inner_candidate_jobs=24,inner_winners=winners,final_recipe=final,gates=gates,files=files,new_q2_q3_metrics=False,uploads=0,outer_bestof8=False,output_sha256={p.name:sha(p) for p in outputs});write_json(manifest_path,manifest);print(json.dumps(dict(event='complete',gates=gates,final_recipe=final,files=files)),flush=True);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['raw','out','feature-cache','baseline-cv','context-test','benchmark']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan',type=Path,default=DEFAULT_PLAN);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with(args.out/'.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)


if __name__=='__main__':main()
