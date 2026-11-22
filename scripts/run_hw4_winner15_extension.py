"""Three fixed OLD-only extensions of winner15; no coefficient search.

Screen freezes source/data/recipes before AP. Final exports one admitted formula;
only selected C requires seven new full CatBoost fits, never repeats OLD fits.
"""
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from io import BytesIO
from importlib import metadata
from pathlib import Path
import hashlib,json,sys

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_hw4_final_research as primary
import run_hw4_context_bag10 as bag
from hw4_inspections.next_round import evaluate_candidates

OLD=['2023-07-01','2023-10-01','2024-01-01']
META=['row_id','site_id','forecast_period','target']
NAMES=['A_context3_native32_odds50','B_context3_native4_native32_hierarchical','C_context10_native4_odds50']
WINNER='winner15_context3_native4_odds50'
ADDED=[7,23,101,123,456,777,1000]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def csv(path):return pd.read_csv(path,float_precision='round_trip')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(path,value):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=data:raise ValueError(f'Refusing changed artifact:{path}')
    else:
        with path.open('xb') as stream:stream.write(data)


def align(frame,expected):
    if frame.row_id.isna().any() or not frame.row_id.is_unique or set(frame.row_id)!=set(expected.row_id) or len(frame)!=len(expected):raise ValueError('Different forecast identities')
    frame=frame.set_index('row_id').loc[expected.row_id].reset_index()
    common=[c for c in expected if c in frame]
    pd.testing.assert_frame_equal(expected[common],frame[common],check_dtype=False,check_exact=True)
    return frame


def prob(values,n):
    values=np.asarray(values,dtype=np.float64)
    if values.shape!=(n,) or not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('Invalid probabilities')
    return values


def formulas(c3,i4,i32,c10):
    l=lambda p:logit(np.clip(p,1e-6,1-1e-6))
    return {WINNER:expit(.5*l(c3)+.5*l(i4)),NAMES[0]:expit(.5*l(c3)+.5*l(i32)),
        NAMES[1]:expit(.5*l(c3)+.25*l(i4)+.25*l(i32)),NAMES[2]:expit(.5*l(c10)+.5*l(i4))}


def paths(work):
    old=work/'analysis/hw4-post-public-20261122';six=work/'analysis/hw4-six-hour-20261122';deep=work/'analysis/hw4-deep-research-20261122'
    return dict(raw=work/'data/hw4/raw',baseline=work/'analysis/hw4-next-round-20261121/baseline_oof.csv',
        baseline_audit=work/'analysis/hw4-next-round-20261121/baseline-audit.json',winner_old=old/'ensemble/duo50/old-oof.csv',winner_old_freeze=old/'ensemble/duo50/freeze.json',winner_old_complete=old/'ensemble/duo50/complete.json',
        winner_freeze=old/'exploratory-duo-reserve-freeze.json',winner_csv=work/'outputs/hw4/post-public-20261122/submission_15_context50_tabicl94_odds.csv',winner_manifest=work/'outputs/hw4/post-public-20261122/manifest.json',
        c3_test=work/'outputs/hw4/final-20261120/submission_04_context_cb.csv',i4_old=six/'gpu-analysis/tabicl/old-oof.csv',i4_test=six/'gpu-analysis/tabicl/raw-selected-test.csv',
        native32=old/'tabicl32-result',bag=deep/'context-bag10',cache=deep/'features.joblib',delivery=work/'outputs/hw4/final-20261120')


def verify_inputs(p):
    train,test,sample=[csv(p['raw']/n) for n in ['train.csv','test_student.csv','sample_submission.csv']]
    if (len(train),len(test),len(sample))!=(21017,2246,2246):raise ValueError('Wrong official sizes')
    if not np.array_equal(test.row_id,sample.row_id):raise ValueError('Sample/test order differs')
    meta=pd.concat([train.loc[train.forecast_period.eq(q),META] for q in OLD],ignore_index=True)
    baseline=align(csv(p['baseline']),meta);i4=align(csv(p['i4_old']),meta);duo=align(csv(p['winner_old']),meta);c10=align(csv(p['bag']/'old-oof.csv'),meta)
    if sha(p['baseline'])!='0096456a838264be615680be4eead0ed8b449a4182d9f20acd519937e7cc1f7c' or sha(p['winner_csv'])!='3a64705549d64f3cc729ee32af6f522b8d8f61ce47bc9fa5effe9a2284851bef':raise ValueError('Wrong signed incumbent')
    wf=read(p['winner_freeze']);wm=read(p['winner_manifest']);df=read(p['winner_old_freeze']);dc=read(p['winner_old_complete'])
    if sha(p['winner_freeze'])!='58e39482fb2a94d16b29699ee0803649bf8aa143f1a8b9ea4d798a1d9f119ced' or wm['freeze_sha256']!=sha(p['winner_freeze']) or wm['sha256']!=sha(p['winner_csv']):raise ValueError('Changed winner15 recipe/CSV')
    for name,h in wf['old_gate_evidence'].items():
        if sha(p['winner_old'].parent/name)!=h:raise ValueError('Changed signed winner OLD evidence')
    if dc['signature']!=df['signature'] or any(sha(p['winner_old'].parent/n)!=h for n,h in dc['artifacts'].items()):raise ValueError('Changed original duo result')
    c3=prob(baseline.prediction,len(meta));native4=prob(i4.tabicl_v2_compact94__raw,len(meta))
    for other in [i4.context04,duo.context04,c10.context_3seed]:
        if not np.array_equal(c3,np.asarray(other)):raise ValueError('CB parent OOF differs')
    if not np.array_equal(native4,duo.icl94_raw):raise ValueError('Native4 parent OOF differs')
    bf=read(p['bag']/'freeze.json');bm=read(p['bag']/'forecast-manifest.json')
    if bf['signature']!=bm['signature'] or digest({k:v for k,v in bf.items() if k!='signature'})!=bf['signature'] or bf['old_seeds']!=[42,17,2026] or bf['additional_seeds']!=ADDED:raise ValueError('Different bag10 recipe')
    if bf['model']['spec']['params']!=bag.PARAMS or bf['model']['spec']['max_iterations']!=500 or not bf['old_full_mean_matches_context04']:raise ValueError('Different bag10 parameters')
    if any(sha(p['bag']/n)!=h for n,h in bm['sha256'].items()):raise ValueError('Changed completed bag10 evidence')
    seed_arrays=[];job_proof=[]
    for seed in ADDED:
        parts=[]
        for q in OLD:
            path=p['bag']/'jobs'/f'context_bag10__{q}__s{seed}';r=read(path/'complete.json')
            expected=dict(case='context_bag10',fixed_count=500,period=q,predict_rows=int(train.forecast_period.eq(q).sum()),seed=seed,signature=bf['signature'],train_rows=int(train.forecast_period.lt(q).sum()))
            if r['identity']!=expected or r['iterations']!=500 or r['inner_quarter'] is not None or not r['reload_exact'] or any(sha(path/n)!=h for n,h in r['sha256'].items()):raise ValueError('Changed old bag10 checkpoint')
            parts.append(prob(np.load(path/'prediction.npy',allow_pickle=False),expected['predict_rows']))
            job_proof.append(dict(seed=seed,quarter=q,marker_sha256=sha(path/'complete.json'),**r['sha256']))
        seed_arrays.append(np.concatenate(parts))
    reconstructed=bag.bag_mean(c3,seed_arrays)
    if not np.array_equal(reconstructed,c10.context_bag10):raise ValueError('Bag10 OOF does not reconstruct exactly')
    n32=p['native32'];np_=read(n32/'protocol.json');nc=read(n32/'completion.json');ni=read(n32/'modelinfo.json');nv=read(n32/'verification.json')
    if digest({k:v for k,v in np_.items() if k!='signature'})!=np_['signature'] or np_['signature']!=nc['signature'] or nv['signature']!=np_['signature'] or nv['status']!='PASS' or np_['settings']['n_estimators']!=32 or np_['settings']['random_state']!=42 or nc['jobs']!=4 or nc['new_2024q2_q3_metrics'] or ni['runtime_network_attempts']!=0:raise ValueError('Different native32 provenance')
    if any(sha(n32/n)!=h for n,h in nc['artifacts'].items()):raise ValueError('Changed native32 artifacts')
    for name,h in np_['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Different executed native32 source')
    i32=align(csv(n32/'old_oof.csv'),meta);test32=align(csv(n32/'test_predictions.csv'),test[['row_id','forecast_period']])
    for item in ni['jobs']:
        identity=item['identity'];q=identity['period'];selected=test32 if q=='full' else i32.loc[i32.forecast_period.eq(q)];pred=prob(selected.tabicl_v2_compact94,len(selected));buf=BytesIO();np.save(buf,pred,allow_pickle=False)
        prior=train if q=='full' else train.loc[train.forecast_period.lt(q)]
        prior=prior.sort_values(['forecast_period','row_id'],kind='stable')
        if identity['training_ids_sha256']!=digest(prior.row_id.tolist()) or identity['prediction_ids_sha256']!=digest(selected.row_id.tolist()) or item['n_estimators']!=32 or hashlib.sha256(buf.getvalue()).hexdigest()!=item['prediction_sha256']:raise ValueError('Different native32 contexts/prediction bridge')
    values=formulas(c3,native4,prob(i32.tabicl_v2_compact94,len(meta)),reconstructed)
    if not np.array_equal(values[WINNER],duo.context50_icl50_odds):raise ValueError('Winner15 signed OLD formula differs')
    # Verify full parent identity and reproduce winner15 without fitting.
    full3=prob(align(csv(p['c3_test']),test[['row_id']]).target,len(test));full4=prob(align(csv(p['i4_test']),test[['row_id']]).prediction,len(test));full32=prob(test32.tabicl_v2_compact94,len(test))
    winner=align(csv(p['winner_csv']),test[['row_id']]).target.to_numpy()
    if not np.array_equal(formulas(full3,full4,full32,full3)[WINNER],winner):raise ValueError('Winner15 full formula differs')
    proof=dict(status='PASS',bag10_old_jobs_verified=21,bag10_mean10_exact=True,winner15_old_and_full_formula_exact=True,native32_contexts_and_NPY_bridge_verified=4,old_rows=len(meta),test_rows=len(test),bag_jobs=job_proof,new_fits=0,new_recent_AP=0)
    return train,test,sample,meta,values,(full3,full4,full32),proof


def freeze_plan(a,p):
    sources=[Path(__file__),ROOT/'scripts/run_hw4_context_bag10.py',ROOT/'scripts/run_hw4_final_research.py',*[ROOT/'src/hw4_inspections'/n for n in ['next_round.py','classical_analysis.py','deep_boosting.py','classical_models.py','features.py','extensions.py','delivery.py']]]
    parents=[p[n] for n in ['baseline','baseline_audit','winner_old','winner_old_freeze','winner_old_complete','winner_freeze','winner_csv','winner_manifest','c3_test','i4_old','i4_test']]
    parents += [p['native32']/n for n in ['protocol.json','completion.json','modelinfo.json','verification.json','old_oof.csv','test_predictions.csv']]
    parents += [p['bag']/n for n in ['freeze.json','forecast-manifest.json','old-oof.csv']]
    plan=dict(format='hw4-winner15-extension-fixed3-1',reference=WINNER,nonreference_forms=NAMES,coefficient_search=False,
        formulas={NAMES[0]:'expit(.5logitclip(C3)+.5logitclip(I32))',NAMES[1]:'expit(.5logitclip(C3)+.25logitclip(I4)+.25logitclip(I32))',NAMES[2]:'expit(.5logitclip(Cbag10_meanprob)+.5logitclip(I4))'},logit_clip=[1e-6,1-1e-6],
        cb_mean_definition='C3=arithmetic mean of native probabilities seeds42/17/2026; Cbag10=(3*C3+sum7added native probabilities in frozen order)/10, then logit.',
        old_quarters=OLD,gate='mean delta vs winner15 >0; >=2 nonnegative quarters; worst>=-.002;2023Q4>=0',
        selection='At most one gate-passing candidate, highest equal-quarter OLD mean; ties fixed A/B/C order. No extra variants.',
        bootstrap='Only if any AP-gate passes:2000 common-site paired draws seed20261123; nominal after Public/OLD adaptive research.',
        full_rule='Selected A/B reuses existing full parents, zero fits. Selected C:report separate seven-full-fit plan first; reuse verified original three full models, fit only seven added CB seeds. Existing21OLD models never fit again.',
        sources_sha256={str(x.resolve()):sha(x) for x in sources},parent_sha256={str(x.resolve()):sha(x) for x in parents},raw_sha256={n:sha(p['raw']/n) for n in primary.RAW_FILES},
        versions={n:metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','catboost','joblib']},
        observed_incumbent_public=.35716,interpretation='Exploratory extension after winner15 Public feedback. OLD quarters heavily reused; not independent confirmation. No current/recent quarter metric or coefficient tuning.',new_recent_AP=0,uploads=0)
    plan['signature']=digest(plan)
    return plan


def screen(a,p):
    a.out.mkdir(parents=True,exist_ok=True);plan=freeze_plan(a,p)
    if (a.out/'freeze.json').exists():
        if read(a.out/'freeze.json')!=plan:raise ValueError('Changed fixed3 freeze')
        if not (a.out/'screen-complete.json').exists():raise ValueError('Incomplete screen; inspect before repeating metrics')
        done=read(a.out/'screen-complete.json')
        if any(sha(a.out/n)!=h for n,h in done['artifacts'].items()):raise ValueError('Changed completed screen')
        return done
    if any(a.out.iterdir()):raise ValueError('New screen requires empty child output')
    # Freeze before reading/scoring any new OLD formula.
    write(a.out/'freeze.json',plan)
    train,test,sample,meta,values,full,proof=verify_inputs(p);write(a.out/'input-verification.json',proof)
    result=evaluate_candidates(meta,values,reference=WINNER,bootstrap=0,seed=20261123)
    scores=result['quarter_scores'].pivot(index='forecast_period',columns='model',values='ap')
    summary=result['summary'].copy();summary['q4_delta']=summary.model.map((scores.loc['2023-10-01']-scores.loc['2023-10-01',WINNER]).to_dict())
    summary['eligible']=summary.model.isin(NAMES)&summary.delta_vs_reference.gt(0)&summary.nonnegative_quarters.ge(2)&summary.worst_quarter_delta.ge(-.002)&summary.q4_delta.ge(0)
    eligible=summary.loc[summary.eligible,'model'].tolist()
    if eligible:
        interval=evaluate_candidates(meta,values,reference=WINNER,bootstrap=2000,seed=20261123)['summary'].set_index('model')
        for col in ['ci95_low','ci95_high']:summary[col]=summary.model.map(interval[col])
    summary=summary.sort_values(['mean_ap','model'],ascending=[False,True],kind='stable');summary.to_csv(a.out/'summary.csv',index=False)
    result['quarter_scores'].to_csv(a.out/'quarter-metrics.csv',index=False);pd.concat([meta,pd.DataFrame(values)],axis=1).to_csv(a.out/'old-oof.csv',index=False)
    selected=min(eligible,key=lambda n:(-float(summary.set_index('model').loc[n,'mean_ap']),NAMES.index(n))) if eligible else None
    done=dict(status='SCREEN_COMPLETE',signature=plan['signature'],eligible=eligible,selected=selected,bootstrap_computed=bool(eligible),new_fits=0,new_recent_AP=0,uploads=0,
        artifacts={x.name:sha(x) for x in a.out.iterdir() if x.is_file()})
    write(a.out/'screen-complete.json',done);print(json.dumps(done,indent=2));return done


def final(a,p):
    done=screen(a,p)
    if done['selected'] is None:raise ValueError('No formula passed; no final forecast')
    if (a.out/'final-manifest.json').exists():
        old=read(a.out/'final-manifest.json')
        if any(sha(a.out/n)!=h for n,h in old['artifacts'].items()):raise ValueError('Changed final result')
        return old
    train,test,sample,meta,values,(c3,i4,i32),proof=verify_inputs(p);selected=done['selected'];new_fits=0;full_proof={}
    c10=c3
    if selected==NAMES[2]:
        if not (a.out/'seven-full-fit-plan.json').exists():raise ValueError('Publish/review seven-full-fit-plan before running selected C')
        cachemeta=read(p['cache'].with_name('features-meta.json'))
        if sha(p['cache'])!=cachemeta['cache_sha256']:raise ValueError('Feature cache differs')
        args=Namespace(raw=p['raw'],smoke=False,baseline_cv=p['baseline'],context_test=p['c3_test'],delivery_run=p['delivery'])
        tr,te,sa,views,cats=primary.validate_cache(joblib.load(p['cache']),args);frame=views['base']
        old,full_proof=bag.parent_arrays(args,{n:sha(p['raw']/n) for n in primary.RAW_FILES},tr,te,frame,cats,c3,len(meta))
        case={'family':bag.NAME,'spec':{'family':'catboost','params':bag.PARAMS,'max_iterations':500},'view':'base','inner':False}
        arrays=[]
        for seed in ADDED:
            jobpath=a.out/'jobs'/f'{bag.NAME}__full__s{seed}';existed=jobpath.exists()
            value,count=primary.run_job(a.out,done['signature'],bag.NAME,case,seed,'full',500,tr,frame,cats,2,True)
            if count!=500:raise ValueError('Changed fixed tree count')
            arrays.append(value);new_fits+=not existed
        c10=bag.bag_mean(old,arrays)
        pd.DataFrame({'row_id':test.row_id,'prediction':c10}).to_csv(a.out/'context10-full-component.csv',index=False)
    prediction=formulas(c3,i4,i32,c10)[selected]
    output=primary.validate_submission(test,sample,prediction,2246);path=a.out/f'submission_{selected}.csv';output.to_csv(path,index=False)
    pd.testing.assert_frame_equal(output,csv(path),check_exact=True)
    manifest=dict(status='READY_NOT_SUBMITTED',signature=done['signature'],model=selected,file=path.name,sha256=sha(path),rows=2246,public_ap=None,private_ap=None,new_fits_this_invocation=int(new_fits),new_recent_AP=0,uploads=0,
        formula=read(a.out/'freeze.json')['formulas'][selected],existing_full_C3_proof=full_proof,artifacts={x.name:sha(x) for x in a.out.iterdir() if x.is_file()})
    write(a.out/'final-manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k not in ['artifacts','existing_full_C3_proof']},indent=2));return manifest


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--stage',choices=['screen','final'],default='screen');a=parser.parse_args();p=paths(a.workspace)
    (screen if a.stage=='screen' else final)(a,p)
