"""Exploratory post-Public equal-weight blends, OLD quarters only; zero fits."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
from importlib import metadata
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hw4_inspections.next_round import evaluate_candidates

OLD = ['2023-07-01', '2023-10-01', '2024-01-01']
META = ['row_id', 'site_id', 'forecast_period', 'target']
NAMES = ['context50_icl50_probability', 'context50_icl50_odds', 'context50_icl50_rank']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def signed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def align(frame, expected):
    if not frame.row_id.is_unique or set(frame.row_id) != set(expected.row_id) or len(frame) != len(expected):
        raise ValueError('Different forecast identities')
    frame = frame.set_index('row_id').loc[expected.row_id].reset_index()
    common = [name for name in META if name in frame and name in expected]
    pd.testing.assert_frame_equal(frame[common], expected[common], check_dtype=False, check_exact=True)
    return frame


def probabilities(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError('Invalid probability vector')
    return value


def blends(context, provider, quarters):
    context, provider = probabilities(context), probabilities(provider)
    rank = np.empty(len(context), dtype=np.float64)
    for period in pd.unique(quarters):
        mask = np.asarray(quarters) == period
        rank[mask] = (rankdata(context[mask], method='average') + rankdata(provider[mask], method='average')) / (2 * (int(mask.sum()) + 1))
    return dict(zip(NAMES, [.5 * context + .5 * provider,
        expit(.5 * logit(np.clip(context, 1e-6, 1 - 1e-6)) + .5 * logit(np.clip(provider, 1e-6, 1 - 1e-6))), rank]))


def run(a):
    a.out.mkdir(parents=True, exist_ok=True)
    raw = {name: sha(a.raw / name) for name in ['train.csv', 'test_student.csv', 'inspection_history.csv', 'sample_submission.csv']}
    if sha(a.baseline_cv) != '0096456a838264be615680be4eead0ed8b449a4182d9f20acd519937e7cc1f7c':
        raise ValueError('Expected exact verified context04 OOF')
    paths = [a.baseline_cv, a.provider_oof, a.context_test, a.provider_test, Path(__file__),
             ROOT/'src/hw4_inspections/next_round.py', ROOT/'src/hw4_inspections/classical_analysis.py']
    plan = dict(format='hw4-post-public-duo50-1', formulas=NAMES, weights=[.5,.5], logit_clip=[1e-6,1-1e-6],
        rank='Within each OLD quarter or full test: average-tie rank/(n+1), then arithmetic mean across models',
        old_quarters=OLD, raw_sha256=raw, source_and_inputs_sha256={str(p.resolve()):sha(p) for p in paths},
        packages={n:metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn']},
        gate='Mean AP strictly above context04 and raw ICL94; >=2 nonnegative deltas vs ICL94; worst vs ICL94 >=-.005',
        export='At most one gate-passing formula, highest OLD equal-quarter mean; ties follow probability,odds,rank order. No new full fit.',
        bootstrap='Only if a formula passes the AP gate:2000 paired common-site draws, seed20261122, nominal; otherwise skip',
        interpretation='Exploratory after Public feedback (.34646 context04/.34642 ICL94/.34242 GAM75-ICL25); OLD development reused/adaptively consulted. No new recent-quarter metrics.',
        fits=0, new_recent_AP=0, new_2024q2_q3_metrics=False, uploads=0)
    plan['signature']=signed(plan)
    if (a.out/'freeze.json').exists():
        if json.loads((a.out/'freeze.json').read_text()) != plan:
            raise ValueError('Changed exploratory freeze')
        if not (a.out/'complete.json').exists():raise ValueError('Incomplete previous run; inspect before replay')
        complete=json.loads((a.out/'complete.json').read_text())
        if any(sha(a.out/n)!=h for n,h in complete['artifacts'].items()):raise ValueError('Changed completed outputs')
        return complete
    if any(a.out.iterdir()):raise ValueError('New experiment requires empty output directory')
    # Freeze before any OLD scores are computed.
    write(a.out/'freeze.json',plan)
    train=csv(a.raw/'train.csv')
    meta=pd.concat([train.loc[train.forecast_period.eq(q),META] for q in OLD],ignore_index=True)
    if len(meta)!=5815:raise ValueError('Wrong OLD rows')
    baseline=align(csv(a.baseline_cv),meta);provider=align(csv(a.provider_oof),meta)
    context=probabilities(baseline.prediction);icl=probabilities(provider.tabicl_v2_compact94__raw)
    if not np.array_equal(context, provider.context04.to_numpy()):raise ValueError('Provider context parent differs')
    values={'context04':context,'icl94_raw':icl,**blends(context,icl,meta.forecast_period)}
    reports={}
    for ref in ['context04','icl94_raw']:
        reports[ref]=evaluate_candidates(meta,values,reference=ref,bootstrap=0,seed=20261122)
    summary=reports['icl94_raw']['summary'].copy().set_index('model')
    minimum=max(float(summary.loc['context04','mean_ap']),float(summary.loc['icl94_raw','mean_ap']))
    eligible=[n for n in NAMES if summary.loc[n,'mean_ap']>minimum and summary.loc[n,'nonnegative_quarters']>=2 and summary.loc[n,'worst_quarter_delta']>=-.005]
    if eligible:
        for ref in ['context04','icl94_raw']:
            reports[ref]=evaluate_candidates(meta,{k:v for k,v in values.items() if k in ['context04','icl94_raw',*eligible]},reference=ref,bootstrap=2000,seed=20261122)
    for ref,result in reports.items():
        result['summary'].to_csv(a.out/f'summary-vs-{ref}.csv',index=False)
        result['quarter_scores'].to_csv(a.out/f'quarter-metrics-vs-{ref}.csv',index=False)
    pd.concat([meta,pd.DataFrame(values)],axis=1).to_csv(a.out/'old-oof.csv',index=False)
    exports=[]
    if eligible:
        winner=min(eligible,key=lambda n:(-float(summary.loc[n,'mean_ap']),NAMES.index(n)))
        test,sample=[csv(a.raw/n) for n in ['test_student.csv','sample_submission.csv']]
        if len(test)!=2246 or not np.array_equal(test.row_id,sample.row_id):raise ValueError('Different test/sample order')
        c=align(csv(a.context_test),test[['row_id']]);i=align(csv(a.provider_test),test[['row_id']])
        probability=blends(c.target,i.prediction,np.repeat('test',len(test)))[winner]
        output=pd.DataFrame({'row_id':sample.row_id,'target':probabilities(probability)})
        path=a.out/f'submission_{winner}.csv';output.to_csv(path,index=False)
        pd.testing.assert_frame_equal(output,csv(path),check_exact=True)
        exports.append(dict(file=path.name,sha256=sha(path),model=winner,rows=2246,status='READY_NOT_SUBMITTED',public_ap=None))
    complete=dict(status='COMPLETE',signature=plan['signature'],eligible=eligible,files=exports,new_fits=0,new_recent_AP=0,uploads=0,
        bootstrap_computed=bool(eligible),artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()})
    write(a.out/'complete.json',complete)
    print(json.dumps(complete,indent=2))
    return complete


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for name in ['raw','out','baseline-cv','provider-oof','context-test','provider-test']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
