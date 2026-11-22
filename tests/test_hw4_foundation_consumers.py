"""Portable metadata-only archives; no foundation package, weights or fits.

Predictions here are synthetic consistency fixtures, not model results. The
actual native-weight producer archives are replayed separately in the audit.
"""
from argparse import Namespace
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sys
import tarfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import analyze_hw4_foundation as foundation
import analyze_hw4_tabm_preprocessing as tabm


def _json(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=True)+'\n').encode()


def _archive(path,files):
    completion=json.loads(files['completion.json'])
    completion['artifacts']={name:hashlib.sha256(files[name]).hexdigest() for name in files if name!='completion.json'}
    files=dict(files,**{'completion.json':_json(completion)})
    with tarfile.open(path,'w:gz') as output:
        for name,data in files.items():
            header=tarfile.TarInfo(name);header.size=len(data);output.addfile(header,BytesIO(data))


def _raw(root):
    periods=pd.date_range('2022-01-01',periods=12,freq='QS').strftime('%Y-%m-%d')
    data=pd.DataFrame([dict(row_id=f'{q}-{i}',site_id=f's{i}',forecast_period=q,
        history_before=(pd.Timestamp(q)-pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
        facility_type='restaurant',risk_level='high',area_id='a1',target=i) for q in periods for i in [0,1]])
    train=data.loc[data.forecast_period.ne(periods[-1])].reset_index(drop=True)
    test=data.loc[data.forecast_period.eq(periods[-1])].drop(columns='target').reset_index(drop=True)
    sample=pd.DataFrame(dict(row_id=test.row_id,target=.5))
    history=pd.DataFrame(columns=['forecast_period','site_id','inspection_week','inspection_type','result','violation_codes','facility_type','risk_level','area_id'])
    for name,frame in [('train.csv',train),('test_student.csv',test),('sample_submission.csv',sample),('inspection_history.csv',history)]:frame.to_csv(root/name,index=False)
    old=pd.concat([train.loc[train.forecast_period.eq(q),['row_id','site_id','forecast_period','target']] for q in foundation.pfn.OLD],ignore_index=True)
    recent=pd.concat([train.loc[train.forecast_period.eq(q),list(old)] for q in foundation.pfn.RECENT],ignore_index=True)
    return train,test,old,recent,history


def _foundation_files(root,family):
    train,test,old,recent,history=_raw(root)
    producer=foundation.pfn if family=='tabpfn' else foundation.icl
    checkpoint=producer.CHECKPOINT if family=='tabpfn' else producer.WEIGHT
    plan=dict(format='hw4-tabpfn-v2-offline-1' if family=='tabpfn' else 'hw4-tabicl-v2-offline-1',
        cases=list(producer.CASES),seeds=[42],old_quarters=list(foundation.pfn.OLD),
        recent_prediction_only_quarters=list(foundation.pfn.RECENT),smoke=True,device='cpu',
        new_2024q2_q3_metrics=False,kaggle_uploads=0,external_pretrained_weights=True,
        checkpoint_sha256=checkpoint,source_sha256={n:foundation.pfn.sha(ROOT/n) for n in producer.SOURCES},
        input_sha256={n:foundation.pfn.sha(root/n) for n in foundation.pfn.RAW})
    if family=='tabpfn':plan.update(context_cap=64,ensemble_estimators=4,random_state=42,memory_saving_mode=True,
        fit_mode='fit_preprocessors',inference_precision='torch.float32',versions={'tabpfn':'2.2.1','scikit-learn':'1.6.1'},tabpfn_wheel_sha256=producer.WHEEL)
    else:plan.update(settings=producer.SETTINGS,versions={'tabicl':'2.2.0','scikit-learn':'1.7.2'},tabicl_wheel_sha256=producer.WHEEL)
    plan['signature']=foundation.pfn.digest(plan)
    forecast=test[['row_id','forecast_period']].copy()
    for frame in [old,forecast,recent]:
        for case in producer.CASES:frame[case]=np.tile([.2,.8],len(frame)//2)
    rows=pd.concat([train.drop(columns='target'),test],ignore_index=True)
    base,cats=foundation.pfn.build_features(rows,history)
    base=pd.concat([base,foundation.pfn.build_context_features(rows,history)],axis=1)
    views=foundation.pfn.views(base,cats);jobs=[];metrics=[]
    for case,pfn_case in zip(producer.CASES,foundation.pfn.CASES):
        for q in [*foundation.pfn.OLD,'full',*foundation.pfn.RECENT]:
            prior=train if q=='full' else train.loc[train.forecast_period.lt(q)]
            chosen=prior.sort_values(['forecast_period','row_id'],kind='stable')
            if family=='tabpfn':chosen=chosen.tail(64)
            observed=forecast if q=='full' else old.loc[old.forecast_period.eq(q)] if q in foundation.pfn.OLD else recent.loc[recent.forecast_period.eq(q)]
            identity=dict(signature=plan['signature'],case=case,period=q,train_rows=len(chosen),predict_rows=len(observed),
                training_ids_sha256=foundation.pfn.digest(chosen.row_id.tolist()),prediction_ids_sha256=foundation.pfn.digest(observed.row_id.tolist()),
                training_quarter_counts={k:int(v) for k,v in chosen.forecast_period.value_counts().sort_index().items()})
            first=case==producer.CASES[0] and q==foundation.pfn.OLD[0]
            job=dict(identity=identity,prediction_sha256=foundation.npy_digest(observed[case]),repeated_forward_exact=True if first else None)
            if family=='tabpfn':
                identity.update(seed=42,n_estimators=4,available_train_rows=len(prior),feature_columns=list(views[pfn_case][0]),categorical_indices=views[pfn_case][1])
                job.update(actual_ensemble_estimators=4,external_base_checkpoint_sha256=checkpoint,fit_mode='fit_preprocessors',stored_fitted_weights=False)
            else:
                identity.update(features=list(views[pfn_case][0]),categorical_indices_for_dtype_confirmation=views[pfn_case][1])
                job.update(n_estimators=4,checkpoint_sha256=checkpoint)
            jobs.append(job)
            if q in foundation.pfn.OLD:metrics.append(dict(case=case,seed=42,forecast_period=q,n=len(observed),positive=int(observed.target.sum()),ap=1.,train_rows=len(chosen)))
    info=dict(checkpoint_sha256=checkpoint,runtime_network_attempts=0,jobs=jobs)
    completion=dict(signature=plan['signature'],status='complete',jobs=12,expected_jobs=12,new_2024q2_q3_metrics=False,kaggle_uploads=0,old_rows=len(old),test_rows=len(test),recent_prediction_rows=len(recent),artifacts={})
    return {'protocol.json':_json(plan),'completion.json':_json(completion),'modelinfo.json':_json(info),
        'old_oof.csv':old.to_csv(index=False).encode(),'test_predictions.csv':forecast.to_csv(index=False).encode(),
        'recent_predictions.csv':recent.to_csv(index=False).encode(),'fold_metrics.csv':pd.DataFrame(metrics).to_csv(index=False).encode()}


def test_foundation_consumers_reject_nonfiniteAP_actualpins_and_case_omission():
    for family in ['tabpfn','tabicl']:
        with TemporaryDirectory() as temp:
            root=Path(temp);files=_foundation_files(root,family);path=root/'return.tar.gz'
            _archive(path,files);args=Namespace(family=family,archive=path,raw=root,smoke=True)
            foundation.verify(args)
            runtime = (['seed', 'configured_estimators', 'actual_estimators', 'checkpoint', 'fit_mode', 'stored_weights']
                       if family == 'tabpfn' else ['actual_estimators', 'checkpoint'])
            for kind in ['nanAP','outofrangeAP','missingcontext','npybridge', *runtime]:
                changed=dict(files)
                if kind in ['nanAP','outofrangeAP']:
                    metric=pd.read_csv(BytesIO(changed['fold_metrics.csv']));metric.loc[0,'ap']=np.nan if kind=='nanAP' else 1.1;changed['fold_metrics.csv']=metric.to_csv(index=False).encode()
                else:
                    info=json.loads(changed['modelinfo.json'])
                    if kind=='missingcontext':info['jobs'].pop()
                    elif kind=='npybridge':info['jobs'][0]['prediction_sha256']='wrong'
                    elif kind=='seed':info['jobs'][0]['identity']['seed']=17
                    elif kind=='configured_estimators':info['jobs'][0]['identity']['n_estimators']=1
                    elif kind=='actual_estimators':info['jobs'][0]['actual_ensemble_estimators' if family=='tabpfn' else 'n_estimators']=1
                    elif kind=='checkpoint':info['jobs'][0]['external_base_checkpoint_sha256' if family=='tabpfn' else 'checkpoint_sha256']='wrong'
                    elif kind=='fit_mode':info['jobs'][0]['fit_mode']='low_memory'
                    elif kind=='stored_weights':info['jobs'][0]['stored_fitted_weights']=True
                    changed['modelinfo.json']=_json(info)
                _archive(path,changed)
                try:foundation.verify(args);raise AssertionError(f'Accepted{family}:{kind}')
                except (ValueError,AssertionError) as error:
                    if str(error).startswith('Accepted'):raise


def test_tabm_consumer_rejects_nonfiniteAP_and_unbounded_extra_inner_epochs():
    with TemporaryDirectory() as temp:
        root=Path(temp);_,test,old,recent,_=_raw(root)
        plan=dict(format='hw4-tabm-preprocessing-v100-1',cases=tabm.producer.CASES,seeds=list(tabm.producer.SEEDS),
            old_quarters=list(tabm.producer.QUARTERS),recent_quarters=list(tabm.producer.RECENT),smoke=True,device='cpu',max_epochs=1,
            new_2024q2_q3_metrics=False,kaggle_uploads=0,external_pretrained_weights=False,
            versions={'tabm':'0.0.3','rtdl-num-embeddings':'0.0.12'},optimizer=tabm.producer.OPTIMIZER,
            source_sha256={**{'src/hw4_inspections/'+n:tabm.producer.sha(ROOT/'src/hw4_inspections'/n) for n in tabm.producer.SOURCES},
                           'scripts/run_hw4_tabm_preprocessing_v100.py':tabm.producer.sha(ROOT/'scripts/run_hw4_tabm_preprocessing_v100.py')},
            input_sha256={n:tabm.producer.sha(root/n) for n in tabm.producer.RAW})
        plan['signature']=tabm.producer.signature(plan);forecast=test[['row_id','forecast_period']].copy();metrics=[]
        for frame in [old,forecast,recent]:
            for case in tabm.producer.CASES:
                seeds=[]
                for seed in tabm.producer.SEEDS:
                    value=np.tile([.2,.8],len(frame)//2).astype(np.float32).astype(np.float64);frame[f'{case}__s{seed}']=value;seeds.append(value)
                frame[f'{case}__mean3']=np.mean(seeds,axis=0)
        for case in tabm.producer.CASES:
            for seed in tabm.producer.SEEDS:
                for q in tabm.producer.QUARTERS:metrics.append(dict(case=case,seed=seed,forecast_period=q,n=2,positive=1,ap=1.,epochs=1))
        selections={c:dict(full_epochs=1,inner_selected_epochs=[1,1,1]) for c in tabm.producer.CASES}
        completion=dict(signature=plan['signature'],status='complete',jobs=63,expected_jobs=63,new_2024q2_q3_metrics=False,kaggle_uploads=0,recent_scores_computed=0,old_rows=len(old),test_rows=len(test),recent_rows=len(recent),artifacts={})
        files={'protocol.json':_json(plan),'completion.json':_json(completion),'epoch_selections.json':_json(selections),
            'old_oof.csv':old.to_csv(index=False).encode(),'test_predictions.csv':forecast.to_csv(index=False).encode(),
            'recent_predictions.csv':recent.to_csv(index=False).encode(),'fold_metrics.csv':pd.DataFrame(metrics).to_csv(index=False).encode()}
        path=root/'return.tar.gz';_archive(path,files);args=Namespace(archive=path,raw=root,smoke=True);tabm.verify(args)
        for kind in ['nanAP','extraepochs']:
            changed=dict(files)
            if kind=='nanAP':
                metric=pd.read_csv(BytesIO(changed['fold_metrics.csv']));metric.loc[0,'ap']=np.nan;changed['fold_metrics.csv']=metric.to_csv(index=False).encode()
            else:
                epochs=json.loads(changed['epoch_selections.json']);case=next(iter(epochs));epochs[case]=dict(full_epochs=101,inner_selected_epochs=[1,1,1,101,101,101,101]);changed['epoch_selections.json']=_json(epochs)
            _archive(path,changed)
            try:tabm.verify(args);raise AssertionError(f'AcceptedTabM:{kind}')
            except ValueError:pass
