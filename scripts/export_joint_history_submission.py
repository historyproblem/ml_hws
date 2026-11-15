"""Export the explicitly selected pre-embedding CUDA Joint+history mean3."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spend_baselines.data import read_objects
from spend_baselines.suite_data import atomic_json,digest
from spend_tuning.data import FrozenRun

SETTINGS={"hidden_sizes":[128,64],"weight_decay":.0001,"dropout":0.,"learning_rate":.001,
          "batch_size":1024,"max_epochs":200,"patience":20,"min_delta":.0001}
SEEDS=[42,123,2026]


def require_full_cuda(manifest):
    if manifest.get('smoke') or manifest.get('device')!='cuda':
        raise ValueError('Submission requires a full CUDA refit, not a smoke/CPU fallback')
    cfg=manifest['config']
    if cfg['base_settings']!=SETTINGS or cfg['seeds']!=SEEDS:
        raise ValueError('Expected the original Joint+history HP and three fixed seeds')
    base=next(p for p in cfg['profiles'] if p['name']=='baseline')
    if base['overrides']:
        raise ValueError('The baseline profile must not have overrides')


def prediction_frame(values,test):
    keys=pd.DataFrame({'Id':values['Id'].astype(str),'customer_id':values['customer_id'].astype(str),
                       'z':values['z'],'spend_30d':values['spend_30d']})
    if not keys.Id.is_unique or set(keys.Id)!=set(test.Id):
        raise ValueError('Prediction Ids do not match the full official test')
    keys=keys.set_index('Id').loc[test.Id]
    if not np.array_equal(keys.customer_id.to_numpy(),test.customer_id.to_numpy(str)):
        raise ValueError('Prediction customer keys differ from test')
    if not np.isfinite(keys[['z','spend_30d']]).all().all() or (keys.z<0).any() or (keys.spend_30d<0).any():
        raise ValueError('Predictions must be finite and nonnegative')
    np.testing.assert_allclose(keys.spend_30d,np.maximum(np.expm1(keys.z),0),rtol=1e-12,atol=1e-9)
    return keys


def assemble(test,sample,seed_values,mean_values):
    if len(seed_values)!=3:
        raise ValueError('Exactly three seed predictions are required')
    if list(sample.columns)!=['Id','spend_30d'] or not sample.Id.is_unique or set(sample.Id)!=set(test.Id):
        raise ValueError('Sample submission schema/Ids differ')
    aligned=[prediction_frame(p,test) for p in seed_values]
    mean_z=np.mean([p.z.to_numpy() for p in aligned],axis=0)
    saved=prediction_frame(mean_values,test)
    np.testing.assert_allclose(saved.z,mean_z,atol=1e-12,rtol=0)
    values=np.maximum(np.expm1(mean_z),0)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite money forecasts')
    out=pd.DataFrame({'Id':test.Id.to_numpy(str),'spend_30d':values}).set_index('Id').loc[sample.Id].reset_index()
    return out


def load_npz(path):
    with np.load(path,allow_pickle=False) as p:
        return {key:p[key] for key in ['Id','customer_id','z','spend_30d']}


def export(run_dir,reference,data_dir,output):
    run_dir,data_dir,output=Path(run_dir),Path(data_dir),Path(output)
    manifest_path=run_dir/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    require_full_cuda(manifest)
    for name,sha in manifest['source_sha256'].items():
        if not name.startswith(('spend_baselines/','spend_behavior/','spend_tuning/')) or '..' in Path(name).parts:
            raise ValueError('Unexpected model source path')
        if digest(Path(__file__).resolve().parents[1]/'src'/name)!=sha:
            raise ValueError(f'Original refit implementation differs: {name}')
    frozen=FrozenRun(reference,data_dir,SETTINGS)
    if hashlib.sha256(frozen.manifest_bytes).hexdigest()!=manifest['reference_manifest_sha256']:
        raise ValueError('The refit reference differs from the verified Joint+history run')
    test=read_objects(data_dir/'test.csv')
    if test.as_of_date.nunique()!=1 or not test.as_of_date.eq('2018-11-01').all():
        raise ValueError('Expected the official 2018-11-01 test cutoff')
    sample=pd.read_csv(data_dir/'sample_submission.csv',dtype={'Id':str})
    date='2018-11-01'
    directory=run_dir/'refit/baseline'
    marker=json.loads((directory/'result.json').read_text())
    if marker['profile']!='baseline' or marker['cutoff']!=date:
        raise ValueError('Expected baseline refit; do not submit the early-data lr_double winner')
    if digest(directory/'predictions.npz')!=marker['predictions_sha256']:
        raise ValueError('Corrupt mean3 refit forecasts')
    per_seed=[]
    provenance=[]
    for seed in SEEDS:
        jobdir=run_dir/'refit_jobs/baseline'/date/str(seed)
        result=json.loads((jobdir/'result.json').read_text())
        expected={'profile':'baseline','date':date,'seed':seed,'testing':True,'settings':SETTINGS}
        if result['job']!=expected or result['n_validation']!=len(test):
            raise ValueError('Unexpected refit job/model/population')
        if pd.Timestamp(result['max_target_end_train'])>pd.Timestamp(date):
            raise ValueError('Refit used labels beyond test cutoff')
        inner=pd.Timestamp(result['inner_date'])
        if inner!=pd.Timestamp('2018-10-01') or pd.Timestamp(result['max_target_end_inner'])>inner:
            raise ValueError('Unexpected inner temporal split')
        for name,sha in result['sha256'].items():
            if Path(name).name!=name or digest(jobdir/name)!=sha:
                raise ValueError(f'Corrupt refit artifact: {jobdir/name}')
        refit=pd.read_csv(jobdir/'refit.csv')
        selection=pd.read_csv(jobdir/'selection.csv')
        best,epoch=np.inf,0
        for row in selection.itertuples():
            if row.inner_metric<best-SETTINGS['min_delta']:best,epoch=row.inner_metric,row.epoch
        if len(refit)!=result['selected_epoch'] or epoch!=result['selected_epoch']:
            raise ValueError('Refit epoch does not match inner selection')
        per_seed.append(load_npz(jobdir/'predictions.npz'))
        provenance.append({'seed':seed,'selected_epoch':epoch,'n_train':result['n_train'],
                           'result_sha256':digest(jobdir/'result.json')})
    out=assemble(test,sample,per_seed,load_npz(directory/'predictions.npz'))
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.tmp')
    out.to_csv(temporary,index=False)
    checked=pd.read_csv(temporary,dtype={'Id':str})
    if list(checked.columns)!=['Id','spend_30d'] or not checked.Id.equals(sample.Id):
        raise ValueError('Written CSV schema/order check failed')
    if not np.isfinite(checked.spend_30d).all() or checked.spend_30d.lt(0).any():
        raise ValueError('Written CSV value check failed')
    temporary.replace(output)
    receipt={'candidate':'Joint+history mean3, original HP, no embeddings','cutoff':date,
             'rows':len(out),'seeds':SEEDS,'settings':SETTINGS,'source_manifest_sha256':digest(manifest_path),
             'reference_manifest_sha256':manifest['reference_manifest_sha256'],
             'test_sha256':digest(data_dir/'test.csv'),'sample_sha256':digest(data_dir/'sample_submission.csv'),
             'submission_sha256':digest(output),'seed_refits':provenance,'kaggle_uploaded':False}
    atomic_json(output.with_suffix('.json'),receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=Path('outputs/history-tuning-v1'))
    p.add_argument('--reference',type=Path,default=Path('outputs/behavior-v1'))
    p.add_argument('--data-dir',type=Path,default=Path('data/raw'))
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    output=args.output or args.run_dir/'submission_joint_history_mean3.csv'
    result=export(args.run_dir,args.reference,args.data_dir,output)
    print(f'Created {output}: {result["rows"]} rows; SHA256 {result["submission_sha256"]}')
    print('CSV validated. Kaggle upload has not been performed.')


if __name__=='__main__':
    raise SystemExit('Archived Joint+mean3 exporter. Active scope is seminars1-10; use scripts/make_course10_submission.sh')
