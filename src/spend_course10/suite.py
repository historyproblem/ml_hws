"""Reuse evaluated elementary models; refit one approved predictor, no blending."""

import argparse
import fcntl
import io
import json
import platform
import time
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from spend_baselines.cli import write_submission
from spend_baselines.data import RFM_COLUMNS,read_objects
from spend_baselines.suite_data import BASE_COLUMNS,CachedData,atomic_json,digest,split_indices
from spend_baselines.suite_models import Neural,Tabular

REVISION='8a20cdf4b981c2962ef9741821b8d811468cf849'
MODEL_FAMILIES={'ridge':'linear','forest':'forest','mlp':'mlp'}
METHOD_SOURCES={
    'ridge':[{'seminar':2,'cells':[29,30]},{'seminar':6,'cells':[21,23]}],
    'mlp':[{'seminar':3,'cells':[23,33]},{'seminar':8,'cells':[8,12]}],
    'forest':[{'seminar':9,'cells':[57,59,69]}],
}


def read_config(path):
    cfg=json.loads(Path(path).read_text())
    expected={'version':1,'max_seminar':10,'course_revision':REVISION,
              'dataset':'D0','features':'F0','task':'direct','seed':42,
              'models':['ridge','forest','mlp'],'mlp_profile':'compact_l2'}
    for k,v in expected.items():
        if cfg.get(k)!=v:raise ValueError(f'Not the strict course10 recipe: {k}')
    if set(cfg)!=set(expected)|{'validation_dates'}:
        raise ValueError('Unknown configuration extension; audit it before adding methods')
    if not cfg['validation_dates'] or len(cfg['validation_dates'])!=len(set(cfg['validation_dates'])):
        raise ValueError('Empty/duplicate validation dates')
    return cfg


def approved_job(job,cfg,model=None):
    if any(key in job for key in ['mode','member','members','bootstrap','ensemble']):return False
    if job.get('test') or job.get('dataset')!='D0' or job.get('representation')!='F0':return False
    if job.get('task')!='direct' or job.get('seed')!=42:return False
    family=MODEL_FAMILIES.get(model) if model else job.get('family')
    if family not in MODEL_FAMILIES.values() or job.get('family')!=family:return False
    if family=='mlp':
        return job.get('kind')=='neural' and job.get('profile')==cfg['mlp_profile']
    return job.get('kind')=='tabular'


class Reference:
    def __init__(self,path,data_dir,cfg):
        self.path,self.cfg=Path(path),cfg
        self.zip=zipfile.ZipFile(path) if self.path.is_file() else None
        self.manifest=json.loads(self.read('run_manifest.json'))
        definition=self.manifest['definition']
        self.inputs=definition['input_sha256']
        for name,sha in self.inputs.items():
            if digest(Path(data_dir)/name)!=sha:raise ValueError(f'Reference data differ: {name}')
        for name,sha in definition['source_sha256'].items():
            if Path(name).name!=name or digest(Path(__file__).parents[1]/'spend_baselines'/name)!=sha:
                raise ValueError(f'Reference implementation differs: {name}')
        self.settings=definition['config']
        self.train=read_objects(Path(data_dir)/'train.csv')
        self.results={}
        names=(self.zip.namelist() if self.zip else
               [str(p.relative_to(self.path)) for p in (self.path/'jobs').glob('*/result.json')])
        for name in names:
            if not name.startswith('jobs/') or not name.endswith('/result.json'):continue
            r=json.loads(self.read(name));j=r['job']
            if not approved_job(j,cfg):continue
            model=next(m for m,f in MODEL_FAMILIES.items() if f==j['family'])
            key=(model,j['date'])
            if key in self.results:raise ValueError('Duplicate eligible reference result')
            self.results[key]=(name.rsplit('/',1)[0],r)
        if not all((m,d) in self.results for m in cfg['models'] for d in cfg['validation_dates']):
            raise ValueError('Reference lacks the completed single-model course10 controls')

    def read(self,name):
        return self.zip.read(name) if self.zip else (self.path/name).read_bytes()

    def prediction(self,model,date):
        import hashlib
        directory,result=self.results[(model,date)]
        payload=self.read(directory+'/predictions.npz')
        if hashlib.sha256(payload).hexdigest()!=result['sha256']['predictions.npz']:
            raise ValueError('Corrupt reference predictions')
        with np.load(io.BytesIO(payload),allow_pickle=False) as a:
            if len(a['probability']):raise ValueError('A probability head is not a direct regressor')
            frame=pd.DataFrame({'Id':a['Id'],'customer_id':a['customer_id'],'z':a['z']})
        known=self.train[self.train.as_of_date.eq(date)].set_index('Id')
        if not frame.Id.is_unique or set(frame.Id)!=set(known.index):raise ValueError('Reference population differs')
        known=known.loc[frame.Id]
        if not np.array_equal(frame.customer_id.to_numpy(str),known.customer_id.to_numpy(str)):
            raise ValueError('Reference customer keys differ')
        frame['model'],frame['date'],frame['target_z']=model,date,np.log1p(known.spend_30d.to_numpy(float))
        if not np.isfinite(frame[['z','target_z']]).all().all() or frame.z.lt(0).any():
            raise ValueError('Invalid direct-regression forecasts')
        score=float(np.sqrt(np.mean((frame.z-frame.target_z)**2)))
        if abs(score-result['RMSLE'])>1e-10:raise ValueError('Reference RMSLE does not reproduce')
        return frame


def compare(reference,out_dir,cfg):
    frames=[]
    for model in cfg['models']:
        for date in cfg['validation_dates']:
            frame=reference.prediction(model,date)
            path=out_dir/'validation_predictions'/model/f'{date}.csv.gz'
            path.parent.mkdir(parents=True,exist_ok=True)
            frame.to_csv(path,index=False)
            frames.append(frame)
    rows=[]
    for model,frame in pd.concat(frames,ignore_index=True).groupby('model'):
        groups=list(frame.groupby('date'))
        for label,dates in [('pooled_SepOct',['2018-09-01','2018-10-01']),
                            (f"pooled_{len(cfg['validation_dates'])}_dates",cfg['validation_dates'])]:
            if set(dates)<=set(frame.date):groups.append((label,frame[frame.date.isin(dates)]))
        for date,g in groups:
            error=(g.z-g.target_z)**2
            rows.append({'model':model,'date':date,'n':len(g),'RMSLE':float(np.sqrt(error.mean())),
                         'squared_log_error':float(error.sum()),'single_seed':42,'manual_blend':False})
    table=pd.DataFrame(rows)
    table.to_csv(out_dir/'scores.csv',index=False)
    print(table[table.date.eq('pooled_SepOct')][['model','RMSLE']].to_string(index=False))
    atomic_json(out_dir/'comparison.json',{'evaluated_again':False,'new_model_fits':0,
                                          'config':cfg,'method_sources':METHOD_SOURCES})
    return table


def load_cache(path,reference):
    path=Path(path)
    ready=json.loads((path/'ready.json').read_text())
    if ready['columns']!=BASE_COLUMNS or ready['identity']['input_sha256']!=reference.inputs:
        raise ValueError('Feature cache differs from the verified original data')
    if ready['identity']['config'].get('smoke_customers'):
        raise ValueError('A complete cache is required')
    for name,sha in ready['sha256'].items():
        if Path(name).name!=name or digest(path/name)!=sha:raise ValueError('Corrupt feature cache')
    cache=CachedData(path)
    known=cache.objects.loc[cache.objects.official].set_index('Id')
    original=reference.train.set_index('Id')
    if not known.index.is_unique or set(known.index)!=set(original.index):
        raise ValueError('Cached official rows differ')
    original=original.loc[known.index]
    for name in ['customer_id','as_of_date','target_end']:
        if not np.array_equal(known[name].to_numpy(),original[name].to_numpy()):raise ValueError('Cached keys differ')
    np.testing.assert_allclose(known.spend_30d,original.spend_30d,atol=.011,rtol=0)
    return cache


def refit(args,reference,cfg):
    import joblib
    cache_path=args.base_cache or args.reference/'cache'
    cache=load_cache(cache_path,reference)
    objects=cache.objects
    vi=np.flatnonzero(objects.is_test.to_numpy())
    cutoff=objects.iloc[vi].as_of_date.min()
    if objects.iloc[vi].as_of_date.nunique()!=1:raise ValueError('Expected one test cutoff')
    tr=split_indices(objects,'D0',cutoff)
    if not objects.iloc[tr].official.all():raise ValueError('Only original monthly training rows are permitted')
    x=cache.matrix('F0',tr)[0];vx=cache.matrix('F0',vi)[0]
    y=objects.iloc[tr].spend_30d.to_numpy(float)
    family=MODEL_FAMILIES[args.model]
    dest=args.out_dir/'refit'/args.model
    dest.mkdir(parents=True,exist_ok=True)
    marker=dest/'result.json'
    identity={'model':args.model,'family':family,'task':'direct','seed':42,'dataset':'D0','features':'F0',
              'feature_columns':RFM_COLUMNS,'course_revision':REVISION,'max_seminar':10,
              'single_predictor':True,'target_history_features':False,'embedding':False,'manual_blend':False,
              'reference_manifest_sha256':digest(args.reference/'run_manifest.json') if args.reference.is_dir() else digest(args.reference),
              'cache_ready_sha256':digest(cache.directory/'ready.json'),
              'device':args.device if args.model=='mlp' else 'cpu','threads':args.threads,'smoke':args.smoke,
              'python':platform.python_version(),'libraries':{n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']},
              'cutoff':str(cutoff.date()),'source_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
    if args.model=='mlp':identity['libraries']['torch']=version('torch')
    if marker.exists():
        r=json.loads(marker.read_text())
        if not args.resume:raise FileExistsError('Completed refit exists; use --resume')
        if r['identity']!=identity:raise ValueError('Refit definition changed; use a new output directory')
        for name,sha in r['sha256'].items():
            if digest(dest/name)!=sha:raise ValueError('Corrupt completed refit')
        print(f"Verified completed course10 CSV: {dest/r['output_file']}")
        return r
    start=time.monotonic();metadata={}
    if args.model=='mlp':
        profile=next(p for p in reference.settings['profiles'] if p['name']==cfg['mlp_profile'])
        settings={**reference.settings['neural'],**{k:v for k,v in profile.items() if k!='name'}}
        if args.smoke:settings.update(max_epochs=2,patience=1)
        inner=objects.loc[objects.official&objects.target_end.le(cutoff),'as_of_date'].max()
        ti=split_indices(objects,'D0',inner)
        iv=np.flatnonzero((objects.official&objects.as_of_date.eq(inner)).to_numpy())
        selector=Neural('mlp','direct',settings,42,args.device,args.threads)
        selector.fit(cache.matrix('F0',ti)[0],objects.iloc[ti].spend_30d.to_numpy(float),np.ones(len(ti)),
                     validation=(cache.matrix('F0',iv)[0],objects.iloc[iv].spend_30d.to_numpy(float)))
        chosen=selector.best_epoch
        pd.DataFrame(selector.history).to_csv(dest/'selection.csv',index=False)
        del selector
        model=Neural('mlp','direct',settings,42,args.device,args.threads).fit(x,y,np.ones(len(tr)),epochs=chosen)
        if model.network.output.out_features!=1:raise ValueError('Expected one direct regression output')
        model.save(dest/'model.pt')
        pd.DataFrame(model.history).to_csv(dest/'refit.csv',index=False)
        metadata={'inner_date':str(inner.date()),'max_target_end_inner':str(objects.iloc[ti].target_end.max()),
                  'selected_epoch':chosen,'settings':settings}
    else:
        settings=reference.settings['tabular']
        model=Tabular(family,'direct',settings,42,args.threads).fit(x,y,np.ones(len(tr)))
        joblib.dump(model,dest/'model.joblib',compress=3)
        metadata={'settings':settings}
    fit_seconds=time.monotonic()-start
    start=time.monotonic();z=model.predict(vx);infer_seconds=time.monotonic()-start
    test=read_objects(args.data_dir/'test.csv')
    sample=pd.read_csv(args.data_dir/'sample_submission.csv',dtype={'Id':str})
    values=objects.iloc[vi][['Id','customer_id']].copy();values['z']=z
    if set(values.Id)!=set(test.Id):raise ValueError('Full test population required')
    values=values.set_index('Id').loc[test.Id]
    if not np.array_equal(values.customer_id.to_numpy(str),test.customer_id.to_numpy(str)):
        raise ValueError('Prediction customer keys differ')
    output_file='smoke_predictions.csv' if args.smoke else f'submission_course10_{args.model}_seed42.csv'
    csv=write_submission(test,values.z.to_numpy(),dest/output_file,sample_ids=sample.Id)
    if list(csv.columns)!=['Id','spend_30d'] or not csv.Id.equals(sample.Id):raise ValueError('CSV schema/order differs')
    np.savez_compressed(dest/'predictions.npz',Id=test.Id.to_numpy(str),customer_id=test.customer_id.to_numpy(str),z=values.z.to_numpy())
    # Smoke refits are explicitly labelled and must not be uploaded.
    r={'identity':identity,'method_sources':METHOD_SOURCES[args.model],**metadata,
       'output_file':output_file,'upload_ready':not args.smoke,
       'n_train':len(tr),'rows':len(test),'max_target_end_train':str(objects.iloc[tr].target_end.max()),
       'train_seconds':fit_seconds,'inference_seconds':infer_seconds,'kaggle_uploaded':False,
       'sha256':{p.name:digest(p) for p in dest.iterdir() if p.is_file() and p.name!='result.json'}}
    atomic_json(marker,r)
    print(f"Course10 single {args.model}: {dest/output_file} ({len(test)} rows); upload not performed")
    return r


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','compare','refit'])
    p.add_argument('--config',type=Path,default=Path('configs/course10_safe.json'))
    p.add_argument('--reference',type=Path,default=Path('outputs/v100-focused'))
    p.add_argument('--base-cache',type=Path)
    p.add_argument('--data-dir',type=Path,default=Path('data/raw'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs/course10-only'))
    p.add_argument('--model',choices=['ridge','forest','mlp'],default='mlp')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv);cfg=read_config(args.config)
    if args.command=='plan':
        print(json.dumps({'config':cfg,'method_sources':METHOD_SOURCES,
                          'validation_reused':True,'refit_predictors':1,'manual_ensembles':False},indent=2))
        return
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with (args.out_dir/'runner.lock').open('w') as lock,threadpool_limits(limits=args.threads):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reference=Reference(args.reference,args.data_dir,cfg)
        if args.command=='compare':compare(reference,args.out_dir,cfg)
        else:refit(args,reference,cfg)
