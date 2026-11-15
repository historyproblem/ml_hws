"""Deadline-bounded manual tuning of one-output course1-10 regressors.

Other seeds assess training stability; forecasts are NEVER averaged.
"""

import argparse
import copy
import fcntl
import json
import multiprocessing as mp
import platform
import re
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from spend_baselines.cli import write_submission
from spend_baselines.data import read_objects
from spend_baselines.suite_data import CachedData,atomic_json,digest,split_indices
from spend_baselines.suite_models import Neural,Tabular
from spend_course10.suite import Reference,load_cache,read_config as scope_config

WORKER=None


def read_config(path):
    c=json.loads(Path(path).read_text())
    for k,v in {'version':1,'max_seminar':10,'dataset':'D0','features':'F0','task':'direct','final_seed':42,
                'diagnostic_seeds':[42,123,2026]}.items():
        if c.get(k)!=v:raise ValueError(f'Invalid strict course10 tuning field: {k}')
    if pd.to_datetime(c['tuning_dates']).max()+pd.Timedelta(days=30)>pd.to_datetime(c['confirmation_dates']).min():
        raise ValueError('Early labels overlap confirmation')
    allowed={'mlp':{'hidden_sizes','learning_rate','weight_decay','dropout','batch_size'},
             'forest':{'n_estimators','max_depth','min_samples_leaf','max_features'},'ridge':{'alpha'}}
    names=[]
    for p in c['profiles']:
        names.append(p['name'])
        if not re.fullmatch('[a-z0-9_]+',p['name']) or p['model'] not in allowed or not set(p['overrides'])<=allowed[p['model']]:
            raise ValueError('Unapproved predictor or parameter')
        if p.get('reference_model') and p['reference_model']!=p['model']:
            raise ValueError('Incorrect baseline reference mapping')
    if len(names)!=len(set(names)) or c['shortlist_per_family']!=2:raise ValueError('Invalid shortlist')
    return c


def score(frame,dates):
    a=frame[frame.date.isin(dates)]
    if set(a.date)!=set(dates) or not np.isfinite(a[['z','target_z']]).all().all():
        raise ValueError('Incomplete/nonfinite early predictions')
    return float(np.mean((a.z-a.target_z)**2))


def freeze(path,data):
    if path.exists() and json.loads(path.read_text())!=data:raise ValueError('Frozen selection was changed')
    atomic_json(path,data)


class Worker:
    def __init__(self,args,cfg,cache_path,deadline):
        self.args,self.cfg,self.cache,self.deadline=args,cfg,CachedData(cache_path),deadline
        self.objects=self.cache.objects
        self.active=np.ones(len(self.objects),bool)
        if args.smoke:
            import hashlib
            customers=sorted(self.objects.customer_id.unique(),key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:64]
            self.active=self.objects.customer_id.isin(customers).to_numpy()
        self.profiles={p['name']:p for p in cfg['profiles']}

    def indices(self,date,valid=False):
        ix=(np.flatnonzero((self.objects.official&self.objects.as_of_date.eq(date)).to_numpy()) if valid else
            split_indices(self.objects,'D0',date))
        return ix[self.active[ix]]

    def path(self,name,date,seed):return self.args.out_dir/'jobs'/name/date/str(seed)

    def settings(self,profile):
        model=profile['model']
        if model=='mlp':p={**self.cfg['neural_base'],**profile['overrides']}
        elif model=='forest':p={**self.cfg['forest_base'],**profile['overrides']}
        else:p={'alpha':profile['overrides']['alpha']}
        if self.args.smoke:
            if model=='mlp':p.update(max_epochs=2,patience=1)
            if model=='forest':p.update(n_estimators=3)
        return p

    def execute(self,job):
        name,date,seed=job
        directory=self.path(name,date,seed);marker=directory/'result.json'
        profile=self.profiles[name]
        spec={'profile':name,'model':profile['model'],'date':date,'seed':seed,'settings':self.settings(profile)}
        if marker.exists():
            result=json.loads(marker.read_text())
            if result['job']!=spec:raise ValueError('Changed job specification')
            for n,s in result['sha256'].items():
                if digest(directory/n)!=s:raise ValueError('Corrupt completed trial')
            return result
        if time.monotonic()>=self.deadline:return {'time_limit':True}
        directory.mkdir(parents=True,exist_ok=True)
        tr,vi=self.indices(date),self.indices(date,True)
        obj=self.objects;p=spec['settings'];model_type=profile['model']
        x,vx=self.cache.matrix('F0',tr)[0],self.cache.matrix('F0',vi)[0]
        y=obj.iloc[tr].spend_30d.to_numpy(float)
        print(f'Start {name} {date} seed{seed}',flush=True)
        start=time.monotonic();metadata={}
        if model_type=='mlp':
            inner=obj.loc[obj.official&obj.target_end.le(date),'as_of_date'].max()
            ti,iv=self.indices(inner),self.indices(inner,True)
            selector=Neural('mlp','direct',p,seed,self.args.device,self.args.threads)
            selector.fit(self.cache.matrix('F0',ti)[0],obj.iloc[ti].spend_30d.to_numpy(float),np.ones(len(ti)),
                         validation=(self.cache.matrix('F0',iv)[0],obj.iloc[iv].spend_30d.to_numpy(float)))
            epochs=selector.best_epoch
            pd.DataFrame(selector.history).to_csv(directory/'selection.csv',index=False)
            del selector
            model=Neural('mlp','direct',p,seed,self.args.device,self.args.threads).fit(x,y,np.ones(len(tr)),epochs=epochs)
            if model.network.output.out_features!=1:raise ValueError('Only one-output regression MLP is allowed')
            if model.network.output.out_features!=1:raise ValueError('Only one-output MLP is allowed')
            pd.DataFrame(model.history).to_csv(directory/'refit.csv',index=False)
            metadata={'inner_date':str(inner.date()),'selected_epoch':epochs,'max_target_end_inner':str(obj.iloc[ti].target_end.max())}
        else:
            settings={'ridge_alpha':p['alpha'] if model_type=='ridge' else 10.,'logistic_C':1.,
                      'forest':p if model_type=='forest' else self.cfg['forest_base']}
            model=Tabular('linear' if model_type=='ridge' else 'forest','direct',settings,seed,self.args.threads).fit(x,y,np.ones(len(tr)))
        trained=time.monotonic()-start
        start=time.monotonic();z=model.predict(vx);inferred=time.monotonic()-start
        frame=obj.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
        frame['date'],frame['profile'],frame['model'],frame['seed']=date,name,model_type,seed
        frame['z'],frame['target_z']=z,np.log1p(obj.iloc[vi].spend_30d.to_numpy(float))
        if not np.isfinite(z).all() or (z<0).any():raise ValueError('Invalid predictions')
        frame.to_csv(directory/'predictions.csv.gz',index=False)
        result={'job':spec,**metadata,'RMSLE':float(np.sqrt(np.mean((frame.z-frame.target_z)**2))),
                'n_train':len(tr),'max_target_end_train':str(obj.iloc[tr].target_end.max()),
                'train_seconds':trained,'inference_seconds':inferred,'reused_reference':False,
                'sha256':{'predictions.csv.gz':digest(directory/'predictions.csv.gz')}}
        for n in ['selection.csv','refit.csv']:
            if (directory/n).exists():result['sha256'][n]=digest(directory/n)
        atomic_json(marker,result)
        print(f"Done {name} {date} seed{seed}: {result['RMSLE']:.6f}, {trained:.1f}s",flush=True)
        return result


def init_worker(args,cfg,cache_path,deadline):
    global WORKER
    threadpool_limits(limits=args.threads)
    WORKER=Worker(args,cfg,cache_path,deadline)


def execute_job(job):return WORKER.execute(job)


def frame_for(worker,name,dates,seeds):
    frames=[]
    for date in dates:
        for seed in seeds:
            r=worker.path(name,date,seed)/'result.json'
            if not r.exists():raise ValueError('Required trial is incomplete')
            state=json.loads(r.read_text());path=r.parent/'predictions.csv.gz'
            if digest(path)!=state['sha256']['predictions.csv.gz']:raise ValueError('Corrupt trial forecast')
            frames.append(pd.read_csv(path,dtype={'Id':str,'customer_id':str}))
    return pd.concat(frames,ignore_index=True)


def rank_profiles(worker,names,dates,seeds):
    return sorted([{'profile':n,'MSE':score(frame_for(worker,n,dates,seeds),dates)} for n in names],key=lambda r:(r['MSE'],r['profile']))


def import_controls(worker,reference,cfg,args):
    if args.smoke:return
    current={n:version(n) for n in ['numpy','pandas','scikit-learn','scipy','torch']}
    if args.device!='cuda' or any(reference.manifest['libraries'].get(n)!=v for n,v in current.items()):
        print('Environment differs from old CUDA controls; fitting matched controls in this run',flush=True);return
    for profile in cfg['profiles']:
        if not profile.get('reference_model'):continue
        model=profile['model'];settings=worker.settings(profile)
        if model=='mlp':
            p=next(x for x in reference.settings['profiles'] if x['name']=='compact_l2')
            expected={**reference.settings['neural'],**{k:v for k,v in p.items() if k!='name'}}
        elif model=='forest':expected=reference.settings['tabular']['forest']
        else:expected={'alpha':reference.settings['tabular']['ridge_alpha']}
        if settings!=expected:raise ValueError('Baseline parameters differ from completed controls')
        for date in cfg['tuning_dates']+cfg['confirmation_dates']:
            d=worker.path(profile['name'],date,42)
            if (d/'result.json').exists():continue
            a=reference.prediction(profile['reference_model'],date)
            a['profile'],a['seed']=profile['name'],42
            d.mkdir(parents=True,exist_ok=True);a.to_csv(d/'predictions.csv.gz',index=False)
            atomic_json(d/'result.json',{'job':{'profile':profile['name'],'model':profile['model'],'date':date,'seed':42,
                'settings':worker.settings(profile)},'RMSLE':float(np.sqrt(np.mean((a.z-a.target_z)**2))),
                'train_seconds':0.,'inference_seconds':0.,'reused_reference':True,
                'sha256':{'predictions.csv.gz':digest(d/'predictions.csv.gz')}})


def report(worker,cfg):
    rows=[]
    for p in sorted((worker.args.out_dir/'jobs').glob('*/*/*/result.json')):
        r=json.loads(p.read_text());rows.append({**r['job'],'RMSLE':r['RMSLE'],'train_seconds':r['train_seconds'],
                                              'reused_reference':r.get('reused_reference',False)})
    pd.DataFrame(rows).drop(columns=['settings'],errors='ignore').to_csv(worker.args.out_dir/'trial_scores.csv',index=False)
    pooled=[]
    for name,p in worker.profiles.items():
        for label,dates in [('tuning',cfg['tuning_dates']),('confirmation',cfg['confirmation_dates']),
                            ('historical_november_audit',cfg.get('audit_dates',[]))]:
            if not dates:continue
            for seed in cfg['diagnostic_seeds']:
                try:a=frame_for(worker,name,dates,[seed])
                except ValueError:continue
                pooled.append({'profile':name,'model':p['model'],'stage':label,'seed':seed,
                               'RMSLE':np.sqrt(score(a,dates)),'n':len(a)})
    pd.DataFrame(pooled).to_csv(worker.args.out_dir/'scores.csv',index=False)
    # Confidence intervals use a SINGLE seed42 forecast, never an ensemble.
    from spend_behavior.report import comparison
    comparisons=[]
    selected=worker.args.out_dir/'winner.json'
    if selected.exists():
        winners=json.loads(selected.read_text())['family_winners']
        for name in winners.values():
            try:
                frame=frame_for(worker,name,cfg['confirmation_dates'],[42])
                base=frame_for(worker,'mlp_base',cfg['confirmation_dates'],[42]).set_index('Id').loc[frame.Id]
            except ValueError:continue
            frame['reference_z']=base.z.to_numpy()
            comparisons.append({'profile':name,**comparison(frame,20 if worker.args.smoke else 2000)})
    pd.DataFrame(comparisons).to_csv(worker.args.out_dir/'confirmation_scores.csv',index=False)


def refit_selected(worker,cfg,winner):
    import joblib
    args=worker.args;name=winner['profile'];profile=worker.profiles[name];p=worker.settings(profile)
    out=args.out_dir/'selected_refit';out.mkdir(exist_ok=True)
    marker=out/'result.json'
    if marker.exists():
        r=json.loads(marker.read_text())
        if r['profile']!=name:raise ValueError('Changed selected profile')
        for n,sha in r['sha256'].items():
            if digest(out/n)!=sha:raise ValueError('Corrupt selected refit')
        return
    obj=worker.objects;cutoff=obj.loc[obj.is_test,'as_of_date'].min()
    tr=worker.indices(cutoff);vi=np.flatnonzero(obj.is_test.to_numpy()&worker.active)
    y=obj.iloc[tr].spend_30d.to_numpy(float);x=worker.cache.matrix('F0',tr)[0]
    metadata={}
    if profile['model']=='mlp':
        inner=obj.loc[obj.official&obj.target_end.le(cutoff),'as_of_date'].max()
        ti,iv=worker.indices(inner),worker.indices(inner,True)
        selector=Neural('mlp','direct',p,42,args.device,args.threads)
        selector.fit(worker.cache.matrix('F0',ti)[0],obj.iloc[ti].spend_30d.to_numpy(float),np.ones(len(ti)),
                     validation=(worker.cache.matrix('F0',iv)[0],obj.iloc[iv].spend_30d.to_numpy(float)))
        chosen=selector.best_epoch;pd.DataFrame(selector.history).to_csv(out/'selection.csv',index=False)
        del selector
        model=Neural('mlp','direct',p,42,args.device,args.threads).fit(x,y,np.ones(len(tr)),epochs=chosen)
        if model.network.output.out_features!=1:raise ValueError('One regression output required')
        model.save(out/'model.pt');pd.DataFrame(model.history).to_csv(out/'refit.csv',index=False)
        metadata={'selected_epoch':chosen,'inner_date':str(inner.date())}
    else:
        settings={'ridge_alpha':p['alpha'] if profile['model']=='ridge' else 10.,'logistic_C':1.,
                  'forest':p if profile['model']=='forest' else cfg['forest_base']}
        model=Tabular('linear' if profile['model']=='ridge' else 'forest','direct',settings,42,args.threads).fit(x,y,np.ones(len(tr)))
        joblib.dump(model,out/'model.joblib',compress=3)
    z=model.predict(worker.cache.matrix('F0',vi)[0])
    test=obj.iloc[vi].copy().reset_index(drop=True)
    sample=pd.read_csv(args.data_dir/'sample_submission.csv',dtype={'Id':str})
    if args.smoke:sample=sample[sample.Id.isin(test.Id)].reset_index(drop=True)
    file='smoke_predictions.csv' if args.smoke else f"submission_course10_tuned_{profile['model']}_seed42.csv"
    write_submission(test,z,out/file,sample_ids=sample.Id)
    np.savez_compressed(out/'predictions.npz',Id=test.Id.to_numpy(str),customer_id=test.customer_id.to_numpy(str),z=z)
    atomic_json(marker,{'profile':name,'model':profile['model'],'settings':p,'seed':42,'predictor_count':1,'task':'direct',
                       'max_seminar':10,'smoke':args.smoke,'upload_ready':not args.smoke,**metadata,
                       'n_train':len(tr),'cutoff':str(cutoff.date()),'max_target_end_train':str(obj.iloc[tr].target_end.max()),
                       'dataset':'D0','features':'F0','target_history':False,'forecast_averaging':False,'kaggle_uploaded':False,
                       'sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='result.json'}})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','run','refit','export'])
    p.add_argument('--config',type=Path,default=Path('configs/course10_fast_tuning.json'))
    p.add_argument('--reference',type=Path,default=Path('outputs/v100-focused'))
    p.add_argument('--base-cache',type=Path)
    p.add_argument('--data-dir',type=Path,default=Path('data/raw'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs/course10-fast-tuning'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-minutes',type=float,default=65.)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--profile',help='Explicit approved profile for refit-only reproduction')
    p.add_argument('--archive',type=Path,default=Path('course10-tuning-results.tar.gz'))
    args=p.parse_args(argv);cfg=read_config(args.config)
    if args.command=='plan':
        print(json.dumps({'profiles':len(cfg['profiles']),'seed42_screening_jobs':3*len(cfg['profiles']),
                          'methods':['ridge','forest','direct_mlp'],'max_seminar':10,
                          'final_predictors':1,'forecast_averaging':False},indent=2));return
    if args.command=='export':
        export(args.out_dir,args.archive);return
    if args.workers<1 or args.max_minutes<=0:p.error('Use positive workers/time limit')
    if args.smoke:
        cfg=copy.deepcopy(cfg)
        cfg['profiles']=[x for x in cfg['profiles'] if x['name'] in ['mlp_base','mlp_lr_low','forest_base','forest_shallow','ridge_base','ridge_100']]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with (args.out_dir/'runner.lock').open('w') as lock,threadpool_limits(limits=args.threads):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reference=Reference(args.reference,args.data_dir,scope_config('configs/course10_safe.json'))
        cache=load_cache(args.base_cache or args.reference/'cache',reference)
        identity={'config':cfg,'device':args.device,'threads':args.threads,'smoke':args.smoke,
                  'python':platform.python_version(),'libraries':{n:version(n) for n in ['numpy','pandas','scikit-learn','scipy','torch']},
                  'source_sha256':digest(Path(__file__)),'cache_sha256':digest(cache.directory/'ready.json'),
                  'original_inputs':reference.inputs,'course_revision':scope_config('configs/course10_safe.json')['course_revision']}
        manifest=args.out_dir/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Definition changed; use a new output')
        atomic_json(manifest,identity)
        atomic_json(args.out_dir/'execution.json',{'workers':args.workers,'max_minutes':args.max_minutes})
        deadline=time.monotonic()+60*args.max_minutes
        worker=Worker(args,cfg,cache.directory,deadline)
        if args.command=='refit':
            if args.profile not in worker.profiles:raise ValueError('Use an explicit profile from the approved manual list')
            refit_selected(worker,cfg,{'profile':args.profile})
            return
        import_controls(worker,reference,cfg,args)
        complete=True;core_ready=False
        def run_jobs(jobs,pool,phase):
            nonlocal complete
            print(f'PHASE {phase}: {len(jobs)} seed jobs (completed controls are reused)',flush=True)
            futures=[pool.submit(execute_job,j) for j in jobs]
            done=0
            for f in as_completed(futures):
                r=f.result()
                if r.get('time_limit'):
                    complete=False
                    for q in futures:q.cancel()
                    return False
                done+=1
                atomic_json(args.out_dir/'status.json',{'phase':phase,'completed_in_phase':done,'planned_in_phase':len(jobs),
                    'minutes_to_time_limit':max((deadline-time.monotonic())/60,0),'complete':False})
            return True
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                                 initializer=init_worker,initargs=(args,cfg,str(cache.directory),deadline)) as pool:
            jobs=[(x['name'],d,42) for x in cfg['profiles'] for d in cfg['tuning_dates']]
            if run_jobs(jobs,pool,'screen'):
                selected={};rankings={}
                for model in ['mlp','forest','ridge']:
                    names=[x['name'] for x in cfg['profiles'] if x['model']==model]
                    ranking=rank_profiles(worker,names,cfg['tuning_dates'],[42]);rankings[model]=ranking
                    base=next(x['name'] for x in cfg['profiles'] if x['model']==model and x.get('reference_model'))
                    selected[model]=list(dict.fromkeys([r['profile'] for r in ranking[:2]]+[base])) if model!='ridge' else [ranking[0]['profile']]
                freeze(args.out_dir/'shortlist.json',{'profiles':selected,'ranking':rankings})
                jobs=[(name,d,s) for model,names in selected.items() if model!='ridge' for name in names
                      for d in cfg['tuning_dates'] for s in [123,2026]]
                if run_jobs(jobs,pool,'stability'):
                    winners={};scores={}
                    for model,names in selected.items():
                        seeds=[42] if model=='ridge' else cfg['diagnostic_seeds']
                        ranking=rank_profiles(worker,names,cfg['tuning_dates'],seeds)
                        winners[model]=ranking[0]['profile'];scores[model]=ranking[0]['MSE']
                    best=min(scores,key=lambda k:(scores[k],k))
                    winner={'profile':winners[best],'model':best,'family_winners':winners,'early_MSE':scores,
                            'final_seed':42,'forecast_averaging':False,'confirmation_used_for_selection':False}
                    freeze(args.out_dir/'winner.json',winner)
                    jobs=[(name,d,42) for name in winners.values() for d in cfg['confirmation_dates']]
                    core_ready=run_jobs(jobs,pool,'confirmation')
                    if core_ready and cfg.get('audit_dates'):
                        names=list(dict.fromkeys(list(winners.values())+['mlp_base']))
                        run_jobs([(n,d,42) for n in names for d in cfg['audit_dates']],pool,'historical_november_audit')
        report(worker,cfg)
        if core_ready and (args.out_dir/'winner.json').exists():
            refit_selected(worker,cfg,json.loads((args.out_dir/'winner.json').read_text()))
        atomic_json(args.out_dir/'status.json',{'complete':core_ready,'all_diagnostics_complete':complete,
                                              'remaining_kaggle_attempts_untouched':True})
        export(args.out_dir,args.archive)


def export(root,output):
    import tarfile,io,hashlib
    root,output=Path(root),Path(output)
    if output.exists():output=output.with_name(output.stem+f'-{int(time.time())}.tar.gz')
    receipt={'sha256':{},'completed_jobs':[],'unfinished_jobs':[],'weights_omitted':True}
    with tarfile.open(output,'w:gz') as archive:
        def add(name,payload):
            info=tarfile.TarInfo(name);info.size=len(payload)
            archive.addfile(info,io.BytesIO(payload));receipt['sha256'][name]=hashlib.sha256(payload).hexdigest()
        for name in ['manifest.json','execution.json','shortlist.json','winner.json','status.json','scores.csv','trial_scores.csv','confirmation_scores.csv']:
            if (root/name).is_file():add(name,(root/name).read_bytes())
        directories=list((root/'jobs').glob('*/*/*'))+[root/'selected_refit']
        for directory in sorted(directories):
            if not directory.is_dir():continue
            marker=directory/'result.json';key=str(directory.relative_to(root))
            if not marker.exists():receipt['unfinished_jobs'].append(key);continue
            payload=marker.read_bytes();state=json.loads(payload)
            files={'result.json':payload}
            for name,sha in state['sha256'].items():
                if Path(name).name!=name:raise ValueError('Unexpected export path')
                if Path(name).suffix in ['.pt','.joblib']:continue
                data=(directory/name).read_bytes()
                if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Corrupt export artifact')
                files[name]=data
            for name,data in files.items():add(key+'/'+name,data)
            receipt['completed_jobs'].append(key)
        payload=json.dumps(receipt,indent=2).encode();info=tarfile.TarInfo('export_manifest.json');info.size=len(payload)
        archive.addfile(info,io.BytesIO(payload))
    print(f'Archive ready: {output}',flush=True)


if __name__=='__main__':main()
