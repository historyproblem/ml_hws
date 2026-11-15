"""Two small product-embedding branches on the frozen Joint+history baseline."""

import argparse
import fcntl
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from spend_baselines.suite_data import atomic_json,digest,row_weights
from spend_tuning.data import FrozenRun,TuningData
from spend_tuning.suite import BudgetReached,Runner,temporal_split,write_csv

from .data import ProductBags
from .model import ProductJoint


def read_config(path):
    cfg=json.loads(Path(path).read_text())
    if cfg['version']!=1 or cfg['seeds']!=[42,123,2026] or cfg['base_settings']['hidden_sizes']!=[128,64]:
        raise ValueError('Use the frozen Joint history architecture and three seeds')
    if cfg['base_settings']['learning_rate']!=.001:
        raise ValueError('HP tuning did not confirm lr=.002; retain lr=.001')
    dims=[p['overrides']['embedding_dim'] for p in cfg['profiles']]
    if dims!=[16,32] or [p['name'] for p in cfg['profiles']]!=['item16','item32']:
        raise ValueError('Only the two fixed item16/item32 profiles are supported')
    e=cfg['embedding']
    if any(e[k]<=0 for k in ['window_days','half_life_days','max_items','min_customers','initial_std']) or e['weight_decay']<0:
        raise ValueError('Invalid embedding settings')
    if not cfg['validation_dates'] or len(set(cfg['validation_dates']))!=len(cfg['validation_dates']):
        raise ValueError('Empty/duplicate validation dates')
    if not all(0<w<1 for w in cfg['blend_weights']):raise ValueError('Invalid blend weight')
    return cfg


class EmbeddingRunner(Runner):
    def __init__(self,args,cfg,reference,data,bags):
        super().__init__(args,cfg,reference,data)
        self.bags=bags
        for settings in self.settings.values():settings['embedding']=cfg['embedding']

    def execute(self,name,date,seed,testing=False):
        old=self.completed(name,date,seed,testing)
        if old:
            if not self.args.resume and (name,date,seed,testing) not in self.created:
                raise FileExistsError('Completed job exists; use --resume')
            return old
        if self.args.max_minutes and time.monotonic()-self.started>=60*self.args.max_minutes:
            raise BudgetReached('Time budget reached between complete seed jobs')
        directory=self.directory(name,date,seed,testing)
        directory.mkdir(parents=True,exist_ok=True)
        tr,ti,iv,inner=temporal_split(self.data,date)
        vi=self.data.valid(date,testing)
        if not len(vi):raise ValueError('Empty validation/test population')
        obj=self.data.objects
        settings=self.settings[name]
        minimum=settings['embedding']['min_customers']
        print(f'{name} {date} seed {seed}: inner selection {inner.date()}',flush=True)
        start=time.monotonic()
        inner_map,_=self.bags.vocabulary(ti,minimum)
        iids,iw=self.bags.take(ti)
        vids,vw=self.bags.take(iv)
        selector=ProductJoint(settings,seed,self.args.device,self.args.threads)
        selector.fit(self.data.matrix(ti),iids,iw,obj.iloc[ti].spend_30d.to_numpy(float),
                     row_weights(obj,ti,'D2'),inner_map,
                     validation=(self.data.matrix(iv),vids,vw,obj.iloc[iv].spend_30d.to_numpy(float)))
        epoch=selector.best_epoch
        write_csv(directory/'selection.csv',pd.DataFrame(selector.history))
        del selector,iids,iw,vids,vw
        mapping,support=self.bags.vocabulary(tr,minimum)
        ids,weights=self.bags.take(tr)
        model=ProductJoint(settings,seed,self.args.device,self.args.threads)
        model.fit(self.data.matrix(tr),ids,weights,obj.iloc[tr].spend_30d.to_numpy(float),
                  row_weights(obj,tr,'D2'),mapping,epochs=epoch)
        del ids,weights
        write_csv(directory/'refit.csv',pd.DataFrame(model.history))
        model.save(directory/'model.pt',self.bags.item_names)
        seconds=time.monotonic()-start
        start=time.monotonic()
        ids,weights=self.bags.take(vi)
        z=model.predict(self.data.matrix(vi),ids,weights)
        infer=time.monotonic()-start
        del model
        frame=obj.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
        frame['candidate'],frame['date'],frame['z']=name,date,z
        control=None
        if testing:
            money=np.maximum(np.expm1(z),0)
            if not np.isfinite(money).all():raise ValueError('Non-finite money prediction')
            np.savez_compressed(directory/'predictions.npz',Id=frame.Id.to_numpy(str),
                                customer_id=frame.customer_id.to_numpy(str),z=z,spend_30d=money)
        else:
            control=self.control(date).set_index('Id').loc[frame.Id]
            frame['target_z']=np.log1p(obj.iloc[vi].spend_30d.to_numpy(float))
            frame['reference_z']=control.z.to_numpy()
            frame['purchase_days_180']=control.purchase_days_180.to_numpy()
            frame['delta_squared_log_error']=(frame.z-frame.target_z)**2-(frame.reference_z-frame.target_z)**2
            write_csv(directory/'predictions.csv.gz',frame)
        result={'job':{'profile':name,'date':date,'seed':seed,'testing':testing,'settings':settings},
                'inner_date':str(inner.date()),'selected_epoch':epoch,'n_train':len(tr),'n_inner':len(ti),
                'n_validation':len(vi),'max_target_end_train':str(obj.iloc[tr].target_end.max()),
                'max_target_end_inner':str(obj.iloc[ti].target_end.max()),
                'vocabulary_items':int(mapping.max()),'inner_vocabulary_items':int(inner_map.max()),
                'vocabulary_min_customers':minimum,
                'known_weight_fraction':float((weights*(mapping[ids]>0)).sum()/max(weights.sum(),1e-12)),
                'mean_retained_topk_mass':float(self.bags.diagnostics[self.bags.positions[vi],1].mean()),
                'train_seconds':seconds,'inference_seconds':infer,
                'sha256':{p.name:digest(p) for p in directory.iterdir() if p.is_file() and p.name!='result.json'}}
        if not testing:result['RMSLE']=float(np.sqrt(np.mean((z-frame.target_z.to_numpy())**2)))
        atomic_json(directory/'result.json',result)
        self.created.add((name,date,seed,testing))
        score='test' if testing else f"RMSLE={result['RMSLE']:.6f}"
        print(f'Completed {name} {date} seed {seed}: {score}, vocabulary={result["vocabulary_items"]}, fit={seconds:.1f}s',flush=True)
        return result

    def run(self):
        for date in self.cfg['validation_dates']:
            self.control(date)
            for name in self.settings:
                for seed in self.cfg['seeds']:self.execute(name,date,seed)
                self.mean3(name,date)

    def refit(self,profile):
        if profile not in self.settings:raise ValueError('Refit requires explicit --profile item16 or item32')
        date=str(self.data.objects.loc[self.data.objects.is_test,'as_of_date'].min().date())
        predictions=[]
        for seed in self.cfg['seeds']:
            self.execute(profile,date,seed,True)
            with np.load(self.directory(profile,date,seed,True)/'predictions.npz',allow_pickle=False) as a:
                predictions.append({k:a[k] for k in a.files})
        for a in predictions[1:]:
            if not np.array_equal(a['Id'],predictions[0]['Id']):raise ValueError('Unaligned refit seeds')
        z=np.mean([p['z'] for p in predictions],axis=0)
        path=self.args.out_dir/'refit'/profile
        path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/'predictions.npz',Id=predictions[0]['Id'],customer_id=predictions[0]['customer_id'],
                            z=z,spend_30d=np.maximum(np.expm1(z),0))
        atomic_json(path/'result.json',{'profile':profile,'cutoff':date,'submission_created':False,
                                       'predictions_sha256':digest(path/'predictions.npz')})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','run','report','export','refit'])
    p.add_argument('--config',type=Path,default=Path('configs/item_embeddings.json'))
    p.add_argument('--reference',type=Path,default=Path('outputs/behavior-v1'))
    p.add_argument('--base-cache',type=Path)
    p.add_argument('--data-dir',type=Path,default=Path('data/raw'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs/item-embeddings-v1'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-minutes',type=float,default=90.)
    p.add_argument('--profile')
    p.add_argument('--archive',type=Path,default=Path('item-embeddings-results.tar.gz'))
    args=p.parse_args(argv)
    cfg=read_config(args.config)
    if args.max_minutes<0:p.error('--max-minutes must be nonnegative')
    if args.command=='plan':
        jobs=len(cfg['profiles'])*len(cfg['seeds'])*len(cfg['validation_dates'])
        print(json.dumps({'profiles':cfg['profiles'],'dates':cfg['validation_dates'],
                          'seed_evaluations':jobs,'neural_fits':2*jobs,'baseline_retrained':False},indent=2))
        return
    if args.command=='export':
        from spend_tuning.report import export
        print(json.dumps(export(args.out_dir,args.archive),indent=2))
        return
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with (args.out_dir/'runner.lock').open('w') as lock,threadpool_limits(limits=args.threads):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reference=FrozenRun(args.reference,args.data_dir,cfg['base_settings'])
        data=TuningData(reference,args.base_cache,args.smoke)
        bags=ProductBags(data,args.data_dir,args.out_dir,cfg['embedding'])
        if args.smoke:cfg['bootstrap_repeats']=20
        sources=[q for package in ['spend_baselines','spend_behavior','spend_tuning','spend_embeddings']
                 for q in sorted((Path(__file__).parents[1]/package).glob('*.py'))]
        identity={'config':cfg,'smoke':args.smoke,'device':args.device,'threads':args.threads,
                  'python':platform.python_version(),'libraries':{n:version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch']},
                  'source_sha256':{f'{q.parent.name}/{q.name}':digest(q) for q in sources},
                  'reference_manifest_sha256':hashlib.sha256(reference.manifest_bytes).hexdigest(),
                  'reference_prediction_hashes':reference.prediction_hashes,'cache':data.identity,'products':bags.identity}
        marker=args.out_dir/'manifest.json'
        if marker.exists() and json.loads(marker.read_text())!=identity:
            raise ValueError('Run definition changed; use a new output directory')
        atomic_json(marker,identity)
        runner=EmbeddingRunner(args,cfg,reference,data,bags)
        complete=True
        try:
            if args.command=='run':runner.run()
            if args.command=='refit':runner.refit(args.profile)
        except BudgetReached as error:
            complete=False
            print(f'{error}; rerun the same command with --resume',flush=True)
        if args.command!='refit':
            from .report import report
            report(args.out_dir,cfg)
        atomic_json(args.out_dir/'status.json',{'phase':args.command,'phase_complete':complete,
                                               'elapsed_seconds':time.monotonic()-runner.started})
