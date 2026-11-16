"""A bounded calendar/market-regime study; checked CSVs, no notebooks."""
import argparse
import hashlib
import json
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from spend_baselines.cli import write_submission
from spend_baselines.data import load_dataset
from spend_baselines.suite_data import atomic_json, digest
from spend_baselines.suite_models import Neural, Preprocessor
from spend_behavior.report import comparison
from spend_course10.centred_neural import CentredNeural
from spend_course10.market_prior import market_forecasts, centre_features_by_date, centre_targets_by_date
from run_monthly_history import frozen, prepare


def indices(objects, cutoff, valid=False):
    mask = objects.official & (objects.as_of_date.eq(cutoff) if valid else objects.target_end.le(cutoff))
    return np.flatnonzero(mask.to_numpy())


def matrix(x, names, profile, dates):
    calendar = {f'month_{m}' for m in range(1, 13)}
    remove = set(calendar)
    if profile['remove'] in ('calendar', 'season_calendar'):
        remove |= {'month_sin', 'month_cos'}
    if profile['remove'] == 'season_calendar':
        remove |= {n for n in names if n.startswith('year_')}
    if profile['remove'] == 'none':
        remove = set()
    selected = [j for j, n in enumerate(names) if n not in remove]
    a = x[:, selected]
    if profile['centred']:
        a = centre_features_by_date(a, dates)
    return a, [names[j] for j in selected]


def predict(model, profile, x, forecast):
    if profile['model'] == 'ridge':
        raw = model['model'].predict(model['preprocessor'].transform(x).astype(np.float64))
        raw = raw - raw.mean() if profile['centred'] else raw
    else:
        raw = model.predict(x)
    return np.maximum(raw + (forecast if profile['centred'] else 0), 0)


def fit_one(args, cfg, objects, x, forecast, profile, cutoff, vi, directory):
    tr = indices(objects, cutoff)
    assert len(tr) and objects.iloc[tr].target_end.max() <= pd.Timestamp(cutoff)
    target = np.log1p(objects.spend_30d.to_numpy(float))
    start = time.monotonic()
    metadata = {}
    directory.mkdir(parents=True, exist_ok=True)
    if profile['model'] == 'mlp':
        settings = dict(cfg['neural_base'])
        if args.smoke:
            settings.update(max_epochs=2, patience=1)
        inner = objects.iloc[tr].as_of_date.max()
        ti, iv = indices(objects, inner), indices(objects, inner, True)
        assert objects.iloc[iv].target_end.max() <= pd.Timestamp(cutoff)
        cls = CentredNeural if profile['centred'] else Neural
        selector = cls('mlp', 'direct', settings, 42, args.device, args.threads)
        if profile['centred']:
            residual = centre_targets_by_date(target[ti], objects.iloc[ti].as_of_date.to_numpy())
            selector.fit_residual(x[ti], residual, np.ones(len(ti)), validation=(x[iv], target[iv] - forecast[iv]))
        else:
            selector.fit(x[ti], objects.iloc[ti].spend_30d.to_numpy(float), np.ones(len(ti)),
                         validation=(x[iv], objects.iloc[iv].spend_30d.to_numpy(float)))
        epochs = selector.best_epoch
        pd.DataFrame(selector.history).to_csv(directory/'selection.csv', index=False)
        del selector
        model = cls('mlp', 'direct', settings, 42, args.device, args.threads)
        if profile['centred']:
            residual = centre_targets_by_date(target[tr], objects.iloc[tr].as_of_date.to_numpy())
            model.fit_residual(x[tr], residual, np.ones(len(tr)), epochs=epochs)
        else:
            model.fit(x[tr], objects.iloc[tr].spend_30d.to_numpy(float), np.ones(len(tr)), epochs=epochs)
        assert model.network.output.out_features == 1
        model.save(directory/'model.pt')
        pd.DataFrame(model.history).to_csv(directory/'refit.csv', index=False)
        metadata.update(selected_epoch=epochs, inner_date=str(inner), settings=settings)
    else:
        pre = Preprocessor().fit(x[tr], np.ones(len(tr)))
        y = centre_targets_by_date(target[tr], objects.iloc[tr].as_of_date.to_numpy()) if profile['centred'] else target[tr]
        estimator = Ridge(alpha=profile['alpha']).fit(pre.transform(x[tr]).astype(np.float64), y)
        model = {'preprocessor': pre, 'model': estimator}
        joblib.dump(model, directory/'model.joblib', compress=3)
        metadata['settings'] = {'alpha': profile['alpha'], 'precision': 'float64'}
    fit_seconds = time.monotonic() - start
    start = time.monotonic()
    z = predict(model, profile, x[vi], forecast[vi])
    inference_seconds = time.monotonic() - start
    reloaded = joblib.load(directory/'model.joblib') if profile['model']=='ridge' else cls.load(directory/'model.pt', args.device, args.threads)
    np.testing.assert_array_equal(z, predict(reloaded, profile, x[vi], forecast[vi]))
    assert np.isfinite(z).all() and (z >= 0).all()
    return z, {**metadata, 'fit_seconds': fit_seconds, 'inference_seconds': inference_seconds,
               'n_train': len(tr), 'max_target_end': str(objects.iloc[tr].target_end.max()),
               'checkpoint_forward_exact': True}


def trial(args, cfg, objects, x, forecast, profile, date):
    directory = args.out/'jobs'/profile['name']/date
    marker = directory/'result.json'
    if marker.exists():
        r = json.loads(marker.read_text())
        assert r['profile'] == profile
        for name, sha in r['sha256'].items():
            assert digest(directory/name) == sha
        return pd.read_csv(directory/'predictions.csv.gz', dtype={'Id': str, 'customer_id': str})
    vi = indices(objects, date, True)
    # Reuse the exact measured champion's two existing outer forecasts.
    old = args.reference/'jobs/monthly_l2_003'/date/'42'
    if profile['name'] == 'reference' and (old/'result.json').exists() and not args.smoke:
        r = json.loads((old/'result.json').read_text())
        assert r['job']['settings'] == cfg['neural_base']
        assert digest(old/'predictions.csv.gz') == r['sha256']['predictions.csv.gz']
        frame = pd.read_csv(old/'predictions.csv.gz', dtype={'Id': str, 'customer_id': str})
        assert set(frame.Id) == set(objects.iloc[vi].Id)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {'reused_reference': True, 'fit_seconds': r['train_seconds'], 'inference_seconds': r['inference_seconds'],
                'n_train': r['n_train'], 'max_target_end': r['max_target_end_train']}
    else:
        z, meta = fit_one(args, cfg, objects, x, forecast, profile, date, vi, directory)
        frame = objects.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
        frame['z'] = z
        frame['target_z'] = np.log1p(objects.iloc[vi].spend_30d.to_numpy(float))
    frame['date'] = date
    frame['profile'] = profile['name']
    frame['market_forecast'] = float(forecast[vi][0])
    frame.to_csv(directory/'predictions.csv.gz', index=False)
    score = float(np.sqrt(np.mean((frame.z-frame.target_z)**2)))
    atomic_json(marker, {'profile': profile, 'RMSLE': score, **meta,
                        'sha256': {p.name: digest(p) for p in directory.iterdir() if p.name!='result.json'}})
    print(profile['name'], date, f'{score:.6f}', flush=True)
    return frame


def refit(args, cfg, objects, x, names, forecast, profile):
    directory = args.out/'refit'/profile['name']
    marker = directory/'result.json'
    if marker.exists():
        r = json.loads(marker.read_text())
        assert r['profile'] == profile
        for name, sha in r['sha256'].items():
            assert digest(directory/name) == sha
        return r
    vi = np.flatnonzero(objects.is_test.to_numpy())
    cutoff = objects.iloc[vi].as_of_date.min()
    z, meta = fit_one(args, cfg, objects, x, forecast, profile, cutoff, vi, directory)
    sample = pd.read_csv(args.data/'sample_submission.csv', dtype={'Id': str})
    if args.smoke:
        sample = sample[sample.Id.isin(objects.iloc[vi].Id)]
    filename = ('smoke_' if args.smoke else 'submission_')+profile['name']+f'_{args.device}.csv'
    write_submission(objects.iloc[vi], z, directory/filename, sample_ids=sample.Id)
    exported = pd.read_csv(directory/filename, dtype={'Id':str})
    assert exported.Id.tolist() == sample.Id.tolist()
    assert np.isfinite(exported.spend_30d).all() and (exported.spend_30d >= 0).all()
    ordered = pd.Series(z, index=objects.iloc[vi].Id).loc[sample.Id].to_numpy()
    np.testing.assert_allclose(np.log1p(exported.spend_30d), ordered, rtol=0, atol=1e-12)
    np.savez_compressed(directory/'inputs_predictions.npz', Id=objects.iloc[vi].Id.to_numpy(str),
                        X=x[vi], z=z, market_forecast=forecast[vi])
    result = {'profile': profile, 'feature_names': names, 'n_test':len(vi), 'seed':42,
              'smoke':args.smoke, 'upload_ready':not args.smoke, 'Kaggle_uploaded':False,
              'market_forecast':float(forecast[vi][0]), 'mean_prediction_z':float(z.mean()), **meta,
              'prediction_rule':'max(0, market_forecast + centred_residual)' if profile['centred'] else 'max(0, MLP(X))',
              'sha256':{p.name:digest(p) for p in directory.iterdir() if p.name!='result.json'}}
    atomic_json(marker, result)
    if not args.smoke:
        args.quick.mkdir(parents=True,exist_ok=True)
        dest = args.quick/filename
        if dest.exists() and digest(dest) != digest(directory/filename):
            raise ValueError('Refusing to replace a different submission')
        shutil.copyfile(directory/filename, dest)
        atomic_json(args.quick/(filename+'.json'), result)
    print('CSV_READY', directory/filename, flush=True)
    return result


def report(args, objects, profiles, frames):
    base = frames['reference'].set_index('Id')
    scores, slices, bias = [], [], []
    from spend_baselines.suite_data import BASE_COLUMNS, CachedData
    cached=CachedData(args.cache)
    counts=pd.Series(np.rint(np.expm1(cached.features[:,BASE_COLUMNS.index('n_180')])),index=cached.objects.Id)
    for profile in profiles:
        a=frames[profile['name']].copy()
        reference=base.loc[a.Id]
        np.testing.assert_allclose(reference.target_z, a.target_z, rtol=0, atol=1e-12)
        a['reference_z']=reference.z.to_numpy()
        groups=[(d,g) for d,g in a.groupby('date')]
        groups += [('pooled_SepOct',a[a.date.isin(['2018-09-01','2018-10-01'])]),
                   ('pooled_five',a[a.date.str.startswith('2018-')])]
        for date,g in groups:
            scores.append({'profile':profile['name'],'date':date,**comparison(g,2000 if not args.smoke else 30)})
            n=counts.loc[g.Id].to_numpy()
            for label,mask in [('days_0',n==0),('days_1_2',(n>=1)&(n<=2)),('days_3_plus',n>=3),
                               ('zero',g.target_z.to_numpy()==0),('positive',g.target_z.to_numpy()>0)]:
                slices.append({'profile':profile['name'],'date':date,'slice':label,'n':int(mask.sum()),
                               'RMSLE':float(np.sqrt(np.mean((g.z.to_numpy()[mask]-g.target_z.to_numpy()[mask])**2))) if mask.any() else None})
            bias.append({'profile':profile['name'],'date':date,'mean_z':float(g.z.mean()),
                         'target_mean_z':float(g.target_z.mean()),'mean_error':float((g.z-g.target_z).mean())})
    pd.DataFrame(scores).to_csv(args.out/'scores.csv',index=False)
    pd.DataFrame(slices).to_csv(args.out/'slices.csv',index=False)
    pd.DataFrame(bias).to_csv(args.out/'bias.csv',index=False)
    print(pd.DataFrame(scores).query('date == "pooled_SepOct"').to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/course10_season_transfer.json'))
    parser.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'))
    parser.add_argument('--reference',type=Path,default=Path('outputs/mlp-regularization-neighbours'))
    parser.add_argument('--out',type=Path,default=Path('outputs/season-transfer'))
    parser.add_argument('--quick',type=Path,default=Path('outputs/quick-variants-20261116'))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--report-only',action='store_true',help='Read verified completed forecasts; do not fit or alter the study manifest')
    parser.add_argument('--refit-only',action='store_true',help='Only the inner selection and final refit; no outer validation fits')
    parser.add_argument('--profile',help='One predefined candidate, for example transfer_market_mlp')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    assert cfg['dataset']=='D0' and cfg['task']=='direct' and cfg['max_seminar']==10 and cfg['seed']==42
    if args.profile and args.profile not in {p['name'] for p in cfg['profiles']}:
        raise ValueError('Unknown predefined profile')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        from spend_baselines.suite_data import CachedData
        stored=json.loads((args.out/'study.json').read_text())
        assert stored['config']==cfg and stored['smoke']==args.smoke
        cache=CachedData(args.cache)
        objects=cache.objects[cache.objects.official|cache.objects.is_test].reset_index(drop=True)
        profiles=[{'name':'reference','model':'mlp','remove':'none','centred':False}]+cfg['profiles']
        frames={}
        for profile in profiles:
            chunks=[]
            for date in cfg['validation_dates']:
                directory=args.out/'jobs'/profile['name']/date
                r=json.loads((directory/'result.json').read_text())
                assert r['profile']==profile
                for name,sha in r['sha256'].items():assert digest(directory/name)==sha
                chunks.append(pd.read_csv(directory/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}))
            frames[profile['name']]=pd.concat(chunks,ignore_index=True)
        report(args,objects,profiles,frames)
        atomic_json(args.out/'report_verification.json',{'source_sha256':digest(__file__),
                    'fit_manifest_sha256':digest(args.out/'study.json'),'no_fits':True,'verified_jobs':sum(len(cfg['validation_dates']) for _ in profiles)})
        return
    source=['scripts/run_season_transfer.py','src/spend_course10/market_prior.py','src/spend_course10/centred_neural.py',
            'src/spend_baselines/suite_models.py','scripts/run_monthly_history.py']
    frozen(args.out/'study.json',{'config':cfg,'device':args.device,'threads':args.threads,'smoke':args.smoke,
           'python':platform.python_version(),'libraries':{n:version(n) for n in ['numpy','pandas','scikit-learn','torch']},
           'source_sha256':{n:digest(n) for n in source},'forecast_averaging':False})
    feature_cfg={**cfg,'profiles':[{'name':'raw','base':'F0','season':True,'history':'full'}]}
    with threadpool_limits(limits=args.threads):
        objects,matrices=prepare(args,feature_cfg)
        x,names=matrices['raw']
        _,_,history,checks=load_dataset(args.data)
        start=min(v[0][0] for v in history.customers.values())
        start=pd.Timestamp('1970-01-01')+pd.Timedelta(days=int(start))
        forecast,market=market_forecasts(objects,history,start)
        market.to_csv(args.out/'market_forecasts.csv')
        atomic_json(args.out/'target_checks.json',checks)
        regime=[]
        for date,ix in objects.groupby('as_of_date').groups.items():
            regime.append({'date':str(date.date()),'n':len(ix),'year30_known':float(np.isfinite(x[ix,names.index('year_spend30')]).mean()),
                           'lag12_known':float(np.isfinite(x[ix,names.index('past_log_spend_12')]).mean())})
        pd.DataFrame(regime).to_csv(args.out/'history_regimes.csv',index=False)
        if args.smoke:
            keep=sorted(objects.customer_id.unique(),key=lambda v:hashlib.sha256(str(v).encode()).hexdigest())[:64]
            ix=np.flatnonzero(objects.customer_id.isin(keep).to_numpy())
            objects=objects.iloc[ix].reset_index(drop=True);x=x[ix];forecast=forecast[ix]
        reference={'name':'reference','model':'mlp','remove':'none','centred':False}
        profiles=[reference]+cfg['profiles']
        frames={}
        if args.refit_only:
            for profile in cfg['profiles']:
                if args.profile and args.profile!=profile['name']:continue
                a,cols=matrix(x,names,profile,objects.as_of_date.to_numpy())
                refit(args,cfg,objects,a,cols,forecast,profile)
            return
        for profile in profiles:
            a,cols=matrix(x,names,profile,objects.as_of_date.to_numpy())
            frames[profile['name']]=pd.concat([trial(args,cfg,objects,a,forecast,profile,date) for date in cfg['validation_dates']],ignore_index=True)
            if profile['name']!='reference':
                refit(args,cfg,objects,a,cols,forecast,profile)
        report(args,objects,profiles,frames)


if __name__=='__main__':
    main()
