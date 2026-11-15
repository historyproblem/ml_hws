"""Package one followed-up Forest with all feature/model code in a notebook."""
import argparse
import ast
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from spend_baselines.suite_data import BASE_COLUMNS,CHANNELS,FEATURES,CachedData,digest

ROOT=Path(__file__).resolve().parents[1]


def extracted(path,names):
    tree=ast.parse(path.read_text())
    return '\n\n'.join(ast.unparse(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)+'\n'


def feature_source():
    helpers=extracted(ROOT/'src/spend_baselines/suite_data.py',{'digest','atomic_json','range_sum','distinct_ranges','prepare_cache'})
    # Only the raw-history feature builder is embedded; no advanced model imports.
    cache='''
class CachedData:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.objects=pd.read_csv(self.directory/'objects.csv.gz',dtype={'Id':'string','customer_id':'string'},parse_dates=['as_of_date','target_end'])
        self.features=np.load(self.directory/'features.npy',mmap_mode='r')
'''
    return 'import hashlib,json\nBASE_COLUMNS='+repr(BASE_COLUMNS)+'\nCHANNELS='+repr(CHANNELS)+'\n'+helpers+cache


RUN='''
DATA_DIR=Path('data/raw')
THREADS=2
train,test,history,checks=load_dataset(DATA_DIR)
assert len(train)==50852 and len(test)==3188
cache=prepare_cache(DATA_DIR,Path('reproduction_cache'),{'datasets':['D0']},
                    {'input_sha256':{n:digest(DATA_DIR/n) for n in ['train.csv','test.csv','transactions.csv','sample_submission.csv']}})
objects=cache.objects
names=list(RECIPE['base_feature_names'])
X=np.asarray(cache.features[:,[BASE_COLUMNS.index(n) for n in names]])
if RECIPE['profile']['season']:
    start=min(v[0][0] for v in history.customers.values())
    season=seasonal_features(objects,history,pd.Timestamp('1970-01-01')+pd.Timedelta(days=int(start)))
    X=np.c_[X,season];names+=SEASON_COLUMNS
assert names==RECIPE['feature_names']
# Exact portable cache is optional and verified independently against raw CSVs.
if Path('feature_cache.npz').exists():
    with np.load('feature_cache.npz',allow_pickle=False) as c:
        ix=pd.Index(c['Id']).get_indexer(objects.Id.to_numpy(str));assert (ix>=0).all()
        exact=c['X'][ix]
    np.testing.assert_allclose(X,exact,rtol=2e-6,atol=2e-6,equal_nan=True)
    X=exact
assert not np.isinf(X).any()
print(checks)
print(RECIPE['profile'],RECIPE['settings'],len(names),'features')

def predict(cutoff,indices):
    cutoff=pd.Timestamp(cutoff)
    tr=np.flatnonzero((objects.official&objects.target_end.le(cutoff)).to_numpy())
    assert objects.iloc[tr].target_end.max()<=cutoff
    w=np.ones(len(tr))
    half=RECIPE['profile']['half_life_days']
    if half:
        age=(cutoff-objects.iloc[tr].target_end).dt.total_seconds().to_numpy()/86400
        w=np.exp2(-age/half)
    preprocessing=Preprocessor().fit(X[tr],w)
    model=RandomForestRegressor(**RECIPE['settings'],random_state=42,n_jobs=THREADS)
    model.fit(preprocessing.transform(X[tr]),np.log1p(objects.iloc[tr].spend_30d.to_numpy(float)),sample_weight=w)
    z=np.maximum(model.predict(preprocessing.transform(X[indices])),0)
    assert np.isfinite(z).all()
    return z

scores=[]
for date in ['2018-09-01','2018-10-01']:
    vi=np.flatnonzero((objects.official&objects.as_of_date.eq(date)).to_numpy())
    z=predict(date,vi);error=(z-np.log1p(objects.iloc[vi].spend_30d.to_numpy(float)))**2
    scores.append({'date':date,'n':len(vi),'RMSLE':float(np.sqrt(error.mean())),'squared_log_error':float(error.sum())})
print(pd.DataFrame(scores).to_string(index=False))
print('Pooled RMSLE:',np.sqrt(sum(r['squared_log_error'] for r in scores)/sum(r['n'] for r in scores)))

vi=np.flatnonzero(objects.is_test.to_numpy());cutoff=objects.iloc[vi].as_of_date.min()
z=predict(cutoff,vi)
a=pd.DataFrame({'Id':objects.iloc[vi].Id.to_numpy(str),'spend_30d':np.maximum(np.expm1(z),0)})
sample=pd.read_csv(DATA_DIR/'sample_submission.csv',dtype={'Id':str})
a=a.set_index('Id').loc[sample.Id].reset_index()
assert len(a)==3188 and a.Id.is_unique and np.isfinite(a.spend_30d).all() and a.spend_30d.ge(0).all()
a.to_csv('submission_reproduced.csv',index=False)
print('Created submission_reproduced.csv; no automatic Kaggle upload.')
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',required=True)
    p.add_argument('--run',type=Path,default=Path('outputs/course10-forest-followup'))
    p.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    p.add_argument('--data',type=Path,default=Path('data/raw'))
    p.add_argument('--out',type=Path,default=Path('outputs/course10-season-forest-delivery'))
    args=p.parse_args();rdir=args.run/'refit'/args.profile;r=json.loads((rdir/'result.json').read_text())
    assert r['seed']==42 and r['single_predictor'] and r['task']=='direct' and r['max_seminar']==10 and r['dataset']=='D0'
    assert r['n_train']==50852 and pd.Timestamp(r['max_target_end'])<=pd.Timestamp(r['cutoff'])
    for n,s in r['sha256'].items():assert digest(rdir/n)==s
    sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str});a=pd.read_csv(rdir/'submission.csv',dtype={'Id':str})
    assert a.Id.equals(sample.Id) and a.Id.is_unique and len(a)==3188 and list(a.columns)==['Id','spend_30d']
    assert np.isfinite(a.spend_30d).all() and a.spend_30d.ge(0).all()
    args.out.mkdir(parents=True,exist_ok=True)
    recipe={k:r[k] for k in ['profile','settings','feature_names','seed','max_seminar','task','dataset']}
    recipe['base_feature_names']=list(FEATURES[r['profile']['features']])
    cells=[]
    def add(kind,text):
        c={'cell_type':kind,'metadata':{},'id':f'cell-{len(cells):03d}','source':text.strip()+'\n'}
        if kind=='code':c.update(execution_count=None,outputs=[]);compile(c['source'],'solution-cell','exec')
        cells.append(c)
    add('markdown',f'''# HW3: один RandomForest, {args.profile}

Один RandomForestRegressor, seed42, MSE по log1p(target), затем expm1 и нижняя граница0.
Метод: [семинар9](https://github.com/andriygav/MachineLearningSeminars/blob/8a20cdf4b981c2962ef9741821b8d811468cf849/sem9/main.ipynb).
Признаки — агрегаты официальных транзакций строго до cutoff, включая прошлогоднее окно и известный календарь.
Используются исходные50852строки D0. Все функции очистки, признаков и модели приведены ниже.
Прогнозы моделей/seed не смешиваются. Внешние данные, бустинг, AutoML, Joint и neural embeddings не используются.

Предыдущий F0Forest получил public2.75074 по сообщению пользователя. Новый кандидат выбран офлайн,
без подбора множителя по leaderboard: параметры сравнивались на июне–августе, затем проверялись
на сентябре–октябре. Эти даты уже использовались раньше, поэтому проверка не является новым untouched holdout.
Метки ноября2018 неизвестны; выигрыш на нём не доказан. Непокрытое прошлогоднее окно отличается от нулевой активности.
При подготовке кода и анализе использовался Codex.''')
    add('code','RECIPE='+repr(recipe))
    add('markdown','## Очистка и проверка целей\nЦели восстанавливаются для аудита, но никогда не входят в признаки.')
    add('code',(ROOT/'src/spend_baselines/data.py').read_text())
    add('markdown','## Признаки\nБазовые окна, частота, корзины и регулярность считаются по прошлому. Построитель кеша настроен только на D0; последовательные массивы не используются моделью.')
    add('code',feature_source())
    import runpy
    helpers=runpy.run_path(str(ROOT/'scripts/course10_forest_followup.py'),run_name='forest_delivery_helpers')
    season_source='SEASON_COLUMNS='+repr(helpers['SEASON_COLUMNS'])+'\n'+extracted(ROOT/'scripts/course10_forest_followup.py',{'seasonal_features'})
    add('code',season_source)
    add('markdown','## Обычный RandomForest и временная проверка\nPreprocessing fit только на train. Train: target_end<=cutoff, valid: as_of_date=cutoff. При наличии точного кеша он сверяется с сырыми данными; платформы могут давать немного разные деревья.')
    add('code','from sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import RandomForestRegressor\n'+extracted(ROOT/'src/spend_baselines/suite_models.py',{'Preprocessor'}))
    if (args.run/'confirmation_scores.csv').exists():
        scores=pd.read_csv(args.run/'confirmation_scores.csv')
        row=scores[scores.profile.eq(args.profile)].iloc[0]
        add('markdown',f"Зафиксированная проверка Sep/Oct: RMSLE **{row.RMSLE:.6f}**, разница с отправленным F0Forest **{row.delta:+.6f}**, условный95%customer-cluster bootstrap CI **[{row.delta_ci_low:+.6f}, {row.delta_ci_high:+.6f}]**. Интервал не корректируется за повторный анализ дат.")
    add('code',RUN)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'cells':cells}
    (args.out/'solution.ipynb').write_text(json.dumps(notebook,indent=1,ensure_ascii=False))
    (args.out/'recipe.json').write_text(json.dumps(recipe,indent=2))
    shutil.copyfile(rdir/'submission.csv',args.out/'submission.csv');shutil.copyfile(rdir/'result.json',args.out/'refit_result.json')
    for name in ['manifest.json','early_selection.json','early_scores.csv','confirmation_scores.csv','confirmation_dates.csv']:
        if (args.run/name).exists():shutil.copyfile(args.run/name,args.out/name)
    cache=CachedData(args.cache);ix=np.flatnonzero((cache.objects.official|cache.objects.is_test).to_numpy())
    x=cache.matrix(r['profile']['features'],ix)[0]
    if r['profile']['season']:x=np.c_[x,np.load(args.run/'season.npy',allow_pickle=False)[ix]]
    np.savez_compressed(args.out/'feature_cache.npz',Id=cache.objects.iloc[ix].Id.to_numpy(str),X=x)
    (args.out/'requirements.txt').write_text('numpy==2.3.5\npandas==2.2.3\nscikit-learn==1.7.2\nscipy>=1.14,<2\n')
    (args.out/'README.txt').write_text('Put four official CSVs in data/raw and open solution.ipynb. All source is embedded. No repository/network required. submission.csv is ready; notebook reproduces submission_reproduced.csv. No automatic Kaggle upload.\n')
    receipt={'model':'RandomForestRegressor','profile':args.profile,'seed':42,'predictor_count':1,'Kaggle_uploaded':False,
        'sha256':{q.name:digest(q) for q in args.out.iterdir() if q.is_file() and q.suffix!='.zip' and q.name!='delivery_manifest.json'}}
    (args.out/'delivery_manifest.json').write_text(json.dumps(receipt,indent=2))
    archive=args.out/'course10_season_forest_solution.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for q in sorted(args.out.iterdir()):
            if q.is_file() and q.suffix!='.zip':z.write(q,q.name)
    print('Ready:',archive,flush=True)


if __name__=='__main__':main()
