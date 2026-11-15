"""Standalone source notebook for one monthly-history Ridge or RandomForest."""
import argparse,ast,json,runpy,shutil,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from spend_baselines.suite_data import FEATURES,CachedData,digest
from spend_course10.history_features import HISTORY_COLUMNS,LAG_COLUMNS

ROOT=Path(__file__).resolve().parents[1]


def extracted(path,names):
    return '\n\n'.join(ast.unparse(n) for n in ast.parse(path.read_text()).body
                       if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)+'\n'


SETUP='''
DATA_DIR=Path('data/raw')
train,test,history,checks=load_dataset(DATA_DIR)
objects=pd.concat([train.assign(official=True,is_test=False),test.assign(official=False,is_test=True)],ignore_index=True).sort_values(['as_of_date','customer_id']).reset_index(drop=True)
if RECIPE['profile']['base']=='F0':
    X=feature_matrix(history.build_features(objects)).astype(np.float32)
else:
    cache=prepare_cache(DATA_DIR,Path('reproduction_cache'),{'datasets':['D0']},
        {'input_sha256':{n:digest(DATA_DIR/n) for n in ['train.csv','test.csv','transactions.csv','sample_submission.csv']}})
    objects=cache.objects
    X=np.asarray(cache.features[:,[BASE_COLUMNS.index(n) for n in BASE_FEATURES]])
names=list(BASE_FEATURES)
start=min(v[0][0] for v in history.customers.values())
start=pd.Timestamp('1970-01-01')+pd.Timedelta(days=int(start))
if RECIPE['profile']['season']:
    X=np.c_[X,seasonal_features(objects,history,start)];names+=SEASON_COLUMNS
if RECIPE['profile']['history']!='none':
    full=monthly_history_features(objects,history,start)
    cols=np.arange(len(HISTORY_COLUMNS)) if RECIPE['profile']['history']=='full' else np.arange(len(LAG_COLUMNS),len(HISTORY_COLUMNS))
    X=np.c_[X,full[:,cols]];names+=[HISTORY_COLUMNS[k] for k in cols]
assert names==RECIPE['feature_names'] and not np.isinf(X).any()
if Path('feature_cache.npz').exists():
    with np.load('feature_cache.npz',allow_pickle=False) as c:
        ix=pd.Index(c['Id']).get_indexer(objects.Id.to_numpy(str));assert (ix>=0).all()
        exact=c['X'][ix]
    np.testing.assert_allclose(X,exact,rtol=2e-6,atol=2e-6,equal_nan=True)
    X=exact
print(checks)
print(RECIPE['profile'],RECIPE['settings'],len(names),'causal features plus train-only missing indicators')

def predict(cutoff,vi):
    cutoff=pd.Timestamp(cutoff)
    tr=np.flatnonzero((objects.official&objects.target_end.le(cutoff)).to_numpy())
    assert objects.iloc[tr].target_end.max()<=cutoff
    w=np.ones(len(tr));pre=Preprocessor().fit(X[tr],w)
    if RECIPE['profile']['model']=='ridge':model=Ridge(**RECIPE['settings'])
    else:model=RandomForestRegressor(**RECIPE['settings'],random_state=42,n_jobs=2)
    model.fit(pre.transform(X[tr]),np.log1p(objects.iloc[tr].spend_30d.to_numpy(float)),sample_weight=w)
    z=np.maximum(np.asarray(model.predict(pre.transform(X[vi])),dtype=float),0)
    assert np.isfinite(z).all()
    return z

scores=[]
for date in ['2018-09-01','2018-10-01']:
    vi=np.flatnonzero((objects.official&objects.as_of_date.eq(date)).to_numpy())
    z=predict(date,vi);err=(z-np.log1p(objects.iloc[vi].spend_30d.to_numpy(float)))**2
    scores.append({'date':date,'n':len(vi),'RMSLE':float(np.sqrt(err.mean())),'squared_log_error':float(err.sum())})
print(pd.DataFrame(scores).to_string(index=False))
print('Pooled RMSLE:',np.sqrt(sum(r['squared_log_error'] for r in scores)/sum(r['n'] for r in scores)))
vi=np.flatnonzero(objects.is_test.to_numpy());z=predict(objects.iloc[vi].as_of_date.min(),vi)
a=pd.DataFrame({'Id':objects.iloc[vi].Id.to_numpy(str),'spend_30d':np.maximum(np.expm1(z),0)})
sample=pd.read_csv(DATA_DIR/'sample_submission.csv',dtype={'Id':str});a=a.set_index('Id').loc[sample.Id].reset_index()
assert len(a)==3188 and a.Id.is_unique and np.isfinite(a.spend_30d).all() and a.spend_30d.ge(0).all()
a.to_csv('submission_reproduced.csv',index=False)
print('Created submission_reproduced.csv; no automatic Kaggle upload.')
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('outputs/course10-monthly-history'))
    p.add_argument('--profile',default='ridge_full');p.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    p.add_argument('--out',type=Path,default=Path('outputs/monthly-history-ridge-delivery'))
    args=p.parse_args();d=args.run/'refit'/args.profile;r=json.loads((d/'result.json').read_text())
    assert r['single_predictor'] and r['seed']==42 and r['task']=='direct' and r['dataset']=='D0' and r['n_train']==50852
    for n,s in r['sha256'].items():assert digest(d/n)==s
    args.out.mkdir(parents=True,exist_ok=True);recipe={k:r[k] for k in ['profile','settings','feature_names','seed','dataset','task','max_seminar']}
    cells=[]
    def add(kind,text):
        c={'cell_type':kind,'metadata':{},'id':f'cell-{len(cells):03d}','source':text.strip()+'\n'}
        if kind=='code':c.update(execution_count=None,outputs=[]);compile(c['source'],'monthly-notebook-cell','exec')
        cells.append(c)
    add('markdown','''# Прогноз покупок: месячная история и обычный регрессор

Один предиктор, исходные50852строкиD0, целеваяпеременнаяlog1p(spend_30d), MSE.
Ridge/L2 — семинары2/6; RandomForest — семинар9. Все признаки являются агрегатами официальных прошлых транзакций.
ПомимоRFM/календаря используются полностью завершённые30-дневные окна за прошлыйгод,
ихсредниеlog-расходы, долянулевыхокон, регулярность ификсированноевзвешенноесреднее.
Официальныеtarget-столбцы не читаются построителем признаков. Объект использует толькоevent_ts<as_of_date.
Непокрытыйпрошлыйинтервал даётNaN с флагом, а наблюдаемоеотсутствиепокупок — настоящий0.
Всефункцииочистки/признаков/модели доступны ниже; четыреCSV — единственный источникданных.
Прогнозы разныхмоделей/seed не смешиваются. Приподготовке использовалсяCodex.

Подбор идёт наиюне–августе2018, затемпроверкаSep/Oct. Датыранееиспользовались, интервалыусловные.
НовоеулучшениеNovemberне доказано доотправки. Ноябрьскиймножитель поleaderboard не подбирался.''')
    add('code','RECIPE='+repr(recipe)+'\nBASE_FEATURES='+repr(list(FEATURES[r['profile']['base']])))
    add('code',(ROOT/'src/spend_baselines/data.py').read_text())
    if r['profile']['base']!='F0':
        helpers=runpy.run_path(str(ROOT/'scripts/package_course10_forest_followup.py'),run_name='base_feature_source')
        add('code',helpers['feature_source']())
    helpers=runpy.run_path(str(ROOT/'scripts/course10_forest_followup.py'),run_name='season_source')
    add('code','SEASON_COLUMNS='+repr(helpers['SEASON_COLUMNS'])+'\n'+extracted(ROOT/'scripts/course10_forest_followup.py',{'seasonal_features'}))
    add('code',(ROOT/'src/spend_course10/history_features.py').read_text())
    add('code','from sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import Ridge\nfrom sklearn.ensemble import RandomForestRegressor\n'+extracted(ROOT/'src/spend_baselines/suite_models.py',{'Preprocessor'}))
    scores=pd.read_csv(args.run/'confirmation_scores.csv');row=scores[scores.profile.eq(args.profile)].iloc[0]
    add('markdown',f"Исходнаяпроверка:pooledSep–Oct{row.RMSLE:.6f},разница ссезоннымForest{row.delta:+.6f},95%customerbootstrapCI[{row.delta_ci_low:+.6f},{row.delta_ci_high:+.6f}].")
    add('code',SETUP)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'cells':cells}
    (args.out/'solution.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False));(args.out/'recipe.json').write_text(json.dumps(recipe,indent=2))
    name=f"submission_monthly_{args.profile}.csv";shutil.copyfile(d/'submission.csv',args.out/name)
    base=CachedData(args.cache);ix=np.flatnonzero((base.objects.official|base.objects.is_test).to_numpy())
    x=base.matrix(r['profile']['base'],ix)[0]
    with np.load(args.run/'history_features.npz',allow_pickle=False) as f:
        assert np.array_equal(f['Id'],base.objects.iloc[ix].Id.to_numpy(str))
        if r['profile']['season']:x=np.c_[x,f['season']]
        if r['profile']['history']!='none':
            cols=np.arange(len(HISTORY_COLUMNS)) if r['profile']['history']=='full' else np.arange(len(LAG_COLUMNS),len(HISTORY_COLUMNS))
            x=np.c_[x,f['monthly'][:,cols]]
    np.savez_compressed(args.out/'feature_cache.npz',Id=base.objects.iloc[ix].Id.to_numpy(str),X=x)
    for name in ['manifest.json','selection.json','early_scores.csv','confirmation_scores.csv','per_date_scores.csv','pooled5_scores.csv']:
        if (args.run/name).exists():shutil.copyfile(args.run/name,args.out/name)
    (args.out/'requirements.txt').write_text('numpy==2.3.5\npandas==2.2.3\nscikit-learn==1.7.2\nscipy>=1.14,<2\n')
    (args.out/'README.txt').write_text('Put four official CSVs in data/raw and open solution.ipynb. All source is embedded; no repository/network needed. feature_cache.npz is optional and independently checked against raw history. CSV is ready; notebook creates submission_reproduced.csv. No automatic Kaggle upload.\n')
    receipt={'profile':args.profile,'seed':42,'single_predictor':True,'Kaggle_uploaded':False,'CSV_SHA256':digest(d/'submission.csv'),
             'sha256':{q.name:digest(q) for q in args.out.iterdir() if q.is_file() and q.suffix!='.zip' and q.name!='delivery_manifest.json'}}
    (args.out/'delivery_manifest.json').write_text(json.dumps(receipt,indent=2))
    with zipfile.ZipFile(args.out/'monthly_history_solution.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for q in sorted(args.out.iterdir()):
            if q.is_file() and q.suffix!='.zip':z.write(q,q.name)
    print('READY',args.out,flush=True)


if __name__=='__main__':main()
