"""Standalone source for the stable Ridge with current/product-season aggregates."""
import argparse,ast,json,shutil,zipfile
from pathlib import Path
import numpy as np
from spend_baselines.suite_data import digest
from spend_course10.item_season_features import ITEM_SEASON_COLUMNS,CURRENT_COLUMNS,YEAR_COLUMNS

ROOT=Path(__file__).resolve().parents[1]


def functions(path,names):
    return '\n\n'.join(ast.unparse(n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',default='ridge_item_current')
    p.add_argument('--out',type=Path,default=Path('outputs/item-current-ridge-delivery'))
    args=p.parse_args();run=ROOT/'outputs/course10-item-season';d=run/'refit'/args.profile
    r=json.loads((d/'result.json').read_text());assert r['precision']=='float64' and r['single_predictor'] and r['seed']==42 and r['task']=='direct'
    for n,s in r['sha256'].items():assert digest(d/n)==s
    args.out.mkdir(parents=True,exist_ok=True)
    nb=json.loads((ROOT/'outputs/monthly-history-ridge-delivery/solution.ipynb').read_text())
    recipe={'profile':r['profile'],'settings':{'alpha':r['profile']['alpha']},'seed':42,'task':'direct','dataset':'D0','max_seminar':10,'feature_names':r['feature_names']}
    nb['cells'][0]['source']='''# Ridge на месячной истории и свойствах любимых товаров

Одна обычная Ridge (семинары2/6), MSE(log1p target), вычисление вfloat64.
Месячная история дополняется средними свойствами товаров клиента: популярностью и изменением спроса.
Свойства товаров считаются только по транзакциям строго до каждой даты, затем усредняются по прошлому
ассортименту клиента. Обучаемых товарных эмбеддингов нет. Внешние данные и прогнозы других моделей не используются.
Для расчётов нужны только четыре официальныхCSV; весь исходный код приведён ниже. Kaggle upload отсутствует.
Новые варианты выбраны по июню–августу доSep/Oct. Эти даты ранее использовались; небольшое локальное
улучшение по сравнению с измереннойRidge не доказано статистически. Код подготовлен с помощьюCodex.
'''
    nb['cells'][1]['source']='RECIPE='+repr(recipe)+'\nBASE_FEATURES='+repr(r['feature_names'][:15])+'\n'
    tree=ast.parse((ROOT/'src/spend_course10/item_season_features.py').read_text())
    tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='spend_embeddings.data')]
    item_source=functions(ROOT/'src/spend_behavior/data.py',{'positive_lines'})+functions(ROOT/'src/spend_embeddings/data.py',{'build_bags'})+ast.unparse(tree)+'\n'
    nb['cells'].insert(6,{'cell_type':'code','metadata':{},'id':'product-feature-source','source':item_source,'execution_count':None,'outputs':[]})
    import pandas as pd
    scores=pd.read_csv(run/'confirmation_scores.csv');row=scores[scores.profile.eq(args.profile)&scores.reference.eq('measured_ridge32')].iloc[0]
    nb['cells'][7]['source']=f"Проверка:pooledSep–Oct{row.RMSLE:.6f},разница сранееизмереннойRidge{row.delta:+.6f},95%customerbootstrapCI[{row.delta_ci_low:+.6f},{row.delta_ci_high:+.6f}]. Улучшение ещё нужно проверить наNovember.\n"
    text=nb['cells'][8]['source']
    addition='''
lines,item_start=positive_lines(DATA_DIR)
product=item_season_features(objects,lines,item_start,prior=5.,max_items=128)
item_names=ITEM_SEASON_COLUMNS if RECIPE['profile']['items']=='all' else CURRENT_COLUMNS if RECIPE['profile']['items']=='current' else YEAR_COLUMNS
X=np.c_[X,product[:,[ITEM_SEASON_COLUMNS.index(n) for n in item_names]]];names+=item_names
'''
    text=text.replace("assert names==RECIPE['feature_names']",addition+"\nassert names==RECIPE['feature_names']")
    text=text.replace('pre.transform(X[tr])','pre.transform(X[tr]).astype(np.float64)').replace('pre.transform(X[vi])','pre.transform(X[vi]).astype(np.float64)')
    nb['cells'][8]['source']=text
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(cell['source'],'item-ridge-notebook','exec')
    (args.out/'solution.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False))
    with np.load(ROOT/'outputs/monthly-history-ridge-delivery/feature_cache.npz',allow_pickle=False) as b,np.load(run/'item_features.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['Id'],b['Id'])
        selected=ITEM_SEASON_COLUMNS if r['profile']['items']=='all' else CURRENT_COLUMNS if r['profile']['items']=='current' else YEAR_COLUMNS
        x=np.c_[b['X'],a['X'][:,[ITEM_SEASON_COLUMNS.index(n) for n in selected]]]
        np.savez_compressed(args.out/'feature_cache.npz',Id=b['Id'],X=x)
    filename=f'submission_{args.profile}.csv';shutil.copyfile(d/'submission.csv',args.out/filename)
    for n in ['manifest.json','item_manifest.json','selection.json','early_scores.csv','confirmation_scores.csv','per_date_scores.csv']:
        shutil.copyfile(run/n,args.out/n)
    (args.out/'recipe.json').write_text(json.dumps(recipe,indent=2))
    (args.out/'requirements.txt').write_text('numpy==2.3.5\npandas==2.2.3\nscikit-learn==1.7.2\nscipy>=1.14,<2\n')
    (args.out/'README.txt').write_text('Put four official CSVs in data/raw and open solution.ipynb. All source is embedded. Optional feature cache is independently checked. CSV is ready; notebook creates submission_reproduced.csv. One Ridge, float64, no automatic upload.\n')
    receipt={'profile':args.profile,'CSV_SHA256':digest(d/'submission.csv'),'Kaggle_uploaded':False,'sha256':{f.name:digest(f) for f in args.out.iterdir() if f.is_file() and f.suffix!='.zip' and f.name!='delivery_manifest.json'}}
    (args.out/'delivery_manifest.json').write_text(json.dumps(receipt,indent=2))
    with zipfile.ZipFile(args.out/'item_ridge_solution.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(args.out.iterdir()):
            if f.is_file() and f.suffix!='.zip':z.write(f,f.name)
    print('READY',args.out,flush=True)


if __name__=='__main__':main()
