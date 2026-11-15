"""Create a readable, standalone course1-10 notebook and a compact delivery ZIP."""

import argparse
import ast
import json
import shutil
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from spend_baselines.data import RFM_COLUMNS
from spend_baselines.suite_data import BASE_COLUMNS,CachedData,atomic_json,digest

REPO=Path(__file__).resolve().parents[1]


class DirectOnly(ast.NodeTransformer):
    """Specialise our own generic trainer to an ordinary one-output MLP."""
    def visit_Attribute(self,node):
        if isinstance(node.ctx,ast.Load) and isinstance(node.value,ast.Name) and node.value.id=='self':
            if node.attr in {'task','architecture','sequence_shape'}:
                return ast.copy_location(ast.Constant({'task':'direct','architecture':'mlp','sequence_shape':None}[node.attr]),node)
        return self.generic_visit(node)

    def visit_FunctionDef(self,node):
        if node.name=='predict_probability':return None
        return self.generic_visit(node)

    def visit_Compare(self,node):
        node=self.generic_visit(node)
        if len(node.ops)==1 and isinstance(node.left,ast.Constant) and isinstance(node.comparators[0],ast.Constant):
            a,b=node.left.value,node.comparators[0].value;op=node.ops[0]
            if isinstance(op,ast.Eq):return ast.copy_location(ast.Constant(a==b),node)
            if isinstance(op,ast.NotEq):return ast.copy_location(ast.Constant(a!=b),node)
            if isinstance(op,ast.Is):return ast.copy_location(ast.Constant(a is b),node)
            if isinstance(op,ast.IsNot):return ast.copy_location(ast.Constant(a is not b),node)
        return node

    def visit_If(self,node):
        node=self.generic_visit(node)
        if isinstance(node.test,ast.Constant):return node.body if node.test.value else node.orelse
        return node

    def visit_IfExp(self,node):
        node=self.generic_visit(node)
        if isinstance(node.test,ast.Constant):return node.body if node.test.value else node.orelse
        return node


def model_source():
    parsed=ast.parse((REPO/'src/spend_baselines/suite_models.py').read_text())
    pre=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name=='Preprocessor')
    trainer=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name=='Neural')
    trainer=DirectOnly().visit(trainer);trainer.name='DirectMLP'
    ctor=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    ctor.body.insert(0,ast.parse("assert architecture == 'mlp' and task == 'direct' and sequence_shape is None").body[0])
    ast.fix_missing_locations(trainer)
    helpers='''
import os, random
from copy import deepcopy
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def make_network(n_features, architecture, task, settings, sequence_shape=None):
    import torch
    assert architecture == 'mlp' and task == 'direct' and sequence_shape is None
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers=[];previous=n_features
            for width in settings['hidden_sizes']:
                layers.extend([torch.nn.Linear(previous,width),torch.nn.ReLU(),torch.nn.Dropout(settings['dropout'])])
                previous=width
            self.body=torch.nn.Sequential(*layers)
            self.output=torch.nn.Linear(previous,1)
        def forward(self,x):return self.output(self.body(x))
    return Network()

def output_prediction(raw,task):
    assert task == 'direct'
    return raw[:,0].clamp_min(0)

def neural_loss(raw,y,w,task):
    import torch
    assert task == 'direct'
    return (((raw[:,0]-torch.log1p(y))**2)*w).sum()/w.sum().clamp_min(1e-12)
'''
    result=textwrap.dedent(helpers)+'\n'+ast.unparse(pre)+'\n\n'+ast.unparse(trainer)+'\n'
    assert all(s not in result for s in ['hurdle','ziln','Embedding','GRU','LSTM','predict_probability'])
    return result


REPRODUCTION='''
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THREADS = 2
DATA_DIR = Path('data/raw')  # Put the four official CSV files here.
train, test, history, checks = load_dataset(DATA_DIR)
objects = pd.concat([train.assign(is_test=False),test.assign(is_test=True)],ignore_index=True)
objects = objects.sort_values(['as_of_date','customer_id']).reset_index(drop=True)
features = history.build_features(objects)
X = feature_matrix(features, 'rfm').astype(np.float32)
assert X.shape[1] == 15 and np.isfinite(X).all()
# Optional exact cache, included in the delivery ZIP; independently check it against raw history.
if Path('rfm_cache.npz').exists():
    with np.load('rfm_cache.npz',allow_pickle=False) as cached:
        order = pd.Index(cached['Id']).get_indexer(objects.Id.to_numpy(str))
        assert (order >= 0).all()
        exact = cached['X'][order]
    np.testing.assert_allclose(X,exact,rtol=2e-6,atol=2e-6)
    X = exact
print(checks)
print('Model:', RECIPE['model'], 'seed:', 42, 'features:', RFM_COLUMNS)
print('Settings:', RECIPE['settings'])

def fit_predict(cutoff, prediction_indices):
    cutoff = pd.Timestamp(cutoff)
    tr = np.flatnonzero((~objects.is_test & objects.target_end.le(cutoff)).to_numpy())
    y = objects.iloc[tr].spend_30d.to_numpy(float)
    settings = RECIPE['settings']
    if RECIPE['model'] == 'mlp':
        inner = objects.loc[~objects.is_test & objects.target_end.le(cutoff),'as_of_date'].max()
        ti = np.flatnonzero((~objects.is_test & objects.target_end.le(inner)).to_numpy())
        iv = np.flatnonzero((~objects.is_test & objects.as_of_date.eq(inner)).to_numpy())
        selector = DirectMLP('mlp','direct',settings,42,DEVICE,THREADS)
        selector.fit(X[ti],objects.iloc[ti].spend_30d.to_numpy(float),np.ones(len(ti)),
                     validation=(X[iv],objects.iloc[iv].spend_30d.to_numpy(float)))
        epochs = selector.best_epoch
        del selector
        model = DirectMLP('mlp','direct',settings,42,DEVICE,THREADS)
        model.fit(X[tr],y,np.ones(len(tr)),epochs=epochs)
        assert model.network.output.out_features == 1
        z = model.predict(X[prediction_indices])
    else:
        preprocessing = Preprocessor().fit(X[tr],np.ones(len(tr)))
        a = preprocessing.transform(X[tr])
        if RECIPE['model'] == 'ridge': model = Ridge(alpha=settings['alpha'])
        elif RECIPE['model'] == 'forest': model = RandomForestRegressor(**settings,random_state=42,n_jobs=THREADS)
        else: raise ValueError('Unapproved model')
        model.fit(a,np.log1p(y),sample_weight=np.ones(len(tr)))
        z = np.maximum(model.predict(preprocessing.transform(X[prediction_indices])),0)
    assert np.isfinite(z).all() and (z >= 0).all()
    return z
'''

VALIDATE='''
scores=[]
for date in ['2018-09-01','2018-10-01']:
    vi=np.flatnonzero((~objects.is_test & objects.as_of_date.eq(date)).to_numpy())
    z=fit_predict(date,vi)
    error=(z-np.log1p(objects.iloc[vi].spend_30d.to_numpy(float)))**2
    scores.append({'date':date,'n':len(vi),'squared_log_error':float(error.sum()),'RMSLE':float(np.sqrt(error.mean()))})
display(pd.DataFrame(scores))
print('Pooled RMSLE:', np.sqrt(sum(r['squared_log_error'] for r in scores)/sum(r['n'] for r in scores)))
'''

FINAL='''
vi=np.flatnonzero(objects.is_test.to_numpy())
z=fit_predict('2018-11-01',vi)
predictions=pd.DataFrame({'Id':objects.iloc[vi].Id.to_numpy(str),'spend_30d':np.maximum(np.expm1(z),0)})
sample=pd.read_csv(DATA_DIR/'sample_submission.csv',dtype={'Id':str})
assert predictions.Id.is_unique and set(predictions.Id)==set(sample.Id)
predictions=predictions.set_index('Id').loc[sample.Id].reset_index()
assert np.isfinite(predictions.spend_30d).all() and predictions.spend_30d.ge(0).all()
predictions.to_csv('submission_reproduced.csv',index=False)
print('Created submission_reproduced.csv:',len(predictions),'rows. No automatic Kaggle upload.')
'''


def notebook(recipe,metrics):
    cells=[]
    def add(kind,text):
        c={'cell_type':kind,'metadata':{},'id':f'cell-{len(cells):03d}','source':text.strip()+'\n'}
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    add('markdown',f"""# HW3: прогноз покупок клиента за30дней

Финальная модель: **{recipe['model']}**, фиксированный seed42,15RFM-признаков.
Модель и параметры зафиксированы в следующей ячейке. Прогнозы разных seed
не усредняются. Внешние данные, бустинг, AutoML, Joint/hurdle и нейронные
товарные ветки не используются. Правило допустимости: семинары1–10.

Источники: [L2/линейная регрессия,2](https://github.com/andriygav/MachineLearningSeminars/blob/master/sem2/main.ipynb),
[MLP,3](https://github.com/andriygav/MachineLearningSeminars/blob/master/sem3/main.ipynb),
[MLP-регрессия/MSE,8](https://github.com/andriygav/MachineLearningSeminars/blob/master/sem8/main.ipynb),
[RandomForest,9](https://github.com/andriygav/MachineLearningSeminars/blob/master/sem9/main.ipynb).

Метрика RMSLE соответствует MSE по log1p(target). После прогноза выполняется
обратное expm1 с ограничением снизу0. Данные берутся только из четырёх
официальных CSV. Очистка/проверка известных таргетов и все функции ниже
представлены открытым исходным кодом; интернет для выполнения не нужен.

При подготовке кода, проверке данных и планировании экспериментов использовался
Codex. Для защиты необходимо понимать очистку, признаки, временной purge,
функцию потерь и причины выбора параметров. Public score не заменяет валидацию.""")
    add('code','RECIPE = '+repr(recipe))
    add('markdown','## Подбор параметров\nПроцедура использует заранее заданный ручной список параметров MLP, RandomForest и Ridge. Основной подбор идёт на июне–августе2018; September/October используются после фиксации настроек. Дополнительные seed оценивают устойчивость по ошибкам отдельных запусков и не усредняются в прогнозе. Финальный seed42 фиксирован заранее. Вариант mlp_base может служить исходным резервным контролем; факт завершения подбора и его результаты фиксируются отдельно. Полный список и полученные таблицы включаются в delivery ZIP при упаковке. На Kaggle модель автоматически не отправляется.')
    add('markdown','## Очистка и причинные признаки\nОперации используются строго до даты объекта. Все50852 известных ответа восстанавливаются из истории. В модель входят только RFM_COLUMNS, без других возвращаемых служебных статистик.')
    add('code',(REPO/'src/spend_baselines/data.py').read_text())
    add('markdown','## Обычные регрессоры\nMLP имеет один линейный выход. Adam/L2/dropout — обычные параметры сети. Ограничение нормы градиента — численная защита оптимизации. Подбор эпох использует только более раннюю дату, затем модель обучается с нуля.')
    add('code',model_source())
    add('markdown','## Данные и параметры запуска\nУкажите путь к четырём официальным CSV. При наличии rfm_cache.npz его значения сверяются с расчётом из сырой истории. Для точного воспроизведения исходного запуска предпочтительны прежние версии библиотек и V100; CPU/GPU могут давать разные траектории обучения.')
    add('code',REPRODUCTION)
    add('markdown','## Временная проверка\nTrain: target_end<=cutoff. Valid: as_of_date=cutoff. Число эпох выбирается на более ранней внутренней дате. Сентябрь/октябрь уже просматривались, оценка не является новым untouched holdout.')
    if metrics:add('markdown','Зафиксированные результаты исходного запуска:\n\n'+metrics)
    add('code',VALIDATE)
    add('markdown','## Финальный рефит\nОдна модель с seed42 обучается на всех доступных к2018-11-01 метках. CSV содержит все3188Id в порядке sample. Эта ячейка создаёт файл и не отправляет его в Kaggle.')
    add('code',FINAL)
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
             'language_info':{'name':'python','version':'3.12'}},'cells':cells}


def package(args,kind):
    if kind=='selected':
        directory=args.run_dir/'selected_refit';r=json.loads((directory/'result.json').read_text())
        if r.get('smoke') or not r.get('upload_ready') or r.get('predictor_count')!=1 or r.get('seed')!=42:
            raise ValueError('Not a completed full single-model selected refit')
        if any(r.get(k)!=v for k,v in {'max_seminar':10,'task':'direct','dataset':'D0','features':'F0',
                                      'target_history':False,'forecast_averaging':False}.items()):
            raise ValueError('Selected recipe is outside the strict course scope')
        recipe={k:r[k] for k in ['profile','model','settings','seed']}
        csvs=[p for p in directory.glob('submission_course10_tuned_*.csv')]
    else:
        directory=args.fallback_dir;r=json.loads((directory/'result.json').read_text());identity=r['identity']
        if identity.get('smoke') or not r.get('upload_ready') or not identity.get('single_predictor') or identity['seed']!=42:
            raise ValueError('Not a completed full fallback refit')
        if any(identity.get(k)!=v for k,v in {'max_seminar':10,'task':'direct','dataset':'D0','features':'F0',
                     'target_history_features':False,'embedding':False,'manual_blend':False}.items()):
            raise ValueError('Fallback recipe is outside the strict course scope')
        if identity['model']!='mlp':raise ValueError('Fallback must be the approved ordinary MLP')
        recipe={'profile':'mlp_base','model':'mlp','settings':r['settings'],'seed':42}
        csvs=[directory/r['output_file']]
    if recipe['model'] not in ['ridge','forest','mlp'] or len(csvs)!=1:raise ValueError('Unexpected final model/CSV')
    keys={'ridge':{'alpha'},'forest':{'n_estimators','max_depth','min_samples_leaf','max_features'},
          'mlp':{'hidden_sizes','learning_rate','weight_decay','dropout','batch_size','max_epochs','patience','min_delta'}}
    if set(recipe['settings'])!=keys[recipe['model']]:raise ValueError('Unexpected model extension')
    for n,s in r['sha256'].items():
        if digest(directory/n)!=s:raise ValueError('Corrupt final artifact')
    sample=pd.read_csv(args.data_dir/'sample_submission.csv',dtype={'Id':str})
    forecast=pd.read_csv(csvs[0],dtype={'Id':str})
    if list(forecast.columns)!=['Id','spend_30d'] or not forecast.Id.equals(sample.Id) or not forecast.Id.is_unique:
        raise ValueError('Final CSV does not match official sample order/schema')
    if not np.isfinite(forecast.spend_30d).all() or forecast.spend_30d.lt(0).any():raise ValueError('Invalid final values')
    out=args.output_dir/kind;out.mkdir(parents=True,exist_ok=True)
    metrics=''
    score_path=args.run_dir/'scores.csv'
    if score_path.exists():
        a=pd.read_csv(score_path)
        a=a[a.profile.eq(recipe['profile'])&a.seed.eq(42)]
        metrics='| Этап | RMSLE |\n|---|---:|\n'+'\n'.join(f'| {x.stage} | {x.RMSLE:.6f} |' for x in a.itertuples())
        a.to_csv(out/'validation_scores.csv',index=False)
    nb=notebook(recipe,metrics)
    choice_path=args.run_dir/'final_choice.json'
    if kind=='selected' and choice_path.exists():
        choice=json.loads(choice_path.read_text())
        if choice['profile']!=recipe['profile'] or choice['seed']!=42:
            raise ValueError('Final choice note belongs to another predictor')
        if choice.get('explanation'):
            nb['cells'].insert(3,{'cell_type':'markdown','metadata':{},'id':'final-choice-audit',
                                  'source':choice['explanation']+'\n'})
    grid=json.loads((REPO/'configs/course10_fast_tuning.json').read_text())
    insert={'cell_type':'code','metadata':{},'id':'declared-parameter-grid','execution_count':None,'outputs':[],
            'source':'# Declared manual search plan; not automatically rerun for reproduction.\nMANUAL_SEARCH_PLAN = '+repr(grid)+'\n'}
    nb['cells'].insert(3,insert)
    if score_path.exists():
        table=pd.read_csv(score_path)
        table=table[table.stage.eq('tuning')&table.seed.eq(42)].sort_values('RMSLE')
        text='## Результаты раннего подбора, seed42\n\n| Profile | RMSLE |\n|---|---:|\n'
        text+='\n'.join(f'| {x.profile} | {x.RMSLE:.6f} |' for x in table.itertuples())
        nb['cells'].insert(-4,{'cell_type':'markdown','metadata':{},'id':'search-results','source':text})
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(cell['source'],'notebook-cell','exec')
    (out/'solution.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    atomic_json(out/'model_recipe.json',recipe)
    shutil.copyfile(csvs[0],out/'submission.csv')
    shutil.copyfile(REPO/'configs/course10_fast_tuning.json',out/'manual_parameter_grid.json')
    shutil.copyfile(REPO/'docs/course10-method-proof.json',out/'course_method_sources.json')
    for name in ['scores.csv','confirmation_scores.csv','winner.json','final_choice.json','local_validation.json']:
        if (args.run_dir/name).exists():shutil.copyfile(args.run_dir/name,out/name)
    cache=CachedData(args.base_cache)
    rows=np.flatnonzero((cache.objects.official|cache.objects.is_test).to_numpy())
    np.savez_compressed(out/'rfm_cache.npz',Id=cache.objects.iloc[rows].Id.to_numpy(str),X=cache.matrix('F0',rows)[0])
    (out/'requirements.txt').write_text('numpy==2.3.5\npandas==2.2.3\nscikit-learn==1.7.2\nscipy>=1.14,<2\ntorch==2.5.1\n')
    (out/'README.txt').write_text('Open solution.ipynb. Put the four official CSVs in data/raw, or change DATA_DIR.\nAll code is contained in the notebook; no repository imports or network are needed.\nrfm_cache.npz is optional and independently checked against raw history.\nsubmission.csv is the already generated final file; rerunning the notebook creates submission_reproduced.csv.\nOnly one final predictor and seed42. No automatic Kaggle upload.\n')
    atomic_json(out/'delivery_manifest.json',{'kind':kind,'model':recipe['model'],'seed':42,'max_seminar':10,
                  'submission_sha256':digest(out/'submission.csv'),'sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='delivery_manifest.json'}})
    archive=args.output_dir/f'course10_solution_{kind}.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):z.write(path,path.name)
    print(f'Delivery ready: {archive}; notebook: {out/"solution.ipynb"}',flush=True)
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',choices=['selected','fallback','both'],default='both')
    p.add_argument('--run-dir',type=Path,default=Path('outputs/course10-fast-tuning'))
    p.add_argument('--fallback-dir',type=Path,default=Path('outputs/course10-only/refit/mlp'))
    p.add_argument('--base-cache',type=Path,default=Path('outputs/v100-focused/cache'))
    p.add_argument('--data-dir',type=Path,default=Path('data/raw'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/course10-delivery'))
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    kinds=['fallback','selected'] if args.candidate=='both' else [args.candidate]
    for kind in kinds:
        expected=args.fallback_dir/'result.json' if kind=='fallback' else args.run_dir/'selected_refit/result.json'
        if not expected.exists() and args.candidate=='both':
            print(f'Not ready, skipped {kind}',flush=True);continue
        package(args,kind)


if __name__=='__main__':main()
