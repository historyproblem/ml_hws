"""Readable standalone notebook for the single learned-history MLP."""
import ast,json,runpy,shutil,zipfile
from pathlib import Path
from spend_baselines.suite_data import digest

ROOT=Path(__file__).resolve().parents[1]


def source(path,names):
    return '\n\n'.join(ast.unparse(n) for n in ast.parse(path.read_text()).body
                        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)+'\n'


RUN='''
DATA_DIR=Path('data/raw')
train,test,history,checks=load_dataset(DATA_DIR)
objects=pd.concat([train.assign(is_test=False),test.assign(is_test=True)],ignore_index=True).sort_values(['as_of_date','customer_id']).reset_index(drop=True)
x=feature_matrix(history.build_features(objects)).astype(np.float32)
lines,_=positive_lines(DATA_DIR);names=np.array(sorted(lines.item_id.unique()),dtype=str)
ids,weights,diagnostics=build_bags(objects,lines,names,SETTINGS['embedding'])
scores=[]
for date in ['2018-09-01','2018-10-01','2018-11-01']:
    cutoff=pd.Timestamp(date);testing=date=='2018-11-01'
    tr=np.flatnonzero((~objects.is_test&objects.target_end.le(cutoff)).to_numpy())
    vi=np.flatnonzero((objects.is_test if testing else ~objects.is_test&objects.as_of_date.eq(cutoff)).to_numpy())
    inner=objects.iloc[tr].as_of_date.max()
    ti=np.flatnonzero((~objects.is_test&objects.target_end.le(inner)).to_numpy())
    iv=np.flatnonzero((~objects.is_test&objects.as_of_date.eq(inner)).to_numpy())
    assert objects.iloc[ti].target_end.max()<=inner and objects.iloc[tr].target_end.max()<=cutoff
    imap,_=fit_vocabulary(ids[ti],objects.iloc[ti].customer_id.to_numpy(),len(names),3)
    selector=ProductMLP(SETTINGS,42,'cpu',2).fit(x[ti],ids[ti],weights[ti],objects.iloc[ti].spend_30d.to_numpy(float),np.ones(len(ti)),imap,
             validation=(x[iv],ids[iv],weights[iv],objects.iloc[iv].spend_30d.to_numpy(float)))
    epochs=selector.best_epoch;del selector
    mapping,_=fit_vocabulary(ids[tr],objects.iloc[tr].customer_id.to_numpy(),len(names),3)
    model=ProductMLP(SETTINGS,42,'cpu',2).fit(x[tr],ids[tr],weights[tr],objects.iloc[tr].spend_30d.to_numpy(float),np.ones(len(tr)),mapping,epochs=epochs)
    assert model.network.output.out_features==1
    z=model.predict(x[vi],ids[vi],weights[vi])
    if not testing:
        error=(z-np.log1p(objects.iloc[vi].spend_30d.to_numpy(float)))**2
        scores.append({'date':date,'n':len(vi),'RMSLE':float(np.sqrt(error.mean())),'squared_log_error':float(error.sum())})
    else:
        a=pd.DataFrame({'Id':objects.iloc[vi].Id.to_numpy(str),'spend_30d':np.maximum(np.expm1(z),0)})
        sample=pd.read_csv(DATA_DIR/'sample_submission.csv',dtype={'Id':str});a=a.set_index('Id').loc[sample.Id].reset_index()
        assert len(a)==3188 and a.Id.is_unique and np.isfinite(a.spend_30d).all() and a.spend_30d.ge(0).all()
        a.to_csv('submission_reproduced.csv',index=False)
print(checks)
print(pd.DataFrame(scores).to_string(index=False))
print('Pooled RMSLE',np.sqrt(sum(r['squared_log_error'] for r in scores)/sum(r['n'] for r in scores)))
print('Created submission_reproduced.csv; no automatic Kaggle upload.')
'''


def main():
    run=ROOT/'outputs/history-embedding-direct';out=ROOT/'outputs/history-embedding-direct-delivery';out.mkdir(exist_ok=True)
    proof=json.loads((run/'verification.json').read_text());assert proof['rows']==3188 and proof['all3checkpoint_forwards_verified']
    assert digest(run/'submission_history_embedding32_mlp.csv')==proof['CSV_SHA256']
    settings=runpy.run_path(str(ROOT/'scripts/run_history_embedding_direct.py'),run_name='embedding_delivery')['SETTINGS']
    cells=[]
    def add(kind,text):
        cell={'cell_type':kind,'metadata':{},'id':f'cell-{len(cells):03d}','source':text.strip()+'\n'}
        if kind=='code':cell.update(execution_count=None,outputs=[]);compile(cell['source'],'embedding-notebook','exec')
        cells.append(cell)
    add('markdown','''# Обучаемый вектор истории покупок + MLP

Для каждого объекта используются покупки строго за[t−180,t). Один товар учитывается один раз в покупательский день,
вес2**(−age/90),top128по сумме весов. Веса нормированы. Словарь fit только на обучающей части, минимум3клиента;
неизвестные товары имеют нулевой вектор. История превращается в h=Σw_jE_j размерности32.
Это обучаемое линейное преобразование взвешенного multi-hot вектора товаров. МатрицаE и MLP обучаются совместно.
В MLP128/64 подаются15RFM иh. Один линейный выход, MSE(log1p target), seed42; прогнозexpm1 и нижняя граница0.

Базовые блоки: линейные слои/ReLU/Adam изсеминара3, идея обучаемых представлений из7, регрессияMSE из8.
Точного готового рецепта этой агрегации в семинарах не проверено; это исследовательская композиция знакомых блоков.
Она подготовлена по явной просьбе попробовать обучаемый эмбеддинг истории. Архивная Joint-модель не используется.
Все50852строкиD0официальные, без смешивания моделей/seed, внешних данных, трансформеров, бустинга или AutoML.
Внутренние словари и модели строятся заново с temporalpurge; target_end<=cutoff.
Ранее просмотренныеSep/Octне являются новымholdout. МеткиNovemberнеизвестны. Код подготовлен с помощьюCodex.''')
    add('code','SETTINGS='+repr(settings))
    add('code',(ROOT/'src/spend_baselines/data.py').read_text())
    add('code',source(ROOT/'src/spend_behavior/data.py',{'positive_lines'}))
    add('code',source(ROOT/'src/spend_embeddings/data.py',{'build_bags','fit_vocabulary'}))
    helpers='''
import os,random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
def neural_loss(raw,y,w,task):
    import torch
    assert task=='direct' and raw.shape[1]==1
    z=torch.log1p(y)
    return (((raw[:,0]-z)**2)*w).sum()/w.sum().clamp_min(1e-12)
def output_prediction(raw,task):
    assert task=='direct' and raw.shape[1]==1
    return raw[:,0].clamp_min(0)
'''
    add('code',helpers+'\n'+source(ROOT/'src/spend_baselines/suite_models.py',{'Preprocessor'})+'\n'+source(ROOT/'src/spend_embeddings/direct.py',{'make_network','ProductMLP'}))
    add('markdown','Зафиксировано:September2.631559,October2.676168,pooled2.654549. СезонныйForest2.645327; преимущество этого нейронного кандидата не показано. Проверка и поздняя отправка являются исследовательскими.')
    add('code',RUN)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'cells':cells}
    (out/'solution.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False));shutil.copyfile(run/'submission_history_embedding32_mlp.csv',out/'submission_history_embedding32_mlp.csv')
    for n in ['manifest.json','verification.json','data_checks.json','serialization_migration.json']:
        shutil.copyfile(run/n,out/n)
    (out/'requirements.txt').write_text('numpy==2.3.5\npandas==2.2.3\nscikit-learn==1.7.2\nscipy>=1.14,<2\ntorch==2.5.1\n')
    (out/'delivery_manifest.json').write_text(json.dumps({'CSV_SHA256':proof['CSV_SHA256'],'sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.suffix!='.zip' and p.name!='delivery_manifest.json'}},indent=2))
    with zipfile.ZipFile(out/'history_embedding_direct_solution.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
    print('Ready:',out,flush=True)


if __name__=='__main__':main()
