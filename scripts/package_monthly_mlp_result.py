"""Package an audited CUDA MLP result with standalone inference and training source."""
import argparse,json,runpy,shutil,zipfile
from pathlib import Path
from spend_baselines.suite_data import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,default=Path('analysis/monthly-mlp-v100-results-20261116'))
    p.add_argument('--out',type=Path,default=Path('outputs/monthly-mlp-v100-delivery'))
    args=p.parse_args();proof=json.loads((args.audit/'verification.json').read_text());run=args.audit/'archive'
    assert proof['complete'] and proof['final_seed']==42 and proof['no_forecast_averaging'] and proof['test_rows']==3188
    r=json.loads((run/'refit/result.json').read_text());assert r['upload_ready'] and not r['smoke'] and r['one_predictor']
    csv=run/'refit/submission_monthly_mlp_seed42.csv';assert digest(csv)==proof['CSV_SHA256']
    for n,s in r['sha256'].items():assert digest(run/'refit'/n)==s
    args.out.mkdir(parents=True,exist_ok=True)
    # Reuse fully audited, standalone raw-feature source, replacing only the learner.
    nb=json.loads((ROOT/'outputs/monthly-history-ridge-delivery/solution.ipynb').read_text())
    recipe={'profile':{'name':r['profile'],'model':'mlp','base':'F0','season':True,'history':'full'},
            'settings':r['settings'],'seed':42,'dataset':'D0','task':'direct','max_seminar':10,'feature_names':r['feature_names']}
    nb['cells'][0]['source']='''# Обычная MLP на месячной истории покупок

Один предиктор, seed42,83признака из прошлых транзакций, MSE(log1p target), один выход.
Данные/признаки и временной purge совпадают с экспериментом Ridge. Прогнозы seed не усредняются.
Материалы курса: MLP/ReLU/Adam — семинар3, регрессия/MSE — семинар8.
Код всех признаков и обучения приведён ниже; репозиторий и интернет для запуска не нужны.

Сначала выполняется inference из сохранённых весов. Две последние ячейки позволяют отдельно повторить
временную проверку и полный рефит; они выключены по умолчанию. CPU/CUDA могут давать разные траектории
повторного обучения. Исходный CSV сохранён без изменений; CPU-inference проверяется в log-пространстве.
Kaggle upload отсутствует. При подготовке кода использовался Codex.
'''
    nb['cells'][1]['source']='RECIPE='+repr(recipe)+'\nBASE_FEATURES='+repr(r['feature_names'][:15])+'\n'
    helpers=runpy.run_path(str(ROOT/'scripts/package_course10_solution.py'),run_name='direct_mlp_source')
    nb['cells'][5]['source']=helpers['model_source']()
    nb['cells'][6]['source']='''Проверено на CUDA: seed42 pooledSep–Oct2.632327, matchedRidge2.629809.
Разница+0.002518,95%customer-cluster CI[−0.006028,+0.011530]. Выигрыш MLP не подтверждён.
Seeds123/2026 — диагностика устойчивости; итоговый seed42 выбран заранее. Даты ранее просматривались.
Измеренный пользовательский лучший результат Ridge:Private2.71632/Public2.68000; NovemberMLPещёнеизмерен.
'''
    prefix=nb['cells'][7]['source'].split('\ndef predict(cutoff,vi):')[0]
    nb['cells'][7]['source']=prefix+'\n'
    inference='''
import torch
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
model=DirectMLP.load('model.pt',device=DEVICE,threads=2)
vi=np.flatnonzero(objects.is_test.to_numpy());z=model.predict(X[vi])
with np.load('inputs_predictions.npz',allow_pickle=False) as saved:
    ix=pd.Index(saved['Id']).get_indexer(objects.iloc[vi].Id.to_numpy(str));assert (ix>=0).all()
    np.testing.assert_array_equal(X[vi],saved['X'][ix])
    np.testing.assert_allclose(z,saved['z'][ix],rtol=1e-5,atol=5e-5)
    print('Maximum log prediction difference:',float(np.max(np.abs(z-saved['z'][ix]))))
def write_prediction(z,vi,name):
    a=pd.DataFrame({'Id':objects.iloc[vi].Id.to_numpy(str),'spend_30d':np.maximum(np.expm1(z.astype(float)),0)})
    sample=pd.read_csv(DATA_DIR/'sample_submission.csv',dtype={'Id':str});a=a.set_index('Id').loc[sample.Id].reset_index()
    assert len(a)==3188 and a.Id.is_unique and np.isfinite(a.spend_30d).all() and a.spend_30d.ge(0).all()
    a.to_csv(name,index=False)
write_prediction(z,vi,'submission_from_checkpoint.csv')
print('Created inference CSV; supplied original CUDA CSV remains unchanged.')
'''
    training='''
def fit_predict(cutoff,vi):
    cutoff=pd.Timestamp(cutoff)
    tr=np.flatnonzero((objects.official&objects.target_end.le(cutoff)).to_numpy())
    inner=objects.loc[objects.official&objects.target_end.le(cutoff),'as_of_date'].max()
    ti=np.flatnonzero((objects.official&objects.target_end.le(inner)).to_numpy())
    iv=np.flatnonzero((objects.official&objects.as_of_date.eq(inner)).to_numpy())
    assert objects.iloc[tr].target_end.max()<=cutoff and objects.iloc[ti].target_end.max()<=inner
    settings=RECIPE['settings']
    selector=DirectMLP('mlp','direct',settings,42,DEVICE,2).fit(X[ti],objects.iloc[ti].spend_30d.to_numpy(float),np.ones(len(ti)),validation=(X[iv],objects.iloc[iv].spend_30d.to_numpy(float)))
    epochs=selector.best_epoch;del selector
    model=DirectMLP('mlp','direct',settings,42,DEVICE,2).fit(X[tr],objects.iloc[tr].spend_30d.to_numpy(float),np.ones(len(tr)),epochs=epochs)
    assert model.network.output.out_features==1
    return model.predict(X[vi])

RUN_TEMPORAL_VALIDATION=False
if RUN_TEMPORAL_VALIDATION:
    scores=[]
    for date in ['2018-09-01','2018-10-01']:
        vi=np.flatnonzero((objects.official&objects.as_of_date.eq(date)).to_numpy())
        z=fit_predict(date,vi);err=(z-np.log1p(objects.iloc[vi].spend_30d.to_numpy(float)))**2
        scores.append({'date':date,'n':len(vi),'RMSLE':float(np.sqrt(err.mean()))})
    print(pd.DataFrame(scores).to_string(index=False))
RUN_FINAL_REFIT=False
if RUN_FINAL_REFIT:
    vi=np.flatnonzero(objects.is_test.to_numpy());z=fit_predict(objects.iloc[vi].as_of_date.min(),vi)
    write_prediction(z,vi,'submission_from_new_refit.csv')
'''
    for name,text in [('checkpoint-inference',inference),('optional-retraining',training)]:
        nb['cells'].append({'cell_type':'code','metadata':{},'id':name,'source':text.strip()+'\n','execution_count':None,'outputs':[]})
    for c in nb['cells']:
        if c['cell_type']=='code':compile(c['source'],'monthly-mlp-notebook','exec')
    (args.out/'solution.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False))
    shutil.copyfile(csv,args.out/'submission_monthly_mlp_seed42.csv')
    for n in ['model.pt','inputs_predictions.npz','result.json']:shutil.copyfile(run/'refit'/n,args.out/n)
    for n in ['model_manifest.json','selection.json','confirmation_scores.csv']:shutil.copyfile(run/n,args.out/n)
    shutil.copyfile(args.audit/'verification.json',args.out/'verification.json')
    shutil.copyfile(ROOT/'outputs/monthly-history-ridge-delivery/feature_cache.npz',args.out/'feature_cache.npz')
    (args.out/'requirements.txt').write_text('numpy==2.3.5\npandas==2.2.3\nscikit-learn==1.7.2\nscipy>=1.14,<2\ntorch==2.5.1\n')
    (args.out/'README.txt').write_text('Put four official CSVs in data/raw; open solution.ipynb. Source is standalone. Default runs portable checkpoint inference. Optional switches rerun validation/refit. Supplied original CUDA CSV is preserved. No automatic upload.\n')
    receipt={'original_CSV_SHA256':proof['CSV_SHA256'],'seed':42,'one_predictor':True,'Kaggle_uploaded':False,
        'sha256':{f.name:digest(f) for f in args.out.iterdir() if f.is_file() and f.suffix!='.zip' and f.name!='delivery_manifest.json'}}
    (args.out/'delivery_manifest.json').write_text(json.dumps(receipt,indent=2))
    with zipfile.ZipFile(args.out/'monthly_mlp_solution.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(args.out.iterdir()):
            if f.is_file() and f.suffix!='.zip':z.write(f,f.name)
    print('READY',args.out,flush=True)


if __name__=='__main__':main()
