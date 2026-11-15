"""One learned history vector + ordinary MLP, direct log regression, fixed seed42."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from spend_baselines.data import feature_matrix,load_dataset
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import atomic_json,digest
from spend_embeddings.data import build_bags,fit_vocabulary
from spend_embeddings.direct import ProductMLP
from spend_behavior.data import positive_lines

SETTINGS={'hidden_sizes':[128,64],'weight_decay':.0001,'dropout':0.,'learning_rate':.001,
 'batch_size':1024,'max_epochs':200,'patience':20,'min_delta':.0001,'embedding_dim':32,
 'embedding':{'window_days':180,'half_life_days':90,'max_items':128,'min_customers':3,'weight_decay':.001,'initial_std':.05}}


def run(args):
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    identity={'settings':SETTINGS,'dataset':'D0','features':'F0+learned-history32','seed':42,'task':'direct',
     'one_predictor':True,'input_sha256':{n:digest(args.data/n) for n in ['train.csv','test.csv','transactions.csv','sample_submission.csv']},
     'source_sha256':{n:digest(n) for n in ['scripts/run_history_embedding_direct.py','src/spend_embeddings/direct.py','src/spend_embeddings/data.py']}}
    marker=out/'manifest.json'
    if marker.exists():assert json.loads(marker.read_text())==identity
    else:atomic_json(marker,identity)
    train,test,history,checks=load_dataset(args.data)
    objects=pd.concat([train.assign(is_test=False),test.assign(is_test=True)],ignore_index=True)
    objects=objects.sort_values(['as_of_date','customer_id']).reset_index(drop=True)
    if (out/'bags.npz').exists():
        with np.load(out/'bags.npz',allow_pickle=False) as a:x,ids,weights,names=a['x'],a['ids'],a['weights'],a['names']
    else:
        x=feature_matrix(history.build_features(objects)).astype(np.float32)
        lines,_=positive_lines(args.data);names=np.array(sorted(lines.item_id.unique()),dtype=str)
        ids,weights,_=build_bags(objects,lines,names,SETTINGS['embedding'])
        np.savez_compressed(out/'bags.npz',x=x,ids=ids,weights=weights,names=names)
    atomic_json(out/'data_checks.json',checks)
    dates=['2018-09-01','2018-10-01',str(test.as_of_date.min().date())]
    reports=[]
    for date in dates:
        directory=out/date;directory.mkdir(exist_ok=True)
        if (directory/'result.json').exists():
            r=json.loads((directory/'result.json').read_text())
            for n,s in r['sha256'].items():assert digest(directory/n)==s
            reports.append(r);continue
        cutoff=pd.Timestamp(date);testing=date==dates[-1]
        tr=np.flatnonzero((~objects.is_test&objects.target_end.le(cutoff)).to_numpy())
        vi=np.flatnonzero((objects.is_test if testing else ~objects.is_test&objects.as_of_date.eq(cutoff)).to_numpy())
        inner=objects.iloc[tr].as_of_date.max();ti=np.flatnonzero((~objects.is_test&objects.target_end.le(inner)).to_numpy())
        iv=np.flatnonzero((~objects.is_test&objects.as_of_date.eq(inner)).to_numpy())
        assert objects.iloc[ti].target_end.max()<=inner and objects.iloc[tr].target_end.max()<=cutoff
        imap,_=fit_vocabulary(ids[ti],objects.iloc[ti].customer_id.to_numpy(),len(names),3)
        print(date,'inner',inner.date(),'selecting one direct MLP',flush=True);start=time.monotonic()
        model=ProductMLP(SETTINGS,42,'cpu',args.threads)
        model.fit(x[ti],ids[ti],weights[ti],objects.iloc[ti].spend_30d.to_numpy(float),np.ones(len(ti)),imap,
                  validation=(x[iv],ids[iv],weights[iv],objects.iloc[iv].spend_30d.to_numpy(float)))
        epoch=model.best_epoch;pd.DataFrame(model.history).to_csv(directory/'selection.csv',index=False);del model
        mapping,_=fit_vocabulary(ids[tr],objects.iloc[tr].customer_id.to_numpy(),len(names),3)
        model=ProductMLP(SETTINGS,42,'cpu',args.threads).fit(x[tr],ids[tr],weights[tr],objects.iloc[tr].spend_30d.to_numpy(float),np.ones(len(tr)),mapping,epochs=epoch)
        assert model.network.output.out_features==1
        pd.DataFrame(model.history).to_csv(directory/'refit.csv',index=False);model.save(directory/'model.pt',names)
        z=model.predict(x[vi],ids[vi],weights[vi]);frame=objects.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
        frame['z']=z;frame['date']=date
        if not testing:frame['target_z']=np.log1p(objects.iloc[vi].spend_30d.to_numpy(float))
        frame.to_csv(directory/'predictions.csv.gz',index=False)
        if testing:
            sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str})
            write_submission(objects.iloc[vi],z,out/'submission_history_embedding32_mlp.csv',sample_ids=sample.Id)
        r={'date':date,'testing':testing,'epoch':epoch,'inner_date':str(inner),'n_train':len(tr),
           'max_target_end':str(objects.iloc[tr].target_end.max()),'seconds':time.monotonic()-start,
           'vocabulary':int(mapping.max()),'one_output':True,'one_predictor':True,'seed':42,'task':'direct',
           'sha256':{q.name:digest(q) for q in directory.iterdir() if q.name!='result.json'}}
        if not testing:r['RMSLE']=float(np.sqrt(np.mean((frame.z-frame.target_z)**2)))
        atomic_json(directory/'result.json',r);reports.append(r);print(json.dumps(r),flush=True)
    a=pd.concat([pd.read_csv(out/d/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}) for d in dates[:2]],ignore_index=True)
    atomic_json(out/'summary.json',{'RMSLE_SepOct':float(np.sqrt(np.mean((a.z-a.target_z)**2))),
       'csv_sha256':digest(out/'submission_history_embedding32_mlp.csv'),'reports':reports,'Kaggle_uploaded':False})
    print('READY',out/'submission_history_embedding32_mlp.csv',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('data/raw'))
    p.add_argument('--out',type=Path,default=Path('outputs/history-embedding-direct'));p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    with threadpool_limits(limits=args.threads):run(args)


if __name__=='__main__':main()
