"""Build the 83 causal monthly features for an existing D2 base cache."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from spend_baselines.data import load_dataset
from spend_baselines.suite_data import CachedData,atomic_json,digest
from spend_course10.history_features import monthly_history_features
from course10_forest_followup import seasonal_features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'))
    parser.add_argument('--out',type=Path,default=Path('outputs/monthly-snapshot-features'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    state_path=args.out/'features.ready.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        assert digest(args.out/'features.npy')==state['sha256']
        assert digest(args.cache/'ready.json')==state['identity']['cache_sha256']
        for name,sha in state['identity']['input_sha256'].items():assert digest(args.data/name)==sha
        print('Verified existing features',args.out,flush=True);return
    cache=CachedData(args.cache);objects=cache.objects
    state=json.loads((args.cache/'ready.json').read_text())
    for name in ['objects.csv.gz','features.npy']:assert digest(args.cache/name)==state['sha256'][name]
    _,_,history,checks=load_dataset(args.data)
    known=objects[~objects.is_test]
    assert known.target_end.max()<=objects.loc[objects.is_test,'as_of_date'].min()
    reconstructed=history.reconstruct_targets(known)
    checks['expanded_targets_checked']=len(known)
    checks['expanded_target_max_error']=float(np.max(np.abs(reconstructed-known.spend_30d.to_numpy(float))))
    assert checks['expanded_target_max_error']<.011
    start=min(v[0][0] for v in history.customers.values())
    start=pd.Timestamp('1970-01-01')+pd.Timedelta(days=int(start))
    x=np.c_[cache.matrix('F0',np.arange(len(objects)))[0],seasonal_features(objects,history,start),
            monthly_history_features(objects,history,start)]
    assert x.shape==(len(objects),83) and not np.isinf(x).any()
    np.save(args.out/'features.npy',x)
    source=['scripts/prepare_monthly_snapshot_features.py','scripts/course10_forest_followup.py',
            'src/spend_course10/history_features.py','src/spend_baselines/data.py']
    identity={'cache_sha256':digest(args.cache/'ready.json'),
              'input_sha256':{n:digest(args.data/n) for n in ['train.csv','test.csv','transactions.csv','sample_submission.csv']},
              'source_sha256':{n:digest(n) for n in source}}
    atomic_json(state_path,{'identity':identity,'sha256':digest(args.out/'features.npy'),'checks':checks})
    print('READY',x.shape,args.out,'target max error',checks['expanded_target_max_error'],flush=True)


if __name__=='__main__':main()
