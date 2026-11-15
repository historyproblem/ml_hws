"""Parameter scope, holdout isolation, metric-only replication and export."""

import copy
import hashlib
import json
import runpy
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

h=runpy.run_path('scripts/tune_course10_fast.py',run_name='course10_fast_helpers')


class FastTuningTests(unittest.TestCase):
    def test_manual_grid_only_contains_course10_methods(self):
        c=h['read_config']('configs/course10_fast_tuning.json')
        self.assertEqual(len(c['profiles']),25)
        self.assertEqual({p['model'] for p in c['profiles']},{'mlp','ridge','forest'})
        self.assertEqual(c['final_seed'],42)
        bad=copy.deepcopy(c);bad['profiles'][0]['overrides']['embedding_dim']=32
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):h['read_config'](p)

    def test_selection_does_not_use_confirmation_rows_or_average_forecasts(self):
        a=pd.DataFrame({'date':['early','early','late'],'seed':[42,123,42],
                        'z':[0.,2.,1000.],'target_z':[1.,1.,1.]})
        self.assertEqual(h['score'](a,['early']),1.)
        a.loc[2,'target_z']=-10000.
        self.assertEqual(h['score'](a,['early']),1.)
        # Averaging predictions0 and2 would give a perfect1; we score two separate models.

    def test_time_limit_does_not_start_another_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=h['Worker'].__new__(h['Worker'])
            w.args=SimpleNamespace(out_dir=Path(tmp),smoke=False)
            w.profiles={'ridge':{'model':'ridge','overrides':{'alpha':10.}}}
            w.deadline=-1;w.cfg={}
            self.assertEqual(w.execute(('ridge','2018-08-01',42)),{'time_limit':True})
            self.assertFalse((Path(tmp)/'jobs').exists())

    def test_export_omits_model_weights_and_checks_forecast_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';d=root/'selected_refit';d.mkdir(parents=True)
            (d/'predictions.npz').write_bytes(b'forecasts')
            (d/'model.joblib').write_bytes(b'heavy forest')
            (d/'model.pt').write_bytes(b'weights')
            r={'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}}
            (d/'result.json').write_text(json.dumps(r))
            output=Path(tmp)/'results.tar.gz'
            h['export'](root,output)
            with tarfile.open(output,'r:gz') as t:
                self.assertTrue(all(not n.endswith(('.pt','.joblib')) for n in t.getnames()))
                receipt=json.loads(t.extractfile('export_manifest.json').read())
                for n,s in receipt['sha256'].items():self.assertEqual(hashlib.sha256(t.extractfile(n).read()).hexdigest(),s)


if __name__=='__main__':unittest.main()
