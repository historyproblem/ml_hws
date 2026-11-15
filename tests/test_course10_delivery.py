"""Standalone notebook contains direct regressors and reproduces their math."""

import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from spend_baselines.suite_models import Neural

h=runpy.run_path('scripts/package_course10_solution.py',run_name='delivery_helpers')


class DeliveryTests(unittest.TestCase):
    def test_specialised_notebook_mlp_matches_original_direct_trainer(self):
        scope={};source=h['model_source']();exec(compile(source,'notebook-model','exec'),scope)
        settings={'hidden_sizes':[8,4],'learning_rate':.001,'weight_decay':.0001,'dropout':.1,
                  'batch_size':8,'max_epochs':3,'patience':2,'min_delta':.0001}
        rng=np.random.default_rng(2);x=rng.normal(size=(24,3)).astype(np.float32);x[0,0]=np.nan
        y=np.arange(24,dtype=float);w=np.ones(24)
        a=Neural('mlp','direct',settings,42,'cpu',2).fit(x,y,w,validation=(x,y))
        b=scope['DirectMLP']('mlp','direct',settings,42,'cpu',2).fit(x,y,w,validation=(x,y))
        self.assertEqual(a.best_epoch,b.best_epoch)
        np.testing.assert_array_equal(a.predict(x),b.predict(x))
        self.assertTrue(all(s not in source for s in ['hurdle','Embedding','GRU','predict_probability']))

    def test_notebook_code_is_standalone_and_compiles(self):
        recipe={'model':'ridge','profile':'ridge_base','settings':{'alpha':10.},'seed':42}
        notebook=h['notebook'](recipe,'')
        self.assertEqual(notebook['nbformat'],4)
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                compile(cell['source'],'notebook-cell','exec')
                self.assertNotIn('from spend_',cell['source'])
        self.assertEqual(len({c['id'] for c in notebook['cells']}),len(notebook['cells']))

    def test_smoke_artifacts_cannot_be_packaged_as_final_solution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'selected_refit';d.mkdir()
            (d/'result.json').write_text(json.dumps({'smoke':True,'upload_ready':False,'seed':42,'predictor_count':1}))
            args=SimpleNamespace(run_dir=root)
            with self.assertRaisesRegex(ValueError,'completed full'):
                h['package'](args,'selected')

    def test_standalone_notebook_runs_without_repository_or_old_outputs(self):
        from test_suite import synthetic_dataset
        recipe={'model':'ridge','profile':'ridge_base','settings':{'alpha':10.},'seed':42}
        notebook=h['notebook'](recipe,'')
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data/raw';data.mkdir(parents=True)
            synthetic_dataset(data)
            scope={'display':lambda value:None}
            try:
                os.chdir(root)
                for cell in notebook['cells']:
                    if cell['cell_type']=='code':exec(compile(cell['source'],'standalone-notebook','exec'),scope)
                self.assertTrue((root/'submission_reproduced.csv').exists())
            finally:os.chdir(previous)


if __name__=='__main__':unittest.main()
