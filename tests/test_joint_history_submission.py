"""Prevent wrong-model/smoke submissions and check mean3/log-space conversion."""

import copy
import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

helpers=runpy.run_path('scripts/export_joint_history_submission.py',run_name='submission_helpers')


def prediction(ids,z):
    return {'Id':np.array(ids),'customer_id':np.array(['c'+i for i in ids]),
            'z':np.array(z,dtype=float),'spend_30d':np.expm1(z)}


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.test=pd.DataFrame({'Id':['a','b'],'customer_id':['ca','cb']})
        self.sample=pd.DataFrame({'Id':['b','a'],'spend_30d':[0,0]})

    def test_averages_logs_then_expm1_and_preserves_sample_order(self):
        seeds=[prediction(['a','b'],[1.,2.]),prediction(['b','a'],[4.,3.]),prediction(['a','b'],[2.,3.])]
        mean=prediction(['b','a'],[3.,2.])
        out=helpers['assemble'](self.test,self.sample,seeds,mean)
        self.assertEqual(out.Id.tolist(),['b','a'])
        np.testing.assert_allclose(out.spend_30d,[np.expm1(3.),np.expm1(2.)])

    def test_wrong_customer_or_population_is_rejected(self):
        good=prediction(['a','b'],[1.,2.])
        bad=copy.deepcopy(good);bad['customer_id'][0]='xx'
        with self.assertRaisesRegex(ValueError,'customer'):
            helpers['assemble'](self.test,self.sample,[bad,good,good],good)
        with self.assertRaisesRegex(ValueError,'Ids'):
            helpers['prediction_frame'](prediction(['a','x'],[1.,2.]),self.test)

    def test_wrong_saved_mean_or_nonfinite_values_are_rejected(self):
        good=prediction(['a','b'],[1.,2.])
        with self.assertRaises(AssertionError):
            helpers['assemble'](self.test,self.sample,[good]*3,prediction(['a','b'],[2.,3.]))
        bad=copy.deepcopy(good);bad['z'][0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):
            helpers['assemble'](self.test,self.sample,[bad,good,good],good)

    def test_cpu_smoke_and_lr_double_are_rejected(self):
        manifest={'device':'cuda','smoke':False,'config':{'base_settings':helpers['SETTINGS'],
                  'seeds':[42,123,2026],'profiles':[{'name':'baseline','overrides':{}}]}}
        helpers['require_full_cuda'](manifest)
        for key,value in [('smoke',True),('device','cpu')]:
            bad=copy.deepcopy(manifest);bad[key]=value
            with self.assertRaisesRegex(ValueError,'CUDA'):
                helpers['require_full_cuda'](bad)
        bad=copy.deepcopy(manifest);bad['config']['base_settings']['learning_rate']=.002
        with self.assertRaisesRegex(ValueError,'original'):
            helpers['require_full_cuda'](bad)

    def test_export_checks_refit_provenance_and_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';data=root/'data'
            run.mkdir();data.mkdir()
            manifest={'device':'cuda','smoke':False,'config':{'base_settings':helpers['SETTINGS'],
                      'seeds':[42,123,2026],'profiles':[{'name':'baseline','overrides':{}}]},
                      'source_sha256':{},'reference_manifest_sha256':hashlib.sha256(b'fixture').hexdigest()}
            (run/'manifest.json').write_text(json.dumps(manifest))
            test=self.test.copy();test['as_of_date']='2018-11-01';test['target_end']='2018-12-01'
            test.to_csv(data/'test.csv',index=False)
            self.sample.to_csv(data/'sample_submission.csv',index=False)
            seed_values=[]
            for seed,shift in zip([42,123,2026],[0.,1.,2.]):
                d=run/'refit_jobs/baseline/2018-11-01'/str(seed);d.mkdir(parents=True)
                values=prediction(['a','b'],[1.+shift,2.+shift]);seed_values.append(values)
                np.savez_compressed(d/'predictions.npz',**values)
                pd.DataFrame({'epoch':[1],'train_loss':[1.]}).to_csv(d/'refit.csv',index=False)
                pd.DataFrame({'epoch':[1,2],'inner_metric':[1.,2.]}).to_csv(d/'selection.csv',index=False)
                (d/'model.pt').write_bytes(b'fixture weights, not deserialised')
                result={'job':{'profile':'baseline','date':'2018-11-01','seed':seed,'testing':True,
                               'settings':helpers['SETTINGS']},'n_validation':2,'n_train':30,
                        'max_target_end_train':'2018-10-31','max_target_end_inner':'2018-10-01',
                        'inner_date':'2018-10-01','selected_epoch':1,
                        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}}
                (d/'result.json').write_text(json.dumps(result))
            d=run/'refit/baseline';d.mkdir(parents=True)
            np.savez_compressed(d/'predictions.npz',**prediction(['a','b'],[2.,3.]))
            (d/'result.json').write_text(json.dumps({'profile':'baseline','cutoff':'2018-11-01',
                  'predictions_sha256':hashlib.sha256((d/'predictions.npz').read_bytes()).hexdigest()}))
            fake=lambda *args:SimpleNamespace(manifest_bytes=b'fixture')
            with patch.dict(helpers['export'].__globals__,{'FrozenRun':fake}):
                receipt=helpers['export'](run,root/'reference',data,run/'submission.csv')
            written=pd.read_csv(run/'submission.csv',dtype={'Id':str})
            self.assertEqual(written.Id.tolist(),['b','a'])
            np.testing.assert_allclose(written.spend_30d,[np.expm1(3.),np.expm1(2.)])
            self.assertEqual(receipt['rows'],2)
            self.assertFalse(receipt['kaggle_uploaded'])


if __name__=='__main__':unittest.main()
