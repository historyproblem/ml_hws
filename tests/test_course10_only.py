"""Strict course whitelist and read-only reuse of permitted individual results."""

import copy
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from spend_baselines.suite_data import digest
from spend_course10.suite import METHOD_SOURCES,Reference,approved_job,compare,read_config


class ScopeTests(unittest.TestCase):
    def setUp(self):self.cfg=read_config('configs/course10_safe.json')

    def test_only_pinned_elementary_recipe_is_accepted(self):
        for key,value in [('max_seminar',11),('task','hurdle_joint'),('dataset','D2'),
                           ('features','F2'),('seed',[42,123,2026]),('models',['mlp','wide'])]:
            c=copy.deepcopy(self.cfg);c[key]=value
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'config.json';p.write_text(json.dumps(c))
                with self.assertRaises(ValueError):read_config(p)

    def test_reference_job_gate_rejects_all_old_special_recipes(self):
        j={'kind':'neural','family':'mlp','task':'direct','dataset':'D0','representation':'F0',
           'seed':42,'profile':'compact_l2','date':'2018-10-01'}
        self.assertTrue(approved_job(j,self.cfg))
        for key,value in [('task','hurdle_joint'),('task','positive'),('family','wide'),
                           ('family','gru'),('kind','bagging'),('seed',123),('test',True),
                           ('mode','customer'),('representation','S1_F2'),('member',0)]:
            bad={**j,key:value}
            self.assertFalse(approved_job(bad,self.cfg),(key,value))

    def test_random_forest_is_explicitly_sem9_not_generic_ensemble_permission(self):
        j={'kind':'tabular','family':'forest','task':'direct','dataset':'D0','representation':'F0','seed':42}
        self.assertTrue(approved_job(j,self.cfg,'forest'))
        self.assertEqual(METHOD_SOURCES['forest'][0]['seminar'],9)
        self.assertTrue(all(a['seminar']<=10 for v in METHOD_SOURCES.values() for a in v))

    def test_legacy_submission_wrapper_is_disabled_without_training(self):
        import subprocess
        for name in ['make_joint_history_submission.sh','make_item32_submission.sh']:
            r=subprocess.run(['bash','scripts/'+name],capture_output=True,text=True)
            self.assertEqual(r.returncode,2)
            self.assertIn('make_course10_submission.sh',r.stderr)


class ReferenceTests(unittest.TestCase):
    def test_compare_reuses_only_the_three_direct_single_models_without_fit(self):
        from test_suite import synthetic_dataset
        cfg=read_config('configs/course10_safe.json')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir()
            synthetic_dataset(data)
            source=Path('src/spend_baselines')
            definition={'input_sha256':{n:digest(data/n) for n in ['transactions.csv','train.csv','test.csv','sample_submission.csv']},
                        'source_sha256':{p.name:digest(p) for p in source.glob('*.py')},
                        'config':json.loads(Path('configs/v100_suite.json').read_text())}
            train=pd.read_csv(data/'train.csv',dtype={'Id':str,'customer_id':str})
            archive=root/'reference.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('run_manifest.json',json.dumps({'definition':definition}))
                for family,kind in [('linear','tabular'),('forest','tabular'),('mlp','neural')]:
                    for date in cfg['validation_dates']:
                        j={'kind':kind,'family':family,'task':'direct','dataset':'D0','representation':'F0','seed':42,'date':date}
                        if family=='mlp':j['profile']='compact_l2'
                        frame=train[train.as_of_date.eq(date)]
                        stream=io.BytesIO()
                        np.savez_compressed(stream,Id=frame.Id.to_numpy(str),customer_id=frame.customer_id.to_numpy(str),
                                            z=np.log1p(frame.spend_30d.to_numpy()),probability=np.array([]))
                        payload=stream.getvalue()
                        import hashlib
                        r={'job':j,'RMSLE':0.,'sha256':{'predictions.npz':hashlib.sha256(payload).hexdigest()}}
                        prefix=f'jobs/{family}-{date}'
                        z.writestr(prefix+'/result.json',json.dumps(r))
                        z.writestr(prefix+'/predictions.npz',payload)
                z.writestr('jobs/forbidden/result.json',json.dumps({'job':{'kind':'neural','family':'mlp',
                    'task':'hurdle_joint','dataset':'D0','representation':'F0','seed':42,'profile':'compact_l2'}}))
            ref=Reference(archive,data,cfg)
            self.assertEqual(len(ref.results),15)
            out=root/'results';out.mkdir()
            with (patch('spend_course10.suite.Neural.fit',side_effect=AssertionError('must not train')),
                  patch('spend_course10.suite.Tabular.fit',side_effect=AssertionError('must not train'))):
                table=compare(ref,out,cfg)
            self.assertEqual(set(table.model),{'ridge','forest','mlp'})
            self.assertTrue(table.RMSLE.eq(0).all())
            self.assertEqual(len(list((out/'validation_predictions').glob('*/*.csv.gz'))),15)


if __name__=='__main__':unittest.main()
