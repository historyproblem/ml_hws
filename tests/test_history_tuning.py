"""Holdout isolation, causal cache audit, frozen decisions and safe exports."""

import copy
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from spend_baselines.suite_data import atomic_json, digest, row_weights
from spend_behavior.models import causal_histories
from spend_tuning.data import causal_history_check
from spend_tuning.report import export
from spend_tuning.suite import (
    BudgetReached, Runner, budget, choose_shortlist, choose_winner, freeze, read_config,
    temporal_split,
)


def forecasts(cfg,error):
    return pd.DataFrame([{'Id':f'{date}:{cid}','customer_id':cid,'date':date,
                          'target_z':1.,'z':1.+error}
                         for date in cfg['tuning_dates']+cfg['confirmation_dates']
                         for cid in ['a','b']])


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read_config('configs/history_tuning.json')

    def test_budget_and_fixed_architecture(self):
        limits=budget(self.cfg)
        self.assertEqual(limits['max_seed_evaluations'],39)
        self.assertEqual(limits['max_neural_fits'],78)
        self.assertFalse(limits['control_retrained'])
        self.assertEqual(self.cfg['base_settings']['hidden_sizes'],[128,64])

    def test_shortlist_and_winner_ignore_heldout_predictions(self):
        names=[p['name'] for p in self.cfg['profiles'] if p['name']!='baseline']
        frames={name:forecasts(self.cfg,.1+i*.1) for i,name in enumerate(names)}
        before=choose_shortlist(frames,self.cfg)
        frames[names[-1]].loc[frames[names[-1]].date.isin(self.cfg['confirmation_dates']),'z']=1.
        self.assertEqual(before,choose_shortlist(frames,self.cfg))
        selected={name:frames[name] for name in before['profiles']}
        selected['baseline']=forecasts(self.cfg,.5)
        winner=choose_winner(selected,self.cfg)
        for f in selected.values():
            f.loc[f.date.isin(self.cfg['confirmation_dates']),'z']=1e10
            f.loc[f.date.isin(self.cfg['confirmation_dates']),'target_z']=-1e10
        self.assertEqual(winner,choose_winner(selected,self.cfg))
        self.assertEqual(winner['profile'],names[0])
        self.assertFalse(winner['confirmation_used_for_selection'])

    def test_baseline_kept_when_tuning_gain_is_too_small(self):
        a=forecasts(self.cfg,.2)
        b=forecasts(self.cfg,.1995)
        self.assertEqual(choose_winner({'baseline':a,'lr_half':b},self.cfg)['profile'],'baseline')

    def test_incomplete_screen_cannot_select_winner(self):
        frames={p['name']:forecasts(self.cfg,.2) for p in self.cfg['profiles'][1:]}
        name=self.cfg['profiles'][1]['name']
        frames[name]=frames[name][frames[name].date.ne(self.cfg['tuning_dates'][0])]
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            choose_shortlist(frames,self.cfg)

    def test_overlapping_tuning_and_confirmation_windows_rejected(self):
        c=copy.deepcopy(self.cfg)
        c['tuning_dates']=['2018-08-08']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'
            path.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'available before'):
                read_config(path)

    def test_frozen_winner_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'winner.json'
            freeze(p,{'profile':'a'})
            freeze(p,{'profile':'a'})
            with self.assertRaisesRegex(ValueError,'Frozen'):
                freeze(p,{'profile':'b'})

    def test_confirmation_access_starts_after_winner_is_frozen(self):
        self._confirmation_order(baseline_error=.8,expect_new=True)

    def test_baseline_winner_skips_all_new_confirmation_fits(self):
        self._confirmation_order(baseline_error=.001,expect_new=False)

    def _confirmation_order(self,baseline_error,expect_new):
        cfg=self.cfg
        names=[p['name'] for p in cfg['profiles'][1:]]
        frames={name:forecasts(cfg,.1+i*.1) for i,name in enumerate(names)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            decision=choose_shortlist(frames,cfg)
            atomic_json(root/'shortlist.json',{**decision,'job_result_sha256':{}})
            events=[]
            class FakeRunner(Runner):
                def seed_frame(self,name,date,seed):
                    return frames[name][frames[name].date.eq(date)].copy()
                def execute(self,name,date,seed,testing=False):
                    if date in cfg['confirmation_dates']:
                        assert (root/'winner.json').exists()
                    events.append((name,date,seed))
                def control(self,date):
                    if date in cfg['confirmation_dates']:
                        assert (root/'winner.json').exists()
                    a=forecasts(cfg,baseline_error)
                    return a[a.date.eq(date)].copy()
                def mean3(self,name,date):
                    return self.seed_frame(name,date,42)
            args=SimpleNamespace(out_dir=root,smoke=False,resume=True)
            result=FakeRunner(args,cfg,None,None).confirm()
            late=[e for e in events if e[1] in cfg['confirmation_dates']]
            self.assertEqual(result['profile']!='baseline',expect_new)
            self.assertEqual(len(late),6 if expect_new else 0)


class CausalCacheTests(unittest.TestCase):
    def test_vectorised_history_matches_original_and_ignores_unavailable_labels(self):
        dates=pd.to_datetime(['2018-01-01','2018-01-31','2018-02-01','2018-03-01'])
        objects=pd.DataFrame([{'customer_id':cid,'as_of_date':date,'target_end':date+pd.Timedelta(days=30),
                               'spend_30d':float(i*10+20*(cid=='b')),'official':i!=1}
                              for cid in ['a','b'] for i,date in enumerate(dates)])
        expected,_=causal_histories(objects,np.full(len(objects),np.nan),[2],prior=5,half_life=90)
        actual=causal_history_check(objects,5,90)
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=0)
        poisoned=objects.copy()
        poisoned.loc[poisoned.as_of_date.eq('2018-03-01'),'spend_30d']=1e10
        future=causal_history_check(poisoned,5,90)
        np.testing.assert_array_equal(actual,future)
        # target_end == as_of_date is admitted, earlier than that it is absent.
        self.assertEqual(actual[0,0],0)
        self.assertAlmostEqual(actual[1,0],np.log1p(1),places=6)

    def test_nested_temporal_splits_purge_before_weighting(self):
        dates=pd.to_datetime(['2018-06-01','2018-07-01','2018-08-01','2018-08-08','2018-08-22','2018-09-01'])
        obj=pd.DataFrame({'Id':[str(i) for i in range(len(dates))],'customer_id':'a',
                          'as_of_date':dates,'target_end':dates+pd.Timedelta(days=30),
                          'month':dates.strftime('%Y-%m'),'official':dates.day==1})
        data=SimpleNamespace(objects=obj,active=np.ones(len(obj),bool),
                             valid=lambda date:np.flatnonzero((obj.as_of_date.eq(date)&obj.official).to_numpy()))
        outer,inner,iv,date=temporal_split(data,'2018-09-01')
        self.assertEqual(date,pd.Timestamp('2018-08-01'))
        self.assertLessEqual(obj.iloc[outer].target_end.max(),pd.Timestamp('2018-09-01'))
        self.assertLessEqual(obj.iloc[inner].target_end.max(),date)
        self.assertEqual(obj.iloc[outer].as_of_date.max(),pd.Timestamp('2018-08-01'))
        self.assertEqual(row_weights(obj,outer,'D2').sum(),3)


class PersistenceTests(unittest.TestCase):
    def test_time_budget_stops_before_starting_another_fit(self):
        cfg=read_config('configs/history_tuning.json')
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(out_dir=Path(tmp),smoke=False,max_minutes=1e-12)
            run=Runner(args,cfg,None,None)
            run.started=0.
            with self.assertRaises(BudgetReached):
                run.execute('lr_half','2018-06-01',42)
            self.assertFalse((Path(tmp)/'jobs').exists())

    def test_resume_checks_parameters_and_model_checksum(self):
        cfg=read_config('configs/history_tuning.json')
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(out_dir=Path(tmp),smoke=False)
            run=Runner(args,cfg,None,None)
            d=run.directory('lr_half','2018-06-01',42)
            d.mkdir(parents=True)
            (d/'model.pt').write_bytes(b'checkpoint')
            state={'job':{'profile':'lr_half','date':'2018-06-01','seed':42,'testing':False,
                          'settings':run.settings['lr_half']},'sha256':{'model.pt':digest(d/'model.pt')}}
            atomic_json(d/'result.json',state)
            self.assertIsNotNone(run.completed('lr_half','2018-06-01',42))
            (d/'model.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt'):
                run.completed('lr_half','2018-06-01',42)

    def test_tar_export_omits_weights_and_excludes_unfinished_or_corrupt_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'results'
            root.mkdir()
            (root/'manifest.json').write_text('{}')
            for seed in [42,123,2026]:
                d=root/'jobs/lr_half/2018-06-01'/str(seed)
                d.mkdir(parents=True)
                (d/'predictions.csv.gz').write_bytes(b'row forecasts')
                (d/'model.pt').write_bytes(b'weights')
                if seed!=123:
                    atomic_json(d/'result.json',{'sha256':{'predictions.csv.gz':digest(d/'predictions.csv.gz'),
                                                         'model.pt':digest(d/'model.pt')}})
                if seed==2026:
                    (d/'predictions.csv.gz').write_bytes(b'bad forecasts')
            out=Path(tmp)/'results.tar.gz'
            result=export(root,out)
            self.assertEqual(result['completed_jobs'],1)
            self.assertEqual(result['unfinished_jobs'],1)
            self.assertEqual(result['invalid_jobs'],1)
            with tarfile.open(out,'r:gz') as t:
                self.assertTrue(all(not m.name.endswith('.pt') for m in t.getmembers()))
                receipt=json.loads(t.extractfile('export_manifest.json').read())
                for name,sha in receipt['sha256'].items():
                    self.assertEqual(hashlib.sha256(t.extractfile(name).read()).hexdigest(),sha)


if __name__=='__main__':
    unittest.main()
