"""Synthetic nested8 state recipes, unchanged baseline, full and first replay."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json

from sklearn.exceptions import ConvergenceWarning
from test_hw4_additive_runner import fixture
import run_hw4_gam_state as runner


def test_state_nested8_gated_full_mixedinvalid_and_first_resume_exact():
    with TemporaryDirectory() as temp,redirect_stdout(StringIO()):
        root=Path(temp);args=fixture(root);args.threads=2;args.plan=runner.DEFAULT_PLAN;benchmark=root/'benchmark';benchmark.mkdir();args.benchmark=benchmark
        f=runner.primary.provenance(args,{})
        module=runner.ROOT/'src/hw4_inspections';f['source_sha256'].update({n:runner.sha(module/n) for n in ['additive_logit.py','classical_features.py','additive_search.py']});sig=hashlib.sha256(json.dumps(f,sort_keys=True).encode()).hexdigest();f['signature']=sig;runner.write_json(benchmark/'freeze.json',f)
        oof=runner.read_csv(args.baseline_cv).drop(columns=['prediction','model']);oof['context_3seed']=.5;oof['fixed_teacher']=.5;oof['previous_algorithm']=.5;runner.write_csv(benchmark/'outer-oof.csv',oof);runner.write_json(benchmark/'manifest.json',dict(signature=sig,complete=True,new_q2_q3_metrics=False,output_sha256={n:runner.sha(benchmark/n) for n in ['freeze.json','outer-oof.csv']}))
        actual=runner.fit_recipe
        def mixed(*values):
            if values[2]['id']=='state_load_outcome_C0.1':raise ConvergenceWarning('Synthetic invalid')
            return actual(*values)
        with patch.object(runner,'fit_recipe',side_effect=mixed):first=runner.run(args)
        assert first['complete'] and first['inner_candidate_jobs']==24 and len(first['files'])==2
        assert first['new_q2_q3_metrics'] is False and first['outer_bestof8'] is False
        args.resume=True
        with patch.object(runner,'fit_recipe',side_effect=AssertionError('State resume fit')):assert runner.run(args)==first
        for q in runner.primary.QUARTERS:
            d=runner.read_json(args.out/'decisions'/f'{q}.json');assert len(d['inner_complete_sha256'])==8
