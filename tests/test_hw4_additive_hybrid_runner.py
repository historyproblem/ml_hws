"""Synthetic12-recipe hybrid/control/export with mixed invalid/resume checks."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json

from sklearn.exceptions import ConvergenceWarning

from test_hw4_additive_runner import fixture
import run_hw4_additive_hybrid as runner


def test_hybrid_synthetic_nested_selection_full_export_and_exact_resume():
    with TemporaryDirectory() as temp,redirect_stdout(StringIO()):
        root=Path(temp);args=fixture(root);args.threads=2;args.plan=runner.DEFAULT_PLAN
        previous=root/'previous';previous.mkdir();args.algorithm_control=previous
        frozen=runner.primary.provenance(args,{})
        module=runner.ROOT/'src/hw4_inspections'
        frozen['source_sha256'].update({n:runner.sha(module/n) for n in ['additive_logit.py','classical_features.py','additive_search.py']})
        sig=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest();frozen['signature']=sig
        runner.write_json(previous/'freeze.json',frozen)
        oof=runner.read_csv(args.baseline_cv).drop(columns=['prediction','model'])
        oof['context_3seed']=.5;oof['algorithm_raw']=.5;oof['algorithm_blend75']=.5
        runner.write_csv(previous/'outer-oof.csv',oof)
        runner.write_json(previous/'manifest.json',dict(signature=sig,complete=True,new_q2_q3_metrics=False,outer_selection_of24recipes=False,
            output_sha256={n:runner.sha(previous/n) for n in ['freeze.json','outer-oof.csv']}))
        actual=runner.fit_recipe
        def mixed(*values,**kwargs):
            if values[2]['id']=='hybrid_H2_C0.1':raise ConvergenceWarning('Synthetic mixed error')
            return actual(*values,**kwargs)
        with patch.object(runner,'fit_recipe',side_effect=mixed):first=runner.run(args)
        assert first['complete'] and len(first['files'])==2 and first['inner_candidate_jobs']==36
        assert first['new_q2_q3_metrics'] is False and first['outer_bestof12'] is False
        before=(args.out/'inner-records.csv').read_bytes();args.resume=True
        with patch.object(runner,'fit_recipe',side_effect=AssertionError('Resume fit')):assert runner.run(args)==first
        assert before==(args.out/'inner-records.csv').read_bytes()
        for q in runner.primary.QUARTERS:
            choice=runner.read_json(args.out/'decisions'/f'{q}.json');assert len(choice['inner_complete_sha256'])==12
        for f in first['files']:
            sub=runner.read_csv(args.out/f['csv']);sample=runner.read_csv(args.raw/'sample_submission.csv')
            assert sub.row_id.equals(sample.row_id) and runner.sha(args.out/f['csv'])==f['sha256']
