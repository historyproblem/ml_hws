"""Synthetic causal first-level OOF, zero/CB API and gated full/replay flow."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json

import joblib
import numpy as np

from test_hw4_additive_runner import fixture
import run_hw4_gam_residual as runner
from hw4_inspections.additive_search import fit_recipe,recipe_by_id,predict_recipe


def test_residual_runner_causal_teacher_oof_original_y_full_and_resume():
    with TemporaryDirectory() as temp,redirect_stdout(StringIO()):
        root=Path(temp);args=fixture(root);args.threads=2;args.plan=runner.DEFAULT_PLAN
        control=root/'previous';control.mkdir();args.algorithm_control=control
        frozen=runner.primary.provenance(args,{})
        module=runner.ROOT/'src/hw4_inspections'
        frozen['source_sha256'].update({n:runner.sha(module/n) for n in ['additive_logit.py','classical_features.py','additive_search.py']})
        sig=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest();frozen['signature']=sig
        runner.write_json(control/'freeze.json',frozen)
        oof=runner.read_csv(args.baseline_cv).drop(columns=['prediction','model']);oof['context_3seed']=.5;oof['algorithm_raw']=.5
        runner.write_csv(control/'outer-oof.csv',oof)
        runner.write_json(control/'manifest.json',dict(signature=sig,complete=True,new_q2_q3_metrics=False,
            output_sha256={n:runner.sha(control/n) for n in ['freeze.json','outer-oof.csv']}))
        cache=joblib.load(args.feature_cache);train,test=cache['train'],cache['test'];frame=cache['views']['base']
        teacher=fit_recipe(frame.iloc[:len(train)],train.target,recipe_by_id(runner.TEACHER_RECIPE))
        source=control/'jobs'/f'full__{runner.TEACHER_RECIPE}';source.mkdir(parents=True);joblib.dump(teacher,source/'model.joblib')
        runner.write_json(source/'complete.json',dict(identity=dict(signature=sig,period='full',recipe=runner.TEACHER_RECIPE,train_rows=len(train),predict_rows=len(test)),status='complete',reload_exact=True,sha256={'model.joblib':runner.sha(source/'model.joblib')}))
        actual_assess=runner.assess
        def forced_gate(*values):
            gates=actual_assess(*values)
            for g in gates.values():g['eligible']=True
            return gates
        actual_choose=runner.choose_count
        def fixed_count(*values):
            _,curve=actual_choose(*values)
            return 50,curve
        with patch.object(runner,'assess',side_effect=forced_gate),patch.object(runner,'choose_count',side_effect=fixed_count):
            result=runner.run(args)
        assert result['complete'] and result['selected_outer_counts']==[50,50,50]
        assert result['final_seeds']==[42,17,2026] and result['full_teacher_reused']
        assert len(result['files'])==2 and result['new_q2_q3_metrics'] is False
        teacher_raw=np.load(args.out/'teacher-oof-raw.npy',allow_pickle=False)
        assert np.isnan(teacher_raw[train.forecast_period.eq('2022-01-01')]).all()
        assert np.isfinite(teacher_raw[train.forecast_period.ne('2022-01-01')]).all()
        for path in (args.out/'teacher').glob('*/complete.json'):
            rec=runner.read_json(path);assert rec['evaluation_metrics_computed'] is False
            if rec['predict_period']!='full':assert rec['train_max_period']<rec['predict_period']
        for path in (args.out/'meta').glob('*/complete.json'):
            rec=runner.read_json(path);assert rec['original_binary_targets'] and rec['no_in_sample_teacher_baselines'] and not rec['AP_computed']
        args.resume=True
        with patch.object(runner,'fit_recipe',side_effect=AssertionError('Teacher resume fit')),patch.object(runner,'fit_residual',side_effect=AssertionError('CB resume fit')),patch.object(runner,'assess',side_effect=forced_gate),patch.object(runner,'choose_count',side_effect=fixed_count):
            assert runner.run(args)==result
        for file in result['files']:
            csv=runner.read_csv(args.out/file['csv']);assert csv.row_id.equals(cache['sample'].row_id)
