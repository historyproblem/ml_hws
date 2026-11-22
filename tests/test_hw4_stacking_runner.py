"""Synthetic chronological base reuse/backfill, no-recent-AP and meta replay."""
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
import run_hw4_stacking as runner
from hw4_inspections.additive_search import fit_recipe,recipe_by_id
from hw4_inspections.gam_residual import TEACHER_RECIPE,teacher_raw
from hw4_inspections.classical_models import SPECS
from hw4_inspections.stacking import fit_reference,predict_reference


def test_stacking_pipeline_prequential15CBfits_sources_recent_noAP_full_and_resume():
    with TemporaryDirectory() as temp,redirect_stdout(StringIO()):
        root=Path(temp);args=fixture(root);args.threads=2;args.plan=runner.DEFAULT_PLAN
        library=root/'teacher-library';library.mkdir();args.teacher_library=library
        recent=root/'recent';recent.mkdir();args.recent_components=recent
        control=root/'control';control.mkdir();args.algorithm_control=control
        cache=joblib.load(args.feature_cache);train,test,frame,cats=cache['train'],cache['test'],cache['views']['base'],cache['cats']
        evidence=runner.primary.provenance(args,{})
        module=runner.ROOT/'src/hw4_inspections'
        evidence['source_sha256'].update({n:runner.sha(module/n) for n in ['additive_logit.py','classical_features.py','additive_search.py','gam_residual.py']})
        full=root/'full-teacher';full.mkdir()
        bundle=fit_recipe(frame.iloc[:len(train)],train.target,recipe_by_id(TEACHER_RECIPE));joblib.dump(bundle,full/'model.joblib')
        runner.write_json(full/'complete.json',dict(origin='Synthetic previous full source'))
        evidence['teacher_sources']={'full':dict(path=str(full),complete_sha256=runner.sha(full/'complete.json'),artifacts={'model.joblib':runner.sha(full/'model.joblib')})}
        sig=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest();evidence['signature']=sig;runner.write_json(library/'freeze.json',evidence)
        all_raw=np.full(len(train),np.nan)
        for q in sorted(train.forecast_period.unique())[1:]:
            ti=np.flatnonzero(train.forecast_period.lt(q));vi=np.flatnonzero(train.forecast_period.eq(q));bundle=fit_recipe(frame.iloc[ti],train.target.iloc[ti],recipe_by_id(TEACHER_RECIPE));raw=teacher_raw(bundle,frame.iloc[vi]);all_raw[vi]=raw
            path=library/'teacher'/q;path.mkdir(parents=True);joblib.dump(bundle,path/'model.joblib');np.save(path/'raw.npy',raw);np.save(path/'row-ids.npy',train.row_id.iloc[vi].to_numpy(str))
            runner.write_json(path/'complete.json',dict(identity=dict(signature=sig,quarter=q,train_rows=len(ti),predict_rows=len(vi)),evaluation_metrics_computed=False,train_max_period=train.forecast_period.iloc[ti].max(),sha256={n:runner.sha(path/n) for n in ['model.joblib','raw.npy','row-ids.npy']}))
        np.save(library/'teacher-oof-raw.npy',all_raw)
        runner.write_json(library/'manifest.json',dict(signature=sig,complete=True,output_sha256={'teacher-oof-raw.npy':runner.sha(library/'teacher-oof-raw.npy')}))
        oof=runner.read_csv(args.baseline_cv).drop(columns=['prediction','model']);oof['context_3seed']=.5;oof['algorithm_raw']=.5;runner.write_csv(control/'outer-oof.csv',oof);runner.write_json(control/'manifest.json',dict(complete=True))
        delivered=dict(input_sha256=evidence['raw_sha256'],catboost_specs={'cb_reference':SPECS['cb_reference']},source_sha256={f'src/hw4_inspections/{n}':runner.sha(module/n) for n in ['features.py','extensions.py','classical_features.py','classical_models.py']})
        delivery_sig=hashlib.sha256(json.dumps(delivered,sort_keys=True).encode()).hexdigest();delivered['signature']=delivery_sig;delivered['frozen_before_holdout_scoring']=True;runner.write_json(recent/'frozen_recipe.json',delivered)
        for q in runner.RECENT:
            ti=np.flatnonzero(train.forecast_period.lt(q));vi=np.flatnonzero(train.forecast_period.eq(q))
            for seed in runner.REFERENCE_SEEDS:
                path=recent/f'holdout_{q}'/f'cb_reference_s{seed}';path.mkdir(parents=True);bundle=fit_reference(frame.iloc[ti],train.target.iloc[ti],cats,seed,iterations=5);joblib.dump(bundle,path/'model.joblib');np.save(path/'prediction.npy',predict_reference(bundle,frame.iloc[vi]))
                runner.write_json(path/'complete.json',dict(signature=delivery_sig,stage=f'holdout_{q}',model='cb_reference',seed=seed,loaded_forward_exact=True,train_rows=len(ti),artifacts={n:runner.sha(path/n) for n in ['model.joblib','prediction.npy']}))
        actual_assess=runner.assess
        def forced_gate(*values):
            gate=actual_assess(*values);gate['eligible']=True;return gate
        actual_evaluate=runner.evaluate_candidates
        def old_only(rows,*values,**kwargs):
            assert set(rows.forecast_period)==set(runner.primary.QUARTERS)
            return actual_evaluate(rows,*values,**kwargs)
        with patch.object(runner,'assess',side_effect=forced_gate),patch.object(runner,'evaluate_candidates',side_effect=old_only):result=runner.run(args)
        assert result['complete'] and len(result['files'])==1 and result['new_full_base_fits']==0
        assert result['new_q2_q3_metrics'] is False and len(list((args.out/'early-cb').glob('*/complete.json')))==15
        for q in runner.RECENT:
            rec=runner.read_json(args.out/'meta'/q/'complete.json');assert rec['train_max_period']<q and rec['prediction_metrics_computed'] is False
        args.resume=True
        with patch.object(runner,'fit_reference',side_effect=AssertionError('CB resume fit')),patch.object(runner,'fit_stacker',side_effect=AssertionError('Meta resume fit')),patch.object(runner,'assess',side_effect=forced_gate):assert runner.run(args)==result
