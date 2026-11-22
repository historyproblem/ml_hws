"""Real tiny classifiers verify reuse/replay and two unscored control fits."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_hw4_recent_baselines as runner
from hw4_inspections.additive_search import fit_recipe, recipe_by_id
from hw4_inspections.classical_models import fit_model, predict_model
from test_hw4_additive_runner import fixture as causal_fixture


def fixture(root):
    args = causal_fixture(root)
    args.smoke = True
    args.delivery_run = root / "legacy"
    args.teacher_run = root / "teacher"
    args.delivery_run.mkdir()
    args.teacher_run.mkdir()
    args.teacher_run.joinpath("teacher").mkdir()
    hashes = {n: runner.sha(args.raw / n) for n in runner.RAW}
    source = {"src/hw4_inspections/classical_models.py": runner.sha(ROOT / "src/hw4_inspections/classical_models.py")}
    parent = dict(recipe=dict(seeds=list(runner.SEEDS), cb_iterations=500, holdout=list(runner.QUARTERS)),
        catboost_specs=dict(cb_reference=runner.CORE_SPEC), input_sha256=hashes, source_sha256=source)
    parent["signature"] = runner.digest(parent)
    parent["frozen_before_holdout_scoring"] = True
    runner.write(args.delivery_run / "frozen_recipe.json", parent)
    teacher = dict(raw_sha256=hashes, cache_sha256=runner.sha(args.feature_cache),
        pre_results_plan=dict(teacher=dict(recipe="core_interactions_C0.01")),
        source_sha256={"additive_search.py": runner.sha(ROOT / "src/hw4_inspections/additive_search.py")})
    teacher["signature"] = runner.digest(teacher)
    runner.write(args.teacher_run / "freeze.json", teacher)
    cache = joblib.load(args.feature_cache)
    train, frame, cats = cache["train"], cache["views"]["base"], cache["cats"]
    for quarter in runner.QUARTERS:
        fit = np.flatnonzero(train.forecast_period.lt(quarter))
        valid = np.flatnonzero(train.forecast_period.eq(quarter))
        for seed in runner.SEEDS:
            bundle = fit_model(runner.CORE_SPEC, frame.iloc[fit], train.target.iloc[fit], cats, seed, 2, iterations=500)
            p = predict_model(bundle, frame.iloc[valid])
            path = args.delivery_run / f"holdout_{quarter}" / f"cb_reference_s{seed}"
            path.mkdir(parents=True)
            joblib.dump(bundle, path / "model.joblib", compress=1)
            np.save(path / "prediction.npy", p, allow_pickle=False)
            runner.write(path / "complete.json", dict(signature=parent["signature"], stage=f"holdout_{quarter}",
                model="cb_reference", seed=seed, train_rows=len(fit), prediction_rows=len(valid),
                loaded_forward_exact=True, artifacts={n: runner.sha(path / n) for n in ["model.joblib", "prediction.npy"]}))
        bundle = fit_recipe(frame.iloc[fit], train.target.iloc[fit], recipe_by_id("core_interactions_C0.01"))
        raw = bundle["model"].decision_function(bundle["transformer"].transform(frame.iloc[valid]))
        path = args.teacher_run / "teacher" / quarter
        path.mkdir()
        joblib.dump(bundle, path / "model.joblib", compress=1)
        np.save(path / "raw.npy", raw, allow_pickle=False)
        np.save(path / "row-ids.npy", train.row_id.iloc[valid].to_numpy(dtype=str), allow_pickle=False)
        runner.write(path / "complete.json", dict(identity=dict(signature=teacher["signature"],kind="teacher",
            recipe="core_interactions_C0.01",quarter=quarter,train_rows=len(fit),predict_rows=len(valid)),
            train_max_period=train.forecast_period.iloc[fit].max(),reload_raw_exact=True,evaluation_metrics_computed=False,
            sha256={n: runner.sha(path / n) for n in ["model.joblib", "raw.npy", "row-ids.npy"]}))
    return args


def test_recent_baselines_reuse_native_models_no_scoring_and_zero_fit_resume():
    with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
        args = fixture(Path(temporary))
        result = runner.run(args)
        assert result["status"] == "COMPLETE_UNSCORED" and result["rows"] == 32
        assert result["metrics_computed"] == result["new_context_fits"] == result["new_fixed_E_fits"] == 0
        assert result["new_old_gam_fits"] == 2
        p = runner.csv(args.out / "predictions.csv")
        assert np.array_equal(p.previous07, .75 * p.context_3seed + .25 * p.old_gam_core)
        args.resume = True
        with patch.object(runner, "fit_old_core", side_effect=AssertionError("Resume refit old GAM")):
            assert runner.run(args) == result
        raw = args.out / "old-gam" / runner.QUARTERS[0] / "prediction.npy"
        raw.write_bytes(raw.read_bytes() + b"corrupt")
        try:
            runner.run(args)
            raise AssertionError("Corrupted unscored control checkpoint accepted")
        except ValueError:
            pass
