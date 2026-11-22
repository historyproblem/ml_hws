"""Synthetic exercise of the entire seed bag, including original artifact replay."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_hw4_context_bag10 as bag
from test_hw4_final_research import fixture, expect_error

r = bag.primary


def with_originals(root, perfect_oof=False):
    args = fixture(root, perfect=perfect_oof)
    args.delivery_run = root/"original"
    d = args.delivery_run
    d.mkdir()
    cache = joblib.load(args.feature_cache)
    train,test,sample,views,cats = r.validate_cache(cache,args)
    x=views["base"]
    raw = {n:r.sha(args.raw/n) for n in r.RAW_FILES}
    record = {"recipe":{"seeds":list(bag.OLD),"cb_iterations":3}, "input_sha256":raw,
              "source_sha256":{p:r.sha(r.ROOT/p) for p in bag.PARENT_SOURCES}}
    signature=bag.digest(record)
    record.update(signature=signature, frozen_before_holdout_scoring=True)
    r.write_json(d/"frozen_recipe.json",record)
    values, compatibility = [], []
    for seed in bag.OLD:
        bundle=r.fit_booster({"family":"catboost","params":bag.PARAMS,"max_iterations":3},
                             x.iloc[:len(train)],train.target,cats,seed=seed,threads=1,iterations=3)
        job=d/"full_refit"/f"cb_reference_s{seed}"
        job.mkdir(parents=True)
        joblib.dump(bundle,job/"model.joblib")
        pred=bag.predict_model(bundle,x.iloc[len(train):])
        values.append(pred)
        np.save(job/"prediction.npy",pred)
        artifacts={n:r.sha(job/n) for n in ["model.joblib","prediction.npy"]}
        r.write_json(job/"complete.json",{"signature":signature,"stage":"full_refit","model":"cb_reference",
                      "seed":seed,"train_rows":len(train),"prediction_rows":len(test),"loaded_forward_exact":True,
                      "artifacts":artifacts})
        compatibility.append({"seed":seed,"model_sha256":artifacts["model.joblib"], "feature_names_exact":True,
                              "categorical_names_exact":True})
    sub=r.validate_submission(test,sample,np.mean(values,axis=0),len(test))
    r.write_csv(args.context_test,sub)
    r.write_json(d/"manifest.json",{"signature":signature,"status":"complete","submissions":{
                    "context_cb":{"sha256":r.sha(args.context_test)}},"full_train_rows":len(train),"test_rows":len(test)})
    checks=[{"seed":seed,"quarter":q, "saved_oof_forward_exact":True,"current_prediction_helper_forward_exact":True,
             "feature_schema_exact":True,"categorical_indices_exact":True,
             "all_parameter_differences_are_identical_float32_values":True} for seed in bag.OLD for q in r.QUARTERS]
    r.write_json(args.baseline_cv.with_name("baseline-audit.json"),{"status":"PASS","seeds":list(bag.OLD),
                "quarters":list(r.QUARTERS),"current_official_input_sha256":raw,
                "baseline_oof_sha256":r.sha(args.baseline_cv),"validation_rows":int(train.forecast_period.isin(r.QUARTERS).sum()),
                "baseline_csv_roundtrip_exact":True,"all_effective_parameters_match_except_float_serialization_precision":True,
                "checkpoint_checks":checks,"source_sha256":record["source_sha256"],
                "current_full_refit_compatibility":compatibility})
    return args


def test_bag10_exact_weighting_and_fixed_seed_set():
    assert bag.OLD+bag.ADDED==(42,17,2026,7,23,101,123,456,777,1000)
    old=np.array([.2,.8]); additions=[np.array([.6,.4]) for _ in bag.ADDED]
    expected=(3*old+np.sum(additions,axis=0))/10
    np.testing.assert_array_equal(bag.bag_mean(old,additions),expected)
    expect_error(lambda:bag.bag_mean(old,additions[:-1]))


def test_bag10_real_backends_positive_route_full_seed_and_reload():
    with TemporaryDirectory() as temp,redirect_stdout(StringIO()):
        args=with_originals(Path(temp))
        with patch.object(r,"choose_tree_count",side_effect=AssertionError("No tree tuning")):
            manifest=bag.run(args)
        assert manifest["eligible"] and len(manifest["files"])==1
        jobs=sorted((args.out/"jobs").glob("*/complete.json"))
        assert len(jobs)==28
        for path in jobs:
            marker=r.read_json(path)
            assert marker["iterations"]==3 and marker["reload_exact"]
            assert marker["identity"]["seed"] in bag.ADDED
            assert marker["identity"]["period"] in (*r.QUARTERS,"full")
        source=joblib.load(args.feature_cache)
        components=[np.load(args.delivery_run/"full_refit"/f"cb_reference_s{x}"/"prediction.npy") for x in bag.OLD]
        new=[np.load(args.out/"jobs"/f"{bag.NAME}__full__s{x}"/"prediction.npy") for x in bag.ADDED]
        expected=r.validate_submission(source["test"],source["sample"],bag.bag_mean(np.mean(components,axis=0),new))
        actual=r.read_csv(args.out/manifest["files"][0]["file"])
        r.pd.testing.assert_frame_equal(expected,actual,check_exact=True)
        args.resume=True
        with patch.object(r,"fit_booster",side_effect=AssertionError("No resume fits")):
            assert bag.run(args)==manifest
        # A changed root marker (seed attribution) must fail even when sibling hashes remain valid.
        marker=args.delivery_run/"full_refit"/"cb_reference_s17"/"complete.json"
        entry=r.read_json(marker);entry["seed"]=101
        r.write_json(marker,entry)
        expect_error(lambda:bag.run(args))


def test_bag10_no_full_job_or_csv_on_failure():
    with TemporaryDirectory() as temp,redirect_stdout(StringIO()):
        args=with_originals(Path(temp),perfect_oof=True)
        result=bag.run(args)
        assert not result["eligible"] and not result["files"] and not list(args.out.glob("submission*.csv"))
        assert len(list((args.out/"jobs").glob("*/complete.json")))==21


if __name__=="__main__":
    for name,test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS",flush=True)
