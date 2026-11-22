"""Bounded final-four runner checks, including native-model end-to-end replay."""
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hw4_final_research as runner


def expect_error(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def fixture(root, perfect=False):
    raw = root / "data"
    raw.mkdir()
    qs = pd.date_range("2022-01-01", periods=12, freq="QS").strftime("%Y-%m-%d")
    data = [{"row_id": f"{q}-{i}", "site_id": f"site{i}", "forecast_period": q,
             "target": int(i >= 8)} for q in qs for i in range(16)]
    all_rows = pd.DataFrame(data)
    train = all_rows.loc[all_rows.forecast_period.ne(qs[-1])].reset_index(drop=True)
    test = all_rows.loc[all_rows.forecast_period.eq(qs[-1])].drop(columns="target").reset_index(drop=True)
    sample = pd.DataFrame({"row_id": test.row_id.iloc[::-1].tolist(), "target": .5})
    for frame, name in [(train,"train.csv"),(test,"test_student.csv"),(sample,"sample_submission.csv")]:
        frame.to_csv(raw / name, index=False)
    pd.DataFrame({"site_id": []}).to_csv(raw / "inspection_history.csv", index=False)
    base = pd.DataFrame({"burden": np.tile(np.arange(16), len(qs)), "risk": ["medium","high"]*(8*len(qs))})
    views = {"base": base}
    for view, cols in [("transition",runner.TRANSITION_COLUMNS),("seasonal",runner.SEASONAL_COLUMNS)]:
        views[view] = base.assign(**{name: .1 for name in cols})
    cache = root / "features.joblib"
    joblib.dump({"train": train, "test":test, "sample":sample, "cats":["risk"], "views":views}, cache)
    mod = ROOT / "src" / "hw4_inspections"
    runner.write_json(root / "features-meta.json", {"cache_sha256":runner.sha(cache),
        "raw_sha256":{n:runner.sha(raw/n) for n in runner.RAW_FILES},
        "builder_sha256":{n:runner.sha(mod/n) for n in runner.BUILDERS}})
    cv = train.loc[train.forecast_period.isin(runner.QUARTERS)].copy()
    cv["prediction"] = cv.target if perfect else .5
    cv["model"] = runner.ANCHOR
    cv.iloc[::-1].to_csv(root / "oof.csv", index=False)
    sample.to_csv(root / "anchor.csv", index=False)
    out = root / "run"
    out.mkdir()
    return Namespace(raw=raw, out=out, feature_cache=cache, baseline_cv=root / "oof.csv",
                     context_test=root / "anchor.csv", smoke=True, threads=1, bootstrap=0, resume=False)


def test_final_research_native_fit_confirmation_full_resume_and_tamper():
    with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
        args = fixture(Path(temp))
        original = runner.choose_tree_count
        with patch.object(runner, "choose_tree_count", wraps=original) as selector:
            result = runner.run(args)
        assert selector.call_count == 8 * 3, "Tree search must run once per deep seed42/old fold"
        assert result["complete"] and len(result["files"]) == 4 and result["new_q2_q3_metrics"] is False
        assert len({entry["family"] for entry in result["files"]}) == 4
        assert result["primary"] == result["files"][0]["csv"]
        jobs = list((args.out / "jobs").glob("*/complete.json"))
        assert len(jobs) == 10*3 + 4*2*3 + 4*3
        for job in jobs:
            record = runner.read_json(job)
            id_ = record["identity"]
            assert record["reload_exact"] and record["iterations"] == 3
            assert id_["period"] in (*runner.QUARTERS, "full")
            if id_["period"] == "full":
                assert id_["train_rows"] == 11*16 and id_["predict_rows"] == 16
            elif id_["seed"] == 42 and id_["case"] in runner.SPECS:
                assert record["inner_quarter"] == (pd.Period(id_["period"],freq="Q")-1).start_time.strftime("%Y-%m-%d")
            else:
                assert record["inner_quarter"] is None
        sample = runner.read_csv(args.raw / "sample_submission.csv")
        for entry in result["files"]:
            csv = args.out / entry["csv"]
            assert entry["sha256"] == runner.sha(csv)
            assert runner.read_csv(csv).row_id.equals(sample.row_id)
        args.resume = True
        with patch.object(runner, "fit_booster", side_effect=AssertionError("resume refit")):
            assert runner.run(args) == result
        args.bootstrap = 1
        expect_error(lambda: runner.run(args))
        args.bootstrap = 0
        with (jobs[0].parent/"prediction.npy").open("ab") as handle:
            handle.write(b"tampered")
        expect_error(lambda: runner.run(args))


def test_final_research_empty_gate_produces_no_finals():
    with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
        args = fixture(Path(temp), perfect=True)
        result = runner.run(args)
        assert result["confirmed_cases"] == result["qualified_families"] == 0
        assert result["primary"] is None and not list(args.out.glob("submission*.csv"))
        assert len(list((args.out/"jobs").glob("*/complete.json"))) == 30


def test_final_research_cache_and_anchor_validation():
    with TemporaryDirectory() as temp:
        args = fixture(Path(temp))
        cache = joblib.load(args.feature_cache)
        train,test,_,_,_ = runner.validate_cache(cache, args)
        cache["views"]["transition"] = cache["views"]["transition"].iloc[::-1]
        expect_error(lambda: runner.validate_cache(cache, args))
        meta = train.loc[train.forecast_period.isin(runner.QUARTERS),list(runner.REQUIRED)].reset_index(drop=True)
        _, aligned = runner.align_anchors(meta,test,args)
        assert (aligned == .5).all()
        anchor = runner.read_csv(args.baseline_cv)
        anchor.loc[0,"site_id"] = "foreign"
        anchor.to_csv(args.baseline_cv, index=False)
        expect_error(lambda: runner.align_anchors(meta,test,args), AssertionError)
        provenance = args.feature_cache.with_name("features-meta.json")
        rec = runner.read_json(provenance)
        rec["raw_sha256"]["train.csv"] = "0"*64
        runner.write_json(provenance, rec)
        expect_error(lambda: runner.provenance(args,runner.cases(True)))


def test_final_research_extra_worst_quarter_floor_and_ties():
    outrows = pd.DataFrame({"model":[runner.ANCHOR,"cb_plain_d3__raw","cb_ordered_d5__raw","cb_plain_d6__raw"],
                            "mean_ap":[.26,.27,.27,.28],"delta_vs_reference":[0,.01,.01,.02],
                            "worst_quarter_delta":[0,-.005,-.004,-.00501],"eligible":[False,True,True,True]})
    result = {"summary":outrows,"quarter_scores":pd.DataFrame({"ap":[]})}
    with TemporaryDirectory() as temp, patch.object(runner,"evaluate_candidates",return_value=result):
        p=np.array([.1,.6])
        picked=runner.assess("screen",Path(temp),pd.DataFrame({"target":[0,1]}),p,
                             {k:p.copy() for k in ["cb_plain_d3","cb_ordered_d5","cb_plain_d6"]},
                             runner.cases(False),0)
        assert [r["case"] for r in picked] == ["cb_ordered_d5"]
        summary=runner.read_csv(Path(temp)/"screen-summary.csv").set_index("model")
        assert summary.loc["cb_plain_d3__raw","eligible"] and not summary.loc["cb_plain_d6__raw","eligible"]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS", flush=True)
