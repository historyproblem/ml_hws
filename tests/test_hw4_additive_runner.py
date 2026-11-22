"""Synthetic-only end-to-end additive gate, export and strict resume checks."""
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hw4_additive as runner
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features
from hw4_inspections.seasonal_context import build_seasonal_context_features
from hw4_inspections.transition_features import build_transition_features


def fixture(root, perfect=False):
    raw = root / "data"
    raw.mkdir()
    quarters = pd.date_range("2022-01-01", periods=12, freq="QS")
    data, visits = [], []
    for quarter in quarters:
        cutoff = quarter - pd.Timedelta(days=7)
        for i in range(16):
            fields = dict(site_id=f"s{i}", forecast_period=quarter.strftime("%Y-%m-%d"),
                          facility_type="restaurant", risk_level="high", area_id=f"a{i%2}")
            data.append(dict(fields, row_id=f"{quarter:%Y%m%d}-{i}",
                             history_before=cutoff.strftime("%Y-%m-%d"), target=int(i >= 8)))
            for days in [180, 90]:
                week = cutoff - pd.Timedelta(days=days)
                week -= pd.Timedelta(days=week.dayofweek)
                visits.append(dict(fields, inspection_week=week.strftime("%Y-%m-%d"),
                                   inspection_type="routine", result="fail" if i >= 8 else "pass",
                                   violation_codes="1 2 3 4 5 6 7 8 9 10 11 12" if i >= 8 else "1"))
    rows, history = pd.DataFrame(data), pd.DataFrame(visits)
    train = rows.loc[rows.forecast_period.ne("2024-10-01")].reset_index(drop=True)
    test = rows.loc[rows.forecast_period.eq("2024-10-01")].drop(columns="target").reset_index(drop=True)
    sample = pd.DataFrame({"row_id": test.row_id.iloc[::-1].to_numpy(), "target": .5})
    for frame, name in [(train, "train.csv"), (test, "test_student.csv"),
                         (sample, "sample_submission.csv"), (history, "inspection_history.csv")]:
        frame.to_csv(raw / name, index=False)
    features = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    base, cats = build_features(features, history)
    base = pd.concat([base, build_context_features(features, history)], axis=1)
    views = dict(base=base,
                 transition=pd.concat([base, build_transition_features(features, history)], axis=1),
                 seasonal=pd.concat([base, build_seasonal_context_features(features, history)], axis=1))
    cache = root / "features.joblib"
    joblib.dump(dict(train=train, test=test, sample=sample, views=views, cats=cats), cache)
    module = ROOT / "src" / "hw4_inspections"
    runner.write_json(root / "features-meta.json", dict(cache_sha256=runner.sha(cache),
        raw_sha256={n: runner.sha(raw / n) for n in runner.primary.RAW_FILES},
        builder_sha256={n: runner.sha(module / n) for n in runner.primary.BUILDERS}))
    anchor = train.loc[train.forecast_period.isin(runner.primary.QUARTERS), list(runner.primary.REQUIRED)].copy()
    anchor["prediction"] = anchor.target if perfect else .5
    anchor["model"] = runner.primary.ANCHOR
    anchor.iloc[::-1].to_csv(root / "anchor-oof.csv", index=False)
    sample.to_csv(root / "anchor-test.csv", index=False)
    out = root / "out"
    out.mkdir()
    return Namespace(raw=raw, out=out, feature_cache=cache,
                     baseline_cv=root / "anchor-oof.csv", context_test=root / "anchor-test.csv",
                     bootstrap=0, smoke=True, resume=False)


def test_additive_runner_real_model_export_resume_and_tamper():
    with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
        args = fixture(Path(temp))
        result = runner.run(args)
        assert result["complete"] and result["conditional_full_fits"] == 1
        assert len(result["files"]) == 1 and not result["new_q2_q3_metrics"]
        assert len(list((args.out / "jobs").glob("*/complete.json"))) == 4
        path = args.out / result["primary"]
        sample = runner.read_csv(args.raw / "sample_submission.csv")
        assert runner.read_csv(path).row_id.equals(sample.row_id)
        assert runner.sha(path) == result["files"][0]["sha256"]
        args.resume = True
        with patch.object(runner, "fit_additive", side_effect=AssertionError("Unexpected resume fit")):
            assert runner.run(args) == result
        args.bootstrap = 1
        try:
            runner.run(args)
            raise AssertionError("Changed configuration accepted")
        except ValueError:
            pass
        args.bootstrap = 0
        with (args.out / "jobs" / "2023-07-01" / "prediction.npy").open("ab") as handle:
            handle.write(b"tamper")
        try:
            runner.run(args)
            raise AssertionError("Changed job prediction accepted")
        except ValueError:
            pass


def test_additive_runner_failed_gate_never_fits_full():
    with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
        args = fixture(Path(temp), perfect=True)
        result = runner.run(args)
        assert result["complete"] and result["conditional_full_fits"] == 0
        assert result["primary"] is None and result["files"] == []
        assert not (args.out / "jobs" / "full").exists()
        assert len(list((args.out / "jobs").glob("*/complete.json"))) == 3
