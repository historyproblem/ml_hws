"""Synthetic-only checks for a frozen one-score temporal diagnostic."""
from argparse import Namespace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_hw4_recent_finalists as runner
from hw4_inspections.recent_diagnostic import META, QUARTERS, align_recent, describe_recent


def fixture(root):
    raw = root / "raw"
    raw.mkdir()
    rows = []
    for q in QUARTERS:
        for i in range(8):
            rows.append(dict(row_id=f"{q}-{i}", site_id=f"s{i}", forecast_period=q, target=i % 2))
    train = pd.DataFrame(rows)
    train.to_csv(raw / "train.csv", index=False)
    pd.DataFrame(dict(row_id=["test"], forecast_period=["2024-10-01"])).to_csv(raw / "test_student.csv", index=False)
    pd.DataFrame(dict(row_id=["test"], target=[.5])).to_csv(raw / "sample_submission.csv", index=False)
    pd.DataFrame(dict(site_id=["s1"])).to_csv(raw / "inspection_history.csv", index=False)
    probabilities = train.assign(context=.5, candidate=.2 + .6 * train.target).iloc[::-1]
    probabilities.to_csv(root / "predictions.csv", index=False)
    (root / "source.txt").write_text("fixed synthetic recipe source\n")
    (root / "old-evidence.txt").write_text("synthetic earlier selection\n")
    recipes = dict(format="hw4-recent-finalists-1", reference="context", hyperparameter_search=False,
                   weight_search=False, formulas_frozen=True,
                   finalists=[dict(id=name, recipe=dict(fixed=True), formula=name,
                       sources=["source.txt"], old_selection_evidence=["old-evidence.txt"])
                       for name in ["context", "candidate"]])
    runner.write(root / "recipes.json", recipes)
    out = root / "diagnostic"
    out.mkdir()
    return Namespace(raw=raw, predictions=root / "predictions.csv", recipes=root / "recipes.json",
                     out=out, bootstrap=20, seed=42, smoke=True, resume=False)


def test_recent_alignment_and_descriptive_common_site_bootstrap():
    with TemporaryDirectory() as temporary:
        args = fixture(Path(temporary))
        train = runner.csv(args.raw / "train.csv")
        aligned = align_recent(train, runner.csv(args.predictions), ["context", "candidate"])
        assert aligned.row_id.equals(train.row_id)
        result = describe_recent(aligned, ["context", "candidate"], "context", bootstrap=100, seed=42)
        by_model = result["summary"].set_index("model")
        assert by_model.loc["context", "mean_ap"] == .5
        assert by_model.loc["candidate", "mean_ap"] == 1.
        assert result["protocol"]["bootstrap"]["common_weights_across_models_and_quarters"]
        assert not result["protocol"]["fresh_holdout"] and not result["protocol"]["model_selection_performed"]
        malformed = aligned.copy()
        malformed.loc[0, "forecast_period"] = "2024-10-01"
        try:
            align_recent(train, malformed, ["context", "candidate"])
            raise AssertionError("Wrong official period accepted")
        except AssertionError as error:
            assert str(error) != "Wrong official period accepted"
        duplicated = aligned.iloc[:-1].copy()
        duplicated = pd.concat([duplicated, duplicated.iloc[:1]])
        try:
            align_recent(train, duplicated, ["context", "candidate"])
            raise AssertionError("Duplicate IDs accepted")
        except ValueError:
            pass


def test_recent_freeze_no_score_then_single_score_zero_score_resume_and_tamper():
    with TemporaryDirectory() as temporary:
        args = fixture(Path(temporary))
        with patch.object(runner, "describe_recent", side_effect=AssertionError("Freeze must not score")):
            result = runner.freeze(args)
        assert result["status"] == "FROZEN_UNSCORED" and result["metrics_computed"] == 0
        scored = runner.score(args)
        assert scored["status"] == "COMPLETE"
        output = args.out / "results/summary.csv"
        digest = runner.sha(output)
        args.resume = True
        with patch.object(runner, "describe_recent", side_effect=AssertionError("Resume rescored")):
            assert runner.score(args)["new_metrics_computed"] == 0
        assert runner.sha(output) == digest
        original = args.predictions.read_bytes()
        args.predictions.write_bytes(original + b"\n")
        try:
            runner.score(args)
            raise AssertionError("Changed probabilities accepted")
        except ValueError:
            pass
        args.predictions.write_bytes(original)
        output.write_bytes(output.read_bytes() + b"corrupt")
        try:
            runner.score(args)
            raise AssertionError("Changed completed summary accepted")
        except ValueError:
            pass


def test_recent_incomplete_score_fails_closed_without_new_metrics():
    with TemporaryDirectory() as temporary:
        args = fixture(Path(temporary))
        runner.freeze(args)
        runner.write(args.out / "scoring_started.json", dict(interrupted=True))
        with patch.object(runner, "describe_recent", side_effect=AssertionError("Interrupted scoring restarted")):
            try:
                runner.score(args)
                raise AssertionError("Interrupted diagnostic auto-rescored")
            except ValueError:
                pass


def test_recent_snapshot_digest_is_inside_signed_request():
    with TemporaryDirectory() as temporary:
        args = fixture(Path(temporary))
        runner.freeze(args)
        path = args.out / "predictions.frozen.csv"
        altered = runner.csv(path)
        altered.loc[0, "context"] = .9
        altered.to_csv(path, index=False)
        frozen = runner.read(args.out / "freeze.json")
        # Altering only the convenience envelope cannot rebind the signed data.
        frozen["predictions_sha256"] = runner.sha(path)
        runner.write(args.out / "freeze.json", frozen)
        with patch.object(runner, "describe_recent", side_effect=AssertionError("Altered snapshot scored")):
            try:
                runner.score(args)
                raise AssertionError("Unsigned canonical snapshot alteration accepted")
            except ValueError:
                pass
