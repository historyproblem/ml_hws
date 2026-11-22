"""Synthetic sequential-selection/reuse/numerical-failure interface checks."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sklearn.exceptions import ConvergenceWarning

from test_hw4_additive_runner import fixture
import run_hw4_additive_search as runner


def _args(root):
    args = fixture(root)
    args.fixed_core = None
    args.plan = runner.DEFAULT_PLAN
    return args


def test_additive_search_sequential24_inner_decisions_resume_and_full_interface():
    with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
        args = _args(Path(temp))
        actual_assess = runner.assess
        def forced_qa_gate(*values):
            gates = actual_assess(*values)
            gates["raw"]["eligible"] = True
            gates["blend75"]["eligible"] = False
            return gates
        with patch.object(runner, "assess", side_effect=forced_qa_gate):
            result = runner.run(args)
        assert result["complete"] and len(result["files"]) == 1
        assert result["inner_candidate_jobs"] == 72 and not result["outer_selection_of24recipes"]
        assert result["new_q2_q3_metrics"] is False
        for q, inner in zip(runner.primary.QUARTERS, ["2023-04-01", "2023-07-01", "2023-10-01"]):
            decision = runner.read_json(args.out / "decisions" / f"{q}.json")
            assert decision["inner"] == inner and len(decision["inner_complete_sha256"]) == 24
        assert all(p.name.split("__")[0] not in {"2024-04-01", "2024-07-01"}
                   for p in (args.out / "jobs").iterdir())
        args.resume = True
        with patch.object(runner, "fit_recipe", side_effect=AssertionError("Resume refit")), \
             patch.object(runner, "assess", side_effect=forced_qa_gate):
            assert runner.run(args) == result
        args.bootstrap = 1
        try:
            runner.run(args)
            raise AssertionError("Changed config accepted")
        except ValueError:
            pass


def test_additive_search_all_invalid_records_failure_not_drop_quarter():
    with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
        args = _args(Path(temp))
        with patch.object(runner, "fit_recipe", side_effect=ConvergenceWarning("Synthetic nonconvergence")):
            try:
                runner.run(args)
                raise AssertionError("All-invalid inner quarter was skipped")
            except RuntimeError:
                pass
        jobs = list((args.out / "jobs").glob("*/complete.json"))
        assert len(jobs) == 24
        assert all(runner.read_json(p)["status"] == "invalid" for p in jobs)
        assert not (args.out / "manifest.json").exists()


def test_additive_search_mixed_numerical_failure_csv_is_canonical_on_first_resume():
    with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
        args = _args(Path(temp))
        actual_fit = runner.fit_recipe
        def one_invalid(*values, **kwargs):
            if values[2]["id"] == "compact_all_splines_C10":
                raise ConvergenceWarning("Synthetic ordered error keys")
            return actual_fit(*values, **kwargs)
        with patch.object(runner, "fit_recipe", side_effect=one_invalid):
            first = runner.run(args)
        before = (args.out / "inner-records.csv").read_bytes()
        args.resume = True
        with patch.object(runner, "fit_recipe", side_effect=AssertionError("Resume refit")):
            assert runner.run(args) == first
        assert before == (args.out / "inner-records.csv").read_bytes()
