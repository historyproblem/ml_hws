"""Temporal and reused-reference boundaries of the residual GPU screen."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pandas.testing as pdt

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_hw4_residual", ROOT / "scripts/run_hw4_residual.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_residual_preparation_excludes_closed_labels_and_future_snapshots():
    from test_hw4_extensions import fixture
    rows, history = fixture()
    rows = rows.reset_index(drop=True).assign(row_id=["r1", "r2"], forecast_period="2024-01-01", history_before="2023-12-25")
    history = history.assign(forecast_period="2024-01-01", inspection_week="2023-07-01")
    closed = rows.assign(row_id=["r3", "r4"], forecast_period="2024-04-01", history_before="2024-03-25")
    train = pd.concat([rows, closed], ignore_index=True)
    with TemporaryDirectory() as d:
        raw = Path(d)
        train.to_csv(raw / "train.csv", index=False)
        history.to_csv(raw / "inspection_history.csv", index=False)
        first = runner.prepare_development(raw)
        train.loc[train.forecast_period.gt("2024-01-01"), "target"] = 1 - train.target
        train.to_csv(raw / "train.csv", index=False)
        pd.concat([history, history.assign(forecast_period="2024-04-01", result="fail")]).to_csv(raw / "inspection_history.csv", index=False)
        second = runner.prepare_development(raw)
    pdt.assert_frame_equal(first[0], second[0])
    pdt.assert_frame_equal(first[1], second[1])
    np.testing.assert_array_equal(first[3], second[3])
    np.testing.assert_array_equal(first[4], second[4])
    assert len(first[1].columns) == 96 and first[2] == ["facility_type", "risk_level", "area_id"]
    assert first[5] == second[5]


def test_residual_reference_checks_hashes_specs_and_official_id_alignment():
    train = pd.DataFrame({"row_id": ["a", "b"], "site_id": ["s1", "s2"],
                          "forecast_period": "2023-07-01", "target": [0, 1]})
    files = {"train.csv": "known-input"}
    with TemporaryDirectory() as d:
        directory = Path(d)
        sources = ["src/hw4_inspections/features.py", "src/hw4_inspections/extensions.py",
                   "src/hw4_inspections/classical_features.py", "src/hw4_inspections/classical_models.py"]
        protocol = {"status": "complete", "config": {"smoke": False, "seeds": [42]}, "signature": "fixed",
                    "input_sha256": files, "lockbox_scored": False,
                    "source_sha256": {s: runner.sha(ROOT / s) for s in sources},
                    "specs": {s: runner.SPECS[s] for s in ["cb_reference", "cb_compact"]}}
        for model in ["cb_reference", "cb_compact"]:
            job = directory / f"{model}_s42_2023-07-01"
            job.mkdir()
            path = job / "predictions.csv"
            train.assign(model=model, seed=42, prediction=[.2, .8]).iloc[::-1].to_csv(path, index=False)
            (job / "complete.json").write_text(json.dumps({"signature": "fixed", "artifacts": {"predictions.csv": runner.sha(path)}}))
        p = directory / "protocol.json"
        p.write_text(json.dumps(protocol))
        result, provenance = runner.classical_reference(directory, train, files, ["2023-07-01"])
        assert len(result) == 4 and len(provenance["job_predictions_sha256"]) == 2
        np.testing.assert_array_equal(result.prediction.to_numpy(), [.2, .8, .2, .8])
        for source in sources:
            changed = json.loads(json.dumps(protocol))
            changed["source_sha256"][source] = "changed"
            p.write_text(json.dumps(changed))
            try:
                runner.classical_reference(directory, train, files, ["2023-07-01"])
                raise AssertionError("Changed feature/model source accepted")
            except ValueError:
                pass
        p.write_text(json.dumps(protocol))
        path.write_text(path.read_text() + "corrupt")
        try:
            runner.classical_reference(directory, train, files, ["2023-07-01"])
            raise AssertionError("Changed reference predictions accepted")
        except ValueError:
            pass
