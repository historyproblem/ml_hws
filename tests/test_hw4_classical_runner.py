"""Protect the temporal screen and identity checks at the orchestration boundary."""
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util

import numpy as np
import pandas as pd
import pandas.testing as pdt

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_hw4_classical", ROOT / "scripts/run_hw4_classical.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_classical_lockbox_labels_and_future_history_do_not_enter_features():
    from test_hw4_extensions import fixture
    rows, history = fixture()
    rows = rows.reset_index(drop=True).assign(row_id=["r1", "r2"], forecast_period="2024-01-01", history_before="2023-12-25")
    history = history.assign(forecast_period="2024-01-01", inspection_week="2023-07-01")
    future = rows.assign(forecast_period="2024-04-01", history_before="2024-03-25", row_id=["r3", "r4"])
    train = pd.concat([rows, future], ignore_index=True)
    with TemporaryDirectory() as directory:
        raw = Path(directory)
        train.to_csv(raw / "train.csv", index=False)
        history.to_csv(raw / "inspection_history.csv", index=False)
        early, x, relative, cats = runner.prepare_development(raw)
        changed = train.copy()
        changed.loc[changed.forecast_period > "2024-01-01", "target"] = 1 - changed.target
        changed.to_csv(raw / "train.csv", index=False)
        bad_history = pd.concat([history, history.assign(forecast_period="2024-04-01", result="fail", violation_codes="1 2 3 4")])
        bad_history.to_csv(raw / "inspection_history.csv", index=False)
        later, other, other_relative, other_cats = runner.prepare_development(raw)
    pdt.assert_frame_equal(early, later)
    pdt.assert_frame_equal(x, other)
    pdt.assert_frame_equal(relative, other_relative)
    assert cats == other_cats and early.forecast_period.eq("2024-01-01").all()
    assert not {"row_id", "site_id", "target"}.intersection(x)


def test_classical_reference_rejects_wrong_identity_and_duplicate_rows():
    rows = pd.DataFrame({"row_id": ["a", "b"], "site_id": ["x", "y"],
                         "forecast_period": ["2023-07-01", "2024-01-01"], "target": [0, 1]})
    original = rows.assign(case="plus_context", seed=42, prediction=[.1, .8])
    with TemporaryDirectory() as directory:
        path = Path(directory) / "reference.csv"
        original.iloc[::-1].to_csv(path, index=False)
        out = runner.aligned_reference(path, rows, 42)
        np.testing.assert_array_equal(out.to_numpy(), [.1, .8])
        assert runner.aligned_reference(path, rows, 17) is None
        for bad in [pd.concat([original, original.iloc[:1]]),
                    original.assign(site_id=["wrong", "y"]),
                    original.assign(target=[1, 1]), original.assign(prediction=[np.nan, .8])]:
            bad.to_csv(path, index=False)
            try:
                runner.aligned_reference(path, rows, 42)
                raise AssertionError("Corrupt reference was accepted")
            except ValueError:
                pass
