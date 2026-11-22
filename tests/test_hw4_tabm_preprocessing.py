"""Fit-only boundaries, quantiles, masks and native checkpoint smoke checks."""
from tempfile import TemporaryDirectory
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hw4_inspections.tabm_preprocessing import BoundedTabMPreprocessor, fit_bounded
from hw4_inspections.tabm_training import predict_bundle


def fixture():
    return pd.DataFrame(dict(value=np.arange(32, dtype=float), constant=1.,
                             absent=np.nan, group=["a", "b"] * 16))


def test_bounded_tabm_training_only_ranges_and_binary_missing_flags():
    frame = fixture()
    for mode in ["clip", "quantile"]:
        transform = BoundedTabMPreprocessor(bins=4, mode=mode).fit(frame, ["group"])
        original = transform.transform(frame)[0]
        query = frame.iloc[:3].copy()
        query["value"] = [-1e20, 1e20, np.nan]
        query["constant"] = [1e20, -1e20, np.nan]
        query["group"] = ["unseen", "a", "b"]
        low = transform.transform(query)[0]
        assert np.isfinite(low).all()
        count = len(transform.numeric_names)
        assert (low[:, :count] >= original[:, :count].min(axis=0) - 1e-6).all()
        assert (low[:, :count] <= original[:, :count].max(axis=0) + 1e-6).all()
        assert set(np.unique(low[:, count:])) <= {0., 1.}
        assert low[2, count] == 1.
        assert transform.transform(query)[1][0, 0] == 0
        assert np.array_equal(original, transform.transform(frame)[0])


def test_bounded_tabm_official_fit_checkpoint_and_factory_restored():
    from hw4_inspections import tabm_training
    original_factory = tabm_training.TabMPreprocessor
    frame = fixture()
    for mode in ["clip", "quantile"]:
        bundle = fit_bounded(frame, np.arange(32) % 2, ["group"], mode=mode,
                             k=2, seed=42, threads=1, epochs=1, patience=1,
                             width=8, blocks=1, bins=4, embedding_dim=2,
                             batch_size=16, device="cpu")
        assert tabm_training.TabMPreprocessor is original_factory
        prediction = predict_bundle(bundle, frame)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.joblib"
            joblib.dump(bundle, path)
            assert np.array_equal(prediction, predict_bundle(joblib.load(path), frame))
