"""Tiny-only TabM optimization, inner selection and fresh-refit checks."""
from io import BytesIO
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

from hw4_inspections.tabm_models import TabMPreprocessor, build_tabm
from hw4_inspections.tabm_training import OPTIMIZER, fit_tabm, predict_bundle


def _fixture():
    rng = np.random.default_rng(42)
    train = pd.DataFrame({"x": rng.normal(size=20), "empty": [np.nan] * 20,
                          "category": ["a", "b"] * 10})
    train.loc[3, "x"] = np.nan
    evaluation = pd.DataFrame({"x": [1e20, -1e20, 0., np.nan],
                               "empty": [0., 9., np.nan, -5.],
                               "category": ["unknown", "a", "b", "unknown"]})
    return train, np.array([0, 1] * 10, dtype=np.float32), evaluation, np.array([1, 0, 0, 1], dtype=np.float32)


def _fit(train, y, **kwargs):
    options = dict(k=4, seed=42, device="cpu", epochs=2, patience=2,
                   batch_size=8, width=8, blocks=2, bins=4, embedding_dim=4, threads=1)
    options.update(kwargs)
    return fit_tabm(train, y, ["category"], **options)


def test_tabm_fixed_fit_never_calls_ap_and_runs_exact_epoch_count():
    train, y, evaluation, _ = _fixture()
    before = train.copy(deep=True)
    with patch("hw4_inspections.tabm_training.average_precision_score", side_effect=AssertionError("AP called during fixed fit")):
        for k in [1, 4]:
            bundle = _fit(train, y, k=k)
            assert bundle["epochs"] == bundle["trained_epochs"] == 2
            assert len(bundle["curve"]) == 2 and bundle["best_ap"] is None
            assert not any("inner_ap" in row for row in bundle["curve"])
            assert bundle["selected_on_inner_ap"] is False and bundle["selection_rows"] == 0
            p = predict_bundle(bundle, evaluation)
            assert p.dtype == np.float32 and np.isfinite(p).all()
            assert bundle["model_kwargs"]["k"] == k
            assert bundle["optimizer"] == OPTIMIZER
    pd.testing.assert_frame_equal(train, before)


def test_tabm_inner_selection_uses_earliest_best_and_stops_at_patience():
    train, y, evaluation, target = _fixture()
    with patch("hw4_inspections.tabm_training.average_precision_score", side_effect=[.4, .4, .3]) as metric:
        bundle = _fit(train, y, epochs=8, eval_frame=evaluation, eval_target=target)
    assert metric.call_count == 3
    assert bundle["epochs"] == 1 and bundle["trained_epochs"] == 3
    assert bundle["best_ap"] == .4
    assert [row["inner_ap"] for row in bundle["curve"]] == [.4, .4, .3]
    assert bundle["selected_on_inner_ap"] and not bundle["restored_best_weights"]
    # Even extreme evaluation inputs and unknown categories do not enter fit
    # normalization, dictionaries or PLE quantiles.
    expected = TabMPreprocessor(bins=4).fit(train, ["category"])
    assert bundle["preprocessor"].bin_edges_sha256 == expected.bin_edges_sha256
    np.testing.assert_array_equal(bundle["preprocessor"].base.scaler.mean_, expected.base.scaler.mean_)
    assert "unknown" not in bundle["preprocessor"].base.vocabs["category"]


def test_tabm_repeated_fresh_fit_joblib_and_state_reload_are_exact():
    train, y, evaluation, _ = _fixture()
    first = _fit(train, y)
    expected = predict_bundle(first, evaluation)
    second = _fit(train, y)
    np.testing.assert_array_equal(expected, predict_bundle(second, evaluation))
    for key, value in first["model"].state_dict().items():
        torch.testing.assert_close(value, second["model"].state_dict()[key], rtol=0, atol=0)
    stream = BytesIO()
    joblib.dump(first, stream)
    stream.seek(0)
    restored = joblib.load(stream)
    np.testing.assert_array_equal(expected, predict_bundle(restored, evaluation))
    clone = build_tabm(first["preprocessor"], **first["model_kwargs"])
    clone.load_state_dict(first["model"].state_dict())
    np.testing.assert_array_equal(expected, predict_bundle(dict(first, model=clone), evaluation))


def test_tabm_training_rejects_unpaired_evaluation_invalid_target_and_cuda_fallback():
    train, y, evaluation, _ = _fixture()
    for options in [dict(eval_frame=evaluation), dict(eval_target=np.zeros(4)),
                    dict(epochs=0), dict(batch_size=0), dict(patience=0)]:
        try:
            _fit(train, y, **options)
            raise AssertionError("Invalid training options accepted")
        except ValueError:
            pass
    for target in [np.ones(len(y) - 1), np.full(len(y), np.nan), np.full(len(y), .5)]:
        try:
            _fit(train, target)
            raise AssertionError("Invalid training target accepted")
        except ValueError:
            pass
    with patch("torch.cuda.is_available", return_value=False):
        try:
            _fit(train, y, device="cuda")
            raise AssertionError("CUDA silently fell back to CPU")
        except RuntimeError:
            pass
