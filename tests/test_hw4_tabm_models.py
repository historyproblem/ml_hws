"""Official TabM integration, training-only bins and member loss checks."""
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import rtdl_num_embeddings
import tabm
import torch

from hw4_inspections.tabm_models import (
    TabMPreprocessor, build_tabm, loss_tabm, member_logits, predict_tabm,
)


def _fixture():
    frame = pd.DataFrame({
        "x": np.linspace(-2, 2, 20), "constant": [7.] * 20,
        "empty": [np.nan] * 20, "category": ["a", "b"] * 10,
    })
    frame.loc[3, "x"] = np.nan
    return frame


def test_tabm_preprocessing_is_fit_only_unknown_safe_and_preserves_frame():
    fit = _fixture()
    before = fit.copy(deep=True)
    prep = TabMPreprocessor(bins=4).fit(fit, ["category"])
    assert prep.n_numeric == 6 and len(prep.bins) == 6
    assert prep.all_missing_columns == ["empty"]
    saved = [edges.clone() for edges in prep.bins]
    saved_hash = prep.bin_edges_sha256
    median = prep.base.imputer.statistics_.copy()
    scaler = prep.base.scaler.mean_.copy()
    query = fit.iloc[:3].copy()
    query["x"] = [1e20, np.nan, -1e20]
    query["category"] = "unseen"
    numeric, categories = prep.transform(query)
    assert numeric.dtype == np.float32 and categories.dtype == np.int64
    assert np.isfinite(numeric).all() and np.all(categories == 0)
    assert np.max(np.abs(numeric[:, :3])) <= 8
    assert prep.bin_edges_sha256 == saved_hash
    np.testing.assert_array_equal(prep.base.imputer.statistics_, median)
    np.testing.assert_array_equal(prep.base.scaler.mean_, scaler)
    for a, b in zip(saved, prep.bins):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Constants and all-missing numeric values both standardize to zero; their
    # flags remain separate inputs (zero for a constant, one for missing).
    np.testing.assert_array_equal(prep.transform(fit)[0][:, 1:3], np.zeros((20, 2), dtype=np.float32))
    assert prep.transform(fit)[0][:, 5].min() == 1
    for j in prep.constant_indices:
        center = prep.transform(fit)[0][0, j]
        torch.testing.assert_close(prep.bins[j], torch.tensor([center - 1, center + 1]), rtol=0, atol=0)
    pd.testing.assert_frame_equal(fit, before)
    stream = BytesIO()
    joblib.dump(prep, stream)
    stream.seek(0)
    restored = joblib.load(stream)
    for a, b in zip(prep.transform(query), restored.transform(query)):
        np.testing.assert_array_equal(a, b)


def test_tabm_varying_bins_match_official_quantiles_and_small_fits_are_safe():
    fit = _fixture()
    prep = TabMPreprocessor(bins=4).fit(fit, ["category"])
    numeric, _ = prep.transform(fit)
    variable = [j for j in range(prep.n_numeric) if j not in prep.constant_indices]
    expected = rtdl_num_embeddings.compute_bins(torch.as_tensor(numeric[:, variable]), n_bins=4)
    for j, edges in zip(variable, expected):
        torch.testing.assert_close(prep.bins[j], edges, rtol=0, atol=0)
    for n in [1, 2]:
        small = TabMPreprocessor(bins=4).fit(fit.iloc[:n], ["category"])
        for edges in small.bins:
            assert len(edges) >= 2 and torch.isfinite(edges).all() and (edges.diff() > 0).all()
        model = build_tabm(small, k=1, width=16, blocks=2, dropout=0, bins=4)
        assert np.isfinite(predict_tabm(model, small.transform(fit.iloc[:n]))).all()


def test_tabm_is_official_k1_k32_shapes_reload_exact_and_member_diversity():
    torch.set_num_threads(1)
    fit = _fixture()
    prep = TabMPreprocessor(bins=4).fit(fit, ["category"])
    data = prep.transform(fit)
    target = torch.tensor([0., 1.] * 10)
    for k in [1, 32]:
        torch.manual_seed(42)
        model = build_tabm(prep, k=k, width=16, blocks=2, dropout=0, bins=4)
        assert type(model) is tabm.TabM
        assert type(model.num_module) is rtdl_num_embeddings.PiecewiseLinearEmbeddings
        numeric, cats = [torch.as_tensor(a) for a in data]
        logits = member_logits(model, numeric, cats)
        assert logits.shape == (20, k) and logits.dtype == torch.float32
        if k == 32:
            assert not torch.equal(logits[:, 0], logits[:, 1])
        loss_tabm(logits, target).backward()
        assert model.output.weight.grad.abs().sum() > 0
        assert all(model.output.weight.grad[j].abs().sum() > 0 for j in range(k))
        assert model.num_module.linear0.weight.grad.abs().sum() > 0
        if k == 32:
            assert not torch.equal(model.output.weight.grad[0], model.output.weight.grad[1])
        expected = predict_tabm(model, data, batch_size=8)
        restored = build_tabm(prep, **model.hw4_model_kwargs)
        restored.load_state_dict(model.state_dict())
        np.testing.assert_array_equal(expected, predict_tabm(restored, data, batch_size=8))
        model.eval()
        with torch.no_grad():
            direct = member_logits(model, numeric[:8], cats[:8]).sigmoid().mean(1).numpy()
        np.testing.assert_array_equal(expected[:8], direct)
        assert model.hw4_official_kwargs["num_embeddings"]["version"] == "B"
        assert model.hw4_official_kwargs["n_blocks"] == 2


def test_tabm_loss_is_mean_independent_bce_not_loss_of_mean_prediction():
    logits = torch.tensor([[-5., 5.], [2., 4.]], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([1., 0.])
    actual = loss_tabm(logits, target)
    expected = torch.stack([
        torch.nn.functional.binary_cross_entropy_with_logits(logits[:, j], target)
        for j in range(2)
    ]).mean()
    wrong = torch.nn.functional.binary_cross_entropy_with_logits(logits.mean(1), target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert abs(actual.item() - wrong.item()) > .1
    actual.backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad != 0).all()


def test_tabm_rejects_target_schema_invalid_bins_and_numeric_dtype():
    fit = _fixture()
    for column in ["target", "row_id", "site_id", "forecast_period", "history_before"]:
        try:
            TabMPreprocessor(bins=4).fit(fit.assign(**{column: 0}), ["category"])
            raise AssertionError("Forbidden feature accepted")
        except ValueError:
            pass
    prep = TabMPreprocessor(bins=4).fit(fit, ["category"])
    try:
        build_tabm(prep, bins=16)
        raise AssertionError("Bins mismatching fitted preprocessing accepted")
    except ValueError:
        pass
    model = build_tabm(prep, k=1, width=16, blocks=2, bins=4)
    numeric, cats = [torch.as_tensor(a) for a in prep.transform(fit)]
    for bad_numeric, bad_cats in [(numeric.double(), cats), (numeric, cats.float()),
                                   (numeric[:, :1], cats), (numeric, cats[:2])]:
        try:
            member_logits(model, bad_numeric, bad_cats)
            raise AssertionError("Invalid member tensor accepted")
        except ValueError:
            pass
