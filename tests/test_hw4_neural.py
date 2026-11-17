"""Causal sequence, training vocabulary and padding isolation checks."""
import numpy as np
import pandas as pd
import torch

from hw4_inspections.neural import InspectionNet, TabularTransform, build_sequences
from test_hw4_features import fixture


def test_sequence_snapshot_cutoff_target_and_order_invariance():
    rows, h = fixture()
    expected, mask, audit = build_sequences(rows, h)
    extra = pd.concat([h.assign(inspection_week="2024-09-23"),
                       h.assign(inspection_week="2024-09-30"),
                       h.assign(forecast_period="2025-01-01")])
    actual, actual_mask, _ = build_sequences(rows.assign(target=[1, 0]),
                                            pd.concat([extra, h]).iloc[::-1])
    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(mask, actual_mask)
    assert mask.sum(axis=1).tolist() == [2, 0]
    assert audit["truncated_rows"] == 0
    assert expected[0, 0, 0] == 1
    assert expected[0, 1, 0] == 0
    assert expected[0, 1, 1] == 1


def test_individual_cutoff_and_explicit_truncation():
    rows, h = fixture()
    rows = pd.concat([rows.iloc[:1], rows.iloc[:1]], ignore_index=True)
    rows.loc[1, "history_before"] = "2024-07-01"
    values, valid, audit = build_sequences(rows, h, length=1)
    assert valid.sum(axis=1).tolist() == [1, 1]
    assert audit["truncated_rows"] == 1
    assert values[0, 0, 1] == 1
    assert values[1, 0, 0] == 1


def test_empty_history_and_padding_cannot_change_prediction():
    rows, h = fixture()
    empty, mask, _ = build_sequences(rows, h.iloc[:0])
    model = InspectionNet(2, [3], empty.shape[-1], "visit_transformer", width=16).eval()
    numeric, cats = torch.zeros(2, 2), torch.zeros(2, 1, dtype=torch.int64)
    with torch.no_grad():
        p = model(numeric, cats, torch.as_tensor(empty), torch.as_tensor(mask))
        changed = model(numeric, cats, torch.randn(*empty.shape), torch.as_tensor(mask))
    assert torch.isfinite(p).all()
    torch.testing.assert_close(p, changed, rtol=0, atol=0)


def test_preprocessing_unknown_categories_and_training_only_statistics():
    fit = pd.DataFrame({"x": [0., 2., np.nan], "all_missing": [np.nan] * 3,
                        "category": ["a", "b", "a"]})
    transform = TabularTransform().fit(fit, ["category"])
    before = transform.scaler.mean_.copy()
    numeric, categorical = transform.transform(pd.DataFrame({"x": [1e20], "all_missing": [0.], "category": ["new"]}))
    assert categorical.item() == 0
    assert np.isfinite(numeric).all()
    np.testing.assert_array_equal(before, transform.scaler.mean_)
    assert "new" not in transform.vocabs["category"]


def test_feature_transformer_uses_numeric_and_categorical_tokens():
    torch.manual_seed(42)
    model = InspectionNet(4, [3, 2], 60, "feature_transformer", width=16).eval()
    x = torch.randn(3, 4)
    c = torch.zeros(3, 2, dtype=torch.int64)
    e, mask = torch.zeros(3, 16, 60), torch.zeros(3, 16, dtype=torch.bool)
    with torch.no_grad():
        a = model(x, c, e, mask)
        b = model(x + 1, c, e, mask)
    assert a.shape == (3,) and torch.isfinite(a).all()
    assert not torch.equal(a, b)
