"""Tiny training checks for the legacy Visit Transformer delivery helpers."""
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import torch

from hw4_inspections.delivery_neural import choose_visit_epochs, fit_visit, predict_visit
from hw4_inspections.neural import InspectionNet


def _fixture():
    rng = np.random.default_rng(42)
    train = pd.DataFrame({"x": rng.normal(size=24), "empty": [np.nan] * 24,
                          "category": ["a", "b"] * 12})
    valid = pd.DataFrame({"x": rng.normal(size=8), "empty": [np.nan] * 8,
                          "category": ["new", "a"] * 4})
    labels = np.tile([0, 1, 1, 0], 6).astype(np.float32)
    target = np.tile([0, 1], 4).astype(np.float32)
    events = np.zeros((32, 16, 60), dtype=np.float32)
    mask = np.zeros((32, 16), dtype=bool)
    mask[:, :2] = True
    mask[::3] = False
    events[:, :2, 0] = 1
    events[:, :2, 44] = 1
    events[:, :2, 48] = 1
    return train, labels, valid, target, events[:24], mask[:24], events[24:], mask[24:]


def test_delivery_visit_fixed_refit_has_no_selection_and_reload_is_exact():
    args = _fixture()
    x, y, valid, _, events, mask, ve, vm = args
    before = x.copy(deep=True)
    bundle = fit_visit(x, y, ["category"], events, mask, epochs=2, batch_size=8, threads=1)
    assert bundle["epochs"] == 2 and len(bundle["training_curve"]) == 2
    assert bundle["inner_curve"] == []
    assert not any("inner_ap" in row for row in bundle["training_curve"])
    assert isinstance(bundle["model"], InspectionNet)
    assert bundle["model_kwargs"]["kind"] == "visit_transformer"
    assert bundle["model_kwargs"]["width"] == 64
    p = predict_visit(bundle, valid, ve, vm)
    stream = BytesIO()
    joblib.dump(bundle, stream)
    stream.seek(0)
    restored = joblib.load(stream)
    np.testing.assert_array_equal(p, predict_visit(restored, valid, ve, vm))
    # State-only checkpoint plus preprocessing is the actual delivery format.
    model = InspectionNet(**bundle["model_kwargs"])
    model.load_state_dict(bundle["model"].state_dict())
    state_restored = dict(bundle, model=model)
    np.testing.assert_array_equal(p, predict_visit(state_restored, valid, ve, vm))
    pd.testing.assert_frame_equal(x, before)
    assert "new" not in bundle["preprocessing"].vocabs["category"]
    assert np.isfinite(p).all() and p.dtype == np.float32


def test_delivery_visit_inner_choice_and_fresh_refit_are_deterministic():
    x, y, valid, target, events, mask, ve, vm = _fixture()
    epochs, curve = choose_visit_epochs(x, y, valid, target, ["category"],
                                        events, mask, ve, vm, max_epochs=3,
                                        patience=2, batch_size=8, threads=1)
    assert epochs == max(curve, key=lambda row: (row["inner_ap"], -row["epoch"]))["epoch"]
    assert len(curve) <= 3
    first = fit_visit(x, y, ["category"], events, mask, epochs=epochs, batch_size=8, threads=1)
    second = fit_visit(x, y, ["category"], events, mask, epochs=epochs, batch_size=8, threads=1)
    np.testing.assert_array_equal(predict_visit(first, valid, ve, vm),
                                  predict_visit(second, valid, ve, vm))
    for key, value in first["model"].state_dict().items():
        torch.testing.assert_close(value, second["model"].state_dict()[key], rtol=0, atol=0)


def test_delivery_visit_padding_isolation_and_rejects_target_features():
    x, y, valid, _, events, mask, ve, vm = _fixture()
    bundle = fit_visit(x, y, ["category"], events, mask, epochs=1, batch_size=8, threads=1)
    expected = predict_visit(bundle, valid, ve, vm)
    dirty = ve.copy()
    dirty[~vm] = np.nan
    np.testing.assert_array_equal(expected, predict_visit(bundle, valid, dirty, vm))
    for bad in [x.assign(target=y), x.assign(row_id="r"), x.assign(site_id="s")]:
        try:
            fit_visit(bad, y, ["category"], events, mask, epochs=1, threads=1)
            raise AssertionError("Forbidden feature accepted")
        except ValueError:
            pass
    try:
        predict_visit(bundle, valid, ve[:, :15], vm[:, :15])
        raise AssertionError("Incorrect sequence length accepted")
    except ValueError:
        pass
