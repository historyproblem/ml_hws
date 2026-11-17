"""Mask isolation, label-free pretraining and train-only preprocessing checks."""
import copy
import inspect

import numpy as np
import pandas as pd
import torch

from hw4_inspections.masked import GroupMaskedNet, MaskedTransform, group_schema
from hw4_inspections.masked_training import pretrain, seed_all


def fixture():
    frame = pd.DataFrame({"forecast_year": [2022., 2022., 2023., 2023.],
                          "history_n_visits": [1., 2., 3., 4.],
                          "history_violation_mean": [np.nan, 1., 3., 4.],
                          "code01_count": [0., 1., 2., 3.],
                          "history_fail_rate": [0., 0., 1., 1.],
                          "last_routine_age_days": [np.nan] * 4,
                          "facility_type": ["a", "a", "b", "b"],
                          "lag1_result": ["pass", "pass", "fail", "fail"]})
    prep = MaskedTransform().fit(frame, ["facility_type", "lag1_result"])
    data = tuple(torch.as_tensor(a) for a in prep.transform(frame))
    seed_all(42)
    model = GroupMaskedNet(len(prep.numeric), prep.cardinalities, prep.groups, width=16, layers=1, dropout=0)
    return frame, prep, data, model


def test_group_closure_and_masked_values_cannot_affect_embedding():
    _, prep, data, model = fixture()
    names = [g["name"] for g in prep.groups]
    mask = torch.zeros(len(data[0]), len(names), dtype=torch.bool)
    mask[:, names.index("history_volume")] = True
    mask = model.expanded_mask(mask)
    for group in ["history_volume", "history_codes_burden", "history_outcomes"]:
        assert mask[:, names.index(group)].all()
    changed = [a.clone() for a in data]
    for group in prep.groups:
        if group["name"] in {"history_volume", "history_codes_burden", "history_outcomes"}:
            idx = group["numeric"]
            changed[0][:, idx] += 10
            changed[0][:, [i + len(prep.numeric) for i in idx]] = 1
            for j in group["categorical"]:
                changed[1][:, j] = 0
    model.eval()
    with torch.no_grad():
        expected = model.encode(*data, mask)[0]
        actual = model.encode(*changed, mask)[0]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_rank_transform_finite_unknown_and_target_rejected():
    frame, prep, _, _ = fixture()
    before = prep.quantile.quantiles_.copy()
    changed = frame.copy()
    changed["facility_type"] = "unknown"
    changed["code01_count"] = np.inf
    values, categories = prep.transform(changed)
    assert np.isfinite(values).all() and (categories[:, 0] == 0).all()
    np.testing.assert_array_equal(before, prep.quantile.quantiles_)
    try:
        MaskedTransform().fit(frame.assign(target=1), ["facility_type", "lag1_result"])
    except ValueError:
        pass
    else:
        raise AssertionError("Target must not enter masked pretraining")


def test_reconstruction_has_gradients_and_uses_no_target_argument():
    _, _, data, model = fixture()
    assert "target" not in inspect.signature(pretrain).parameters
    mask = model.sample_mask(len(data[0]), .35, data[0].device)
    assert mask.any(dim=1).all() and (~mask).any(dim=1).all()
    loss = model.pretraining_loss(*data, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.bottleneck.weight.grad is not None
    assert model.head[1].weight.grad is None


def test_frozen_unsupervised_encoder_independent_of_supervised_head():
    _, _, data, model = fixture()
    original_head = copy.deepcopy(model.head.state_dict())
    pretrain(model, data, epochs=1, batch_size=4, mask_probability=.35, seed=42, log=lambda *a, **kw: None)
    for key, value in model.head.state_dict().items():
        torch.testing.assert_close(value, original_head[key], rtol=0, atol=0)
    frozen = copy.deepcopy(model.state_dict())
    model.eval()
    with torch.no_grad():
        expected = model.encode(*data)[0].clone()
        model.bottleneck.weight.add_(1)
    restored = copy.deepcopy(model)
    restored.load_state_dict(frozen)
    with torch.no_grad():
        torch.testing.assert_close(expected, restored.encode(*data)[0], rtol=0, atol=0)


def test_all_related_columns_and_missing_flags_share_one_token():
    groups = group_schema(["history_violation_total", "code01_count", "code01_ever", "history_fail_rate"], ["lag1_result"])
    by_name = {g["name"]: g for g in groups}
    assert by_name["history_codes_burden"]["numeric"] == [0, 1, 2]
    assert by_name["history_outcomes"]["numeric"] == [3]
    assert by_name["history_outcomes"]["categorical"] == [0]
