"""No-outer-evaluation, epoch choice, finite/device and deterministic checks."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from hw4_inspections.residual_training import infer, seed_all, train_epochs


class TinyFiveInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 2)
        self.head = nn.Sequential(nn.Linear(7, 8), nn.ReLU(), nn.Dropout(.1), nn.Linear(8, 1))

    def forward(self, numeric, cats, basis, events, valid):
        event_sum = (events * valid.unsqueeze(-1)).sum(dim=1)
        hidden = torch.cat([numeric, self.embedding(cats[:, 0]), basis.flatten(1), event_sum], dim=1)
        return self.head(hidden).squeeze(-1)


def fixture(n=12):
    generator = torch.Generator().manual_seed(321)
    numeric = torch.randn(n, 2, generator=generator)
    data = (numeric, torch.arange(n).remainder(3).reshape(n, 1),
            torch.randn(n, 1, 1, generator=generator),
            torch.randn(n, 2, 2, generator=generator), torch.ones(n, 2, dtype=torch.bool))
    target = (torch.arange(n).remainder(3) == 0).to(torch.float32)
    return data, target


def raises_error(callback, error=ValueError):
    try:
        callback()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def test_residual_outer_training_never_scores_or_receives_evaluation():
    seed_all(2)
    model = TinyFiveInputNet()
    data, targets = fixture()
    before = deepcopy(model.state_dict())
    events = []
    with patch("hw4_inspections.residual_training.average_precision_score", side_effect=AssertionError("Outer AP called")):
        chosen, curve = train_epochs(model, data, targets, epochs=3, batch_size=5, seed=2,
                                     log=lambda event, **fields: events.append((event, fields)))
    assert chosen == len(curve) == 3
    assert [row["epoch"] for row in curve] == [1, 2, 3]
    assert all("inner_ap" not in row and np.isfinite(row["train_loss"]) for row in curve)
    assert any(not torch.equal(before[key], model.state_dict()[key]) for key in before)
    assert [event for event, _ in events].count("first_batch") == 1
    first = next(fields for event, fields in events if event == "first_batch")
    assert first["logits_device"] == first["loss_device"] == "cpu"
    assert first["cuda_allocated_mib"] == 0
    # Ordinary BCE remains defined for a one-class training batch and dataset.
    train_epochs(TinyFiveInputNet(), data, torch.zeros_like(targets), epochs=1, batch_size=1, seed=4)


def test_residual_inner_epoch_selection_earliest_maximum_and_patience():
    data, targets = fixture()
    model = TinyFiveInputNet()
    # Deliberate tied maximum ensures earliest selection and no resetting tie.
    with patch("hw4_inspections.residual_training.average_precision_score",
               side_effect=[.2, .4, .4, .3, .3]) as scorer:
        chosen, curve = train_epochs(model, data, targets, epochs=9, batch_size=6, seed=5,
                                     eval_data=data, eval_targets=targets, patience=2)
    assert chosen == 2 and len(curve) == scorer.call_count == 4
    assert chosen == max(curve, key=lambda row: (row["inner_ap"], -row["epoch"]))["epoch"]
    assert [row["inner_ap"] for row in curve] == [.2, .4, .4, .3]


def test_residual_seeded_training_inference_batching_and_no_rng_reset():
    data, targets = fixture()
    models = []
    for _ in range(2):
        seed_all(17)
        model = TinyFiveInputNet()
        train_epochs(model, data, targets, epochs=2, batch_size=4, seed=17)
        models.append(model)
    for key in models[0].state_dict():
        torch.testing.assert_close(models[0].state_dict()[key], models[1].state_dict()[key], rtol=0, atol=0)
    with patch("hw4_inspections.residual_training.seed_all", side_effect=AssertionError("Training reset RNG")):
        train_epochs(models[0], data, targets, epochs=1, batch_size=4, seed=17)
    expected = infer(models[0], data, batch_size=4)
    assert expected.shape == (len(targets),) and np.isfinite(expected).all()
    assert ((expected >= 0) & (expected <= 1)).all() and not models[0].training
    # One identical-size batch is a plain five-input forward under eval/no_grad.
    with torch.no_grad():
        actual = models[0](*(value[:4] for value in data)).sigmoid().numpy()
    np.testing.assert_array_equal(expected[:4], actual)
    reversed_p = infer(models[0], tuple(value.flip(0) for value in data), batch_size=4)
    np.testing.assert_allclose(expected[::-1], reversed_p, rtol=0, atol=1e-7)


def test_residual_rejects_incomplete_eval_bad_shapes_nonfinite_and_empty():
    data, target = fixture()
    model = TinyFiveInputNet()
    common = dict(epochs=1, batch_size=4, seed=1)
    raises_error(lambda: train_epochs(model, data, target, **common, eval_data=data))
    raises_error(lambda: train_epochs(model, data, target, **common, eval_targets=target))
    for bad in [target[:, None], target[0], target[:-1], target * 2,
                target.clone().fill_(float("nan"))]:
        raises_error(lambda bad=bad: train_epochs(model, data, bad, **common))
    raises_error(lambda: infer(model, tuple(value[:0] for value in data)))
    raises_error(lambda: infer(model, data, batch_size=0))
    bad = list(data)
    bad[0] = bad[0].clone().fill_(float("inf"))
    raises_error(lambda: infer(model, bad))
    raises_error(lambda: infer(model, data[:4]))
    bad = list(data)
    bad[-1] = bad[-1][:, :1]
    raises_error(lambda: infer(model, bad))
    bad = list(data)
    bad[1] = bad[1].float()
    raises_error(lambda: infer(model, bad))
    bad = list(data)
    bad[2] = bad[2].squeeze(-1)
    raises_error(lambda: infer(model, bad))


def test_residual_nonfinite_or_misshaped_logits_fail_explicitly():
    data, targets = fixture()

    class BadNet(TinyFiveInputNet):
        def forward(self, *data):
            return super().forward(*data) * float("nan")

    raises_error(lambda: infer(BadNet(), data), FloatingPointError)
    raises_error(lambda: train_epochs(BadNet(), data, targets, epochs=1, batch_size=4, seed=1),
                 FloatingPointError)

    class WrongShape(TinyFiveInputNet):
        def forward(self, *data):
            return super().forward(*data).unsqueeze(-1)

    raises_error(lambda: infer(WrongShape(), data))
