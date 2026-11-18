"""Matched controls, trainable embeddings and padding-safe visit attention."""
from io import BytesIO

import torch

from hw4_inspections.residual_models import ResidualBlock, ResidualInspectionNet


def _inputs():
    generator = torch.Generator().manual_seed(17)
    numeric = torch.randn(3, 6, generator=generator)
    numeric[:, 3:] = torch.tensor([[0., 1., 0.], [0., 0., 0.], [1., 0., 1.]])
    categorical = torch.tensor([[0, 1], [2, 0], [1, 2]], dtype=torch.int64)
    basis = torch.rand(3, 3, 4, generator=generator)
    events = torch.zeros(3, 4, 60)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False],
                          [False, False, False, False]])
    events[0, 0, [0, 2, 44, 48]] = 1
    events[0, 1, [45, 49]] = 1  # Genuine visit without observed codes.
    events[1, 0, [3, 46, 50]] = 1
    events[..., -4:] = torch.rand(3, 4, 4, generator=generator)
    return numeric, categorical, basis, events, valid


def _model(kind):
    return ResidualInspectionNet(3, [4, 3], kind=kind, bins=4,
                                 width=16, blocks=2, dropout=0.,
                                 length=4, visit_width=32)


def test_residual_control_matches_state_and_zero_branch_is_exact_identity():
    torch.manual_seed(42)
    control = _model("mlp_control")
    torch.manual_seed(42)
    residual = _model("resnet")
    assert list(control.state_dict()) == list(residual.state_dict())
    for name, value in control.state_dict().items():
        torch.testing.assert_close(value, residual.state_dict()[name], rtol=0, atol=0)
    block = ResidualBlock(16, dropout=0.)
    for parameter in block.parameters():
        parameter.data.zero_()
    x = torch.randn(5, 16)
    torch.testing.assert_close(block(x, use_skip=True), x, rtol=0, atol=0)
    torch.testing.assert_close(block(x, use_skip=False), torch.zeros_like(x), rtol=0, atol=0)
    args = _inputs()
    assert control.representation(*args).shape == residual.representation(*args).shape == (3, 16)
    assert not torch.equal(control(*args), residual(*args))


def test_ple_and_categorical_embeddings_receive_gradient_and_numeric_basis_matters():
    torch.manual_seed(42)
    model = _model("ple_resnet")
    args = _inputs()
    model(*args).sum().backward()
    assert model.ple_weight.grad is not None and model.ple_weight.grad.abs().sum() > 0
    assert all(embedding.weight.grad is not None and embedding.weight.grad.abs().sum() > 0
               for embedding in model.embeddings)
    model.eval()
    changed = list(args)
    changed[2] = args[2].clone()
    changed[2][:, 0, 0] += 1
    with torch.no_grad():
        assert not torch.equal(model(*args), model(*changed))


def test_ple_tower_initialization_matches_hybrid_and_visit_embeddings_learn():
    torch.manual_seed(42)
    tabular = _model("ple_resnet")
    torch.manual_seed(42)
    hybrid = _model("ple_visit_transformer")
    hybrid_state = hybrid.state_dict()
    for name, value in tabular.state_dict().items():
        if not name.startswith("head."):
            torch.testing.assert_close(value, hybrid_state[name], rtol=0, atol=0)
    hybrid(*_inputs()).sum().backward()
    for parameter in [hybrid.code_embedding.weight, hybrid.blank_code_embedding,
                      hybrid.outcome_embedding.weight, hybrid.type_embedding.weight]:
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_visit_padding_nan_isolation_empty_rows_and_reload_exact():
    torch.manual_seed(42)
    args = _inputs()
    model = _model("ple_visit_transformer").eval()
    changed = list(args)
    changed[3] = args[3].clone()
    changed[3][~args[4]] = float("nan")
    with torch.no_grad():
        expected = model(*args)
        actual = model(*changed)
    assert expected.shape == (3,) and torch.isfinite(expected).all()
    assert model.representation(*args).shape == (3, 48)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stream = BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = _model("ple_visit_transformer").eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(*changed), expected, rtol=0, atol=0)
        # Invalid padding may contain arbitrary finite values as well as NaNs.
        changed[3][~args[4]] = 1e20
        torch.testing.assert_close(model(*changed), expected, rtol=0, atol=0)


def test_residual_constructor_and_tensor_shapes_are_validated():
    invalid = [dict(n_numeric=0), dict(cardinalities=[0]), dict(bins=0),
               dict(embedding_dim=0), dict(width=0), dict(blocks=0),
               dict(dropout=1), dict(length=0), dict(kind="unknown"),
               dict(kind="ple_visit_transformer", visit_width=30)]
    for changes in invalid:
        options = dict(n_numeric=3, cardinalities=[4, 3])
        options.update(changes)
        try:
            ResidualInspectionNet(**options)
            raise AssertionError("Invalid constructor dimensions accepted")
        except ValueError:
            pass
    args = list(_inputs())
    model = _model("ple_visit_transformer")
    for index, bad in [(0, torch.zeros(3, 3)), (1, torch.zeros(3, 2)),
                       (2, torch.zeros(3, 3, 3)), (3, torch.zeros(3, 4, 59)),
                       (4, torch.zeros(3, 4))]:
        changed = list(args)
        changed[index] = bad
        try:
            model(*changed)
            raise AssertionError("Invalid tensor dimensions accepted")
        except ValueError:
            pass
