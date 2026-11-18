"""Small training-only checks for numeric piecewise residual inputs."""
import pickle

import numpy as np
import pandas as pd
import pandas.testing as pdt

from hw4_inspections.neural import TabularTransform
from hw4_inspections.residual_preprocessing import ResidualTransform


def assert_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_residual_basis_known_quantiles_scaling_and_extreme_clipping():
    train = pd.DataFrame({"x": [0., 1., 2., 3., 4.], "kind": ["a", "a", "b", "a", "b"]})
    transform = ResidualTransform(bins=2).fit(train, ["kind"])
    np.testing.assert_array_equal(transform.knots[0], [0, 2, 4])
    valid = pd.DataFrame({"x": [-100., 0., 1., 2., 3., 4., 100., np.nan], "kind": ["a"] * 8})
    numeric, categories, basis = transform.transform(valid)
    expected = [[0, 0], [0, 0], [.5, 0], [1, 0], [1, .5], [1, 1], [1, 1], [0, 0]]
    np.testing.assert_array_equal(basis[:, 0], expected)
    assert numeric.shape == (8, 2) and numeric.dtype == np.float32
    assert categories.shape == (8, 1) and categories.dtype == np.int64
    assert basis.shape == (8, 1, 2) and basis.dtype == np.float32
    np.testing.assert_allclose(numeric[1, 0], -2 / np.sqrt(2), atol=1e-7)
    assert numeric[0, 0] == -8 and numeric[-2, 0] == 8
    np.testing.assert_array_equal(numeric[:, 1], [0, 0, 0, 0, 0, 0, 0, 1])
    assert numeric[-1, 0] == 0
    assert transform.numeric_names == ["x"] and transform.cardinalities == [3]
    assert isinstance(transform.base, TabularTransform)


def test_residual_ties_constants_all_missing_and_nonfinite_semantics():
    train = pd.DataFrame({
        "ties": [0., 0., 0., 1.], "constant": [2.] * 4,
        "empty": [np.nan, np.inf, -np.inf, np.nan], "kind": ["a"] * 4,
    })
    before = train.copy(deep=True)
    transform = ResidualTransform(bins=4).fit(train, ["kind"])
    numeric, categories, basis = transform.transform(train)
    pdt.assert_frame_equal(train, before)
    np.testing.assert_array_equal(transform.knots[0], [0, .25, 1])
    np.testing.assert_array_equal(transform.knots[1], [2])
    assert transform.knots[2].size == 0
    assert np.isfinite(numeric).all() and np.isfinite(basis).all()
    assert basis[:, 0, 2:].sum() == 0
    assert basis[:, 1:].sum() == 0
    np.testing.assert_array_equal(basis[-1, 0], [1, 1, 0, 0])
    # Numeric order is scaled ties/constant/empty, then their missing flags.
    np.testing.assert_array_equal(numeric[:, 5], np.ones(4))
    assert categories.shape == (4, 1)
    valid = train.copy()
    valid.loc[:, "ties"] = np.nan
    valid.loc[:, "constant"] = np.inf
    n, _, b = transform.transform(valid)
    assert b.sum() == 0
    np.testing.assert_array_equal(n[:, 3:], np.ones((4, 3)))


def test_residual_transform_is_training_only_unknown_categories_and_repeatable():
    train = pd.DataFrame({"x": [0., 1., 2., np.nan], "kind": ["a", "b", None, "a"]})
    transform = ResidualTransform(bins=4).fit(train, ["kind"])
    saved = pickle.dumps(transform)
    valid = pd.DataFrame({"x": [1e9, -1e9, np.nan, np.inf],
                          "kind": ["unseen", "a", None, "unseen"]}, index=[4, 1, 4, 9])
    n, c, b = transform.transform(valid)
    assert pickle.dumps(transform) == saved
    assert c[0, 0] == 0 and c[3, 0] == 0
    assert c[1, 0] > 0 and c[2, 0] > 0
    assert b[2:].sum() == 0
    np.testing.assert_array_equal(n[:, 1], [0, 0, 1, 1])
    ordinary = TabularTransform().fit(train.copy(), ["kind"]).transform(valid.copy())
    np.testing.assert_array_equal(n, ordinary[0])
    np.testing.assert_array_equal(c, ordinary[1])
    loaded = pickle.loads(saved)
    for actual, expected in zip(loaded.transform(valid), (n, c, b)):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(transform.transform(valid.iloc[::-1]), (n[::-1], c[::-1], b[::-1])):
        np.testing.assert_array_equal(actual, expected)


def test_residual_transform_schema_bins_and_raw_column_guards():
    train = pd.DataFrame({"x": [0., 1.], "kind": ["a", "b"]})
    for bins in [0, 65, 1.5, True, "16", None]:
        assert_value_error(lambda bins=bins: ResidualTransform(bins=bins))
    for bins in [1, 64, np.int64(3)]:
        model = ResidualTransform(bins=bins).fit(train, ["kind"])
        assert model.transform(train)[2].shape == (2, 1, bins)
    for name in ["target", "site_id", "row_id", "forecast_period", "history_before"]:
        assert_value_error(lambda name=name: ResidualTransform().fit(train.assign(**{name: 0}), ["kind"]))
    model = ResidualTransform().fit(train, ["kind"])
    assert_value_error(lambda: model.transform(train[["kind", "x"]]))
    assert_value_error(lambda: model.transform(train.assign(extra=0)))
    assert_value_error(lambda: model.transform(train.drop(columns="x")))
    assert_value_error(lambda: ResidualTransform().transform(train))
    assert_value_error(lambda: ResidualTransform().fit(train, ["unknown"]))
    assert_value_error(lambda: ResidualTransform().fit(train, ["kind", "kind"]))
    assert_value_error(lambda: ResidualTransform().fit(train.iloc[:0], ["kind"]))
    assert_value_error(lambda: ResidualTransform().fit(train[["kind"]], ["kind"]))
    duplicate = pd.concat([train, train[["x"]]], axis=1)
    assert_value_error(lambda: ResidualTransform().fit(duplicate, ["kind"]))


def test_residual_numeric_only_empty_rows_and_no_input_mutation():
    train = pd.DataFrame({"x": [0., np.inf, 2., np.nan]})
    before = train.copy(deep=True)
    transform = ResidualTransform(bins=2).fit(train, [])
    n, c, b = transform.transform(train)
    pdt.assert_frame_equal(train, before)
    assert n.shape == (4, 2) and c.shape == (4, 0) and c.dtype == np.int64
    assert b.shape == (4, 1, 2) and b[[1, 3]].sum() == 0
    n, c, b = transform.transform(train.iloc[:0])
    assert n.shape == (0, 2) and c.shape == (0, 0) and b.shape == (0, 1, 2)
