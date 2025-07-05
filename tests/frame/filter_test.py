from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_filter(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    result = df.filter(nw.col("a") > 1)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    assert_equal_data(result, expected)


def test_filter_with_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.filter(df["a"] > 1)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    assert_equal_data(result, expected)


def test_filter_with_boolean_list(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    context = (
        pytest.raises(TypeError, match="not supported")
        if isinstance(df, nw.LazyFrame)
        else does_not_raise()
    )
    with context:
        result = df.filter([False, True, True])
        expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
        assert_equal_data(result, expected)


def test_filter_raise_on_agg_predicate(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    with pytest.raises(InvalidOperationError):
        df.filter(nw.col("a").max() > 2).lazy().collect()


def test_filter_raise_on_shape_mismatch(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    with pytest.raises(InvalidOperationError):
        df.filter(nw.col("b").unique() > 2).lazy().collect()


def test_filter_with_constrains(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6]}
    df = nw.from_native(constructor(data))
    result_scalar = df.filter(a=3)
    expected_scalar = {"a": [3], "b": [4]}

    assert_equal_data(result_scalar, expected_scalar)

    result_expr = df.filter(a=nw.col("b") // 3)
    expected_expr = {"a": [1, 2], "b": [4, 6]}

    assert_equal_data(result_expr, expected_expr)
