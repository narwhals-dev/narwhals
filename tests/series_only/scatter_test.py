from __future__ import annotations

import pandas as pd

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_scatter(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [142, 124, 132]}), eager_only=True
    )
    result = df.with_columns(
        df["a"].scatter([0, 1], [999, 888]), df["b"].scatter([0, 2, 1], df["b"])
    )
    expected = {"a": [999, 888, 3], "b": [142, 132, 124]}
    assert_equal_data(result, expected)


def test_scatter_indices() -> None:
    s = nw.from_native(pd.Series([2, 3, 6], index=[1, 0, 2]), series_only=True)
    result = s.scatter([1, 0, 2], s)
    expected = pd.Series([3, 2, 6], index=[1, 0, 2])
    pd.testing.assert_series_equal(result.to_native(), expected)


def test_scatter_unchanged(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [142, 124, 132]}), eager_only=True
    )
    df.with_columns(
        df["a"].scatter([0, 1], [999, 888]), df["b"].scatter([0, 2, 1], [142, 124, 132])
    )
    expected = {"a": [1, 2, 3], "b": [142, 124, 132]}
    assert_equal_data(df, expected)


def test_single_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [142, 124, 132]}), eager_only=True
    )
    s = df["a"]
    s.scatter([0, 1], [999, 888])
    expected = {"a": [1, 2, 3]}
    assert_equal_data({"a": s}, expected)


def test_scatter_integer(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [142, 124, 132]}), eager_only=True
    )
    s = df["a"]
    result = s.scatter(0, 999)
    expected = {"a": [999, 2, 3]}
    assert_equal_data({"a": result}, expected)


def test_scatter_unordered_indices(constructor_eager: ConstructorEager) -> None:
    data = {"a": [16, 12, 10, 9, 6, 5, 2]}
    indices = [6, 1, 0, 5, 3, 2, 4]
    df = nw.from_native(constructor_eager(data))
    result = df["a"].scatter(indices, df["a"])
    assert_equal_data({"a": result}, {"a": [10, 12, 5, 6, 2, 9, 16]})
