from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import ShapeError
from tests.utils import POLARS_VERSION, ConstructorEager, assert_equal_data

data = {"a": [1, 1, 2, 2, 3], "b": [1, 2, 3, 3, 4]}


def test_mode_single_expr(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").mode()).sort("a")
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


def test_mode_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.mode().sort()
    expected = {"a": [1, 2]}
    assert_equal_data({"a": result}, expected)


def test_mode_different_lengths(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 10):
        pytest.skip()
    df = nw.from_native(constructor_eager({"a": [1, 1, 2], "b": [4, 5, 6]}))
    with pytest.raises(ShapeError):
        df.select(nw.col("a", "b").mode())
