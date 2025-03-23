from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import POLARS_VERSION
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "a": [1, 1, 2, 3, 2],
    "b": [1, 2, 3, 2, 1],
}


def test_is_last_distinct_expr(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.all().is_last_distinct())
    expected = {
        "a": [False, True, False, True, True],
        "b": [False, False, True, True, True],
    }
    assert_equal_data(result, expected)


def test_is_last_distinct_expr_all(constructor_eager: ConstructorEager) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/2268
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 9):
        pytest.skip(reason="too old version")
    data = {"a": [1, 1, 2, 3, 2], "b": [1, 2, 3, 2, 1], "i": [0, 1, 2, 3, 4]}
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.all().is_last_distinct().over(_order_by="i"))
    expected = {
        "a": [False, True, False, True, True],
        "b": [False, False, True, True, True],
        "i": [True, True, True, True, True],
    }
    assert_equal_data(result, expected)


def test_is_last_distinct_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.is_last_distinct()
    expected = {
        "a": [False, True, False, True, True],
    }
    assert_equal_data({"a": result}, expected)
