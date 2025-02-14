from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import POLARS_VERSION
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "a": [1, 1, 2, 2, 3],
    "b": [1, 2, 3, 3, 4],
}


def test_mode_single_expr(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").mode()).sort("a")
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


def test_mode_multi_expr(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION >= (1, 7, 0):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a", "b").mode()).sort("a", "b")
    expected = {"a": [1, 2], "b": [3, 3]}
    assert_equal_data(result, expected)


def test_mode_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.mode().sort()
    expected = {"a": [1, 2]}
    assert_equal_data({"a": result}, expected)
