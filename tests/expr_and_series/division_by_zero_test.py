from __future__ import annotations

import os

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (-2, 0, float("-inf")),
        (-2.0, 0.0, float("-inf")),
        (0, 0, None),
        (0.0, 0.0, None),
        (2, 0, float("inf")),
        (2.0, 0.0, float("inf")),
    ],
)
def test_series_truediv_by_zero(
    left: float, right: float, expected: float | None, constructor_eager: ConstructorEager
) -> None:
    data: dict[str, list[int | float]] = {"a": [left], "b": [right]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    truediv_result = df["a"] / df["b"]
    assert_equal_data({"a": truediv_result}, {"a": [expected]})


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(-2, 0, float("-inf")), (0, 0, None), (2, 0, float("inf"))],
)
def test_series_floordiv_int_by_zero(
    left: int,
    right: int,
    expected: float | None,
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
) -> None:
    if "pandas" in str(constructor_eager) and PANDAS_VERSION < (2,):
        pytest.skip(reason="converts floordiv by zero to 0")
    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    data: dict[str, list[int]] = {"a": [left], "b": [right]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    # pyarrow backend floordiv raises divide by zero error
    if "pyarrow" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    # polars backend floordiv by zero always returns null
    if "polars" in str(constructor_eager):
        floordiv_result = df["a"] // df["b"]
        assert all(floordiv_result.is_null())
    # pandas[nullable] backend floordiv always returns 0
    elif all(x in str(constructor_eager) for x in ["pandas", "nullable"]):
        floordiv_result = df["a"] // df["b"]
        assert_equal_data({"a": floordiv_result}, {"a": [0]})
    else:
        floordiv_result = df["a"] // df["b"]
        assert_equal_data({"a": floordiv_result}, {"a": [expected]})


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (-2, 0, float("-inf")),
        (-2.0, 0.0, float("-inf")),
        (0, 0, None),
        (0.0, 0.0, None),
        (2, 0, float("inf")),
        (2.0, 0.0, float("inf")),
    ],
)
def test_truediv_by_zero(
    left: float, right: float, expected: float | None, constructor: Constructor
) -> None:
    if "pyspark" in str(constructor):
        # https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.try_divide.html
        # PySpark always returns null when dividing by zero.
        expected = None
    data: dict[str, list[int | float]] = {"a": [left]}
    df = nw.from_native(constructor(data))
    truediv_result = df.select(nw.col("a") / right)
    assert_equal_data(truediv_result, {"a": [expected]})


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(-2, 0, float("-inf")), (0, 0, None), (2, 0, float("inf"))],
)
def test_floordiv_int_by_zero(
    left: int,
    right: int,
    expected: float | None,
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
    if "pandas" in str(constructor) and PANDAS_VERSION < (2,):
        pytest.skip(reason="converts floordiv by zero to 0")
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")

    if "polars_lazy" in str(constructor) and os.environ.get("NARWHALS_POLARS_GPU"):
        request.applymarker(
            pytest.mark.xfail(reason="https://github.com/pola-rs/polars/issues/23365")
        )

    data: dict[str, list[int]] = {"a": [left]}
    df = nw.from_native(constructor(data))
    # pyarrow backend floordiv raises divide by zero error
    # ibis backend floordiv cannot cast value to inf or -inf
    if any(x in str(constructor) for x in ["ibis", "pyarrow"]):
        request.applymarker(pytest.mark.xfail)
    # duckdb backend floordiv return None
    if any(x in str(constructor) for x in ("duckdb", "pyspark")):
        floordiv_result = df.select(nw.col("a") // right)
        assert_equal_data(floordiv_result, {"a": [None]})
    # polars backend floordiv returns null
    elif "polars" in str(constructor) and "lazy" not in str(constructor):
        floordiv_result = df.select(nw.col("a") // right)
        assert all(floordiv_result["a"].is_null())
    # polars lazy floordiv cannot be sliced and returns None
    elif all(x in str(constructor) for x in ["polars", "lazy"]):
        floordiv_result = df.select(nw.col("a") // right)
        assert_equal_data(floordiv_result, {"a": [None]})
    # pandas[nullable] backend floordiv always returns 0
    elif "pandas_nullable" in str(constructor):
        floordiv_result = df.select(nw.col("a") // right)
        assert_equal_data(floordiv_result, {"a": [0]})
    else:
        floordiv_result = df.select(nw.col("a") // right)
        assert_equal_data(floordiv_result, {"a": [expected]})
