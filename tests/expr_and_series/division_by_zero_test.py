from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

data: dict[str, list[float]] = {
    "int": [-2, 0, 2],
    "float": [-2.1, 0.0, 2.1],
    "denominator": [0, 0, 0],
}
expected_truediv: list[float] = [float("-inf"), float("nan"), float("inf")]
expected_floordiv: list[float | None] = [None, None, None]


@pytest.mark.parametrize("get_denominator", [lambda _: 0, lambda df: df["denominator"]])
def test_series_truediv_by_zero(
    constructor_eager: ConstructorEager,
    get_denominator: Callable[[nw.DataFrame[Any]], int | nw.Series[Any]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    denominator = get_denominator(df)
    result = {"int": df["int"] / denominator, "float": df["float"] / denominator}
    expected = {"int": expected_truediv, "float": expected_truediv}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("denominator", [0, nw.lit(0), nw.col("denominator")])
def test_expr_truediv_by_zero(
    constructor: Constructor, denominator: int | nw.Expr
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("int", "float") / denominator)
    expected = {"int": expected_truediv, "float": expected_truediv}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("get_denominator", [lambda _: 0, lambda df: df["denominator"]])
def test_series_floordiv_by_zero(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    get_denominator: Callable[[nw.DataFrame[Any]], int | nw.Series[Any]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    denominator = get_denominator(df)
    result = {"result": df["int"] // denominator}
    expected = {"result": expected_floordiv}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("denominator", [0, nw.lit(0), nw.col("denominator")])
def test_expr_floordiv_by_zero(
    constructor: Constructor, request: pytest.FixtureRequest, denominator: int | nw.Expr
) -> None:
    df = nw.from_native(constructor(data))

    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if "cudf" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    result = df.select(result=nw.col("int") // denominator)
    expected = {"result": expected_floordiv}
    assert_equal_data(result, expected)
    assert_equal_data(result.select(nw.col("result").is_null().all()), {"result": [True]})


@pytest.mark.parametrize(
    ("numerator", "expected"),
    list(
        zip(
            [*data["int"], *data["float"]],
            [*expected_truediv, *expected_truediv],
            strict=True,
        )
    ),
)
def test_series_rtruediv_by_zero(
    constructor_eager: ConstructorEager, numerator: float, expected: float
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {"result": numerator / df["denominator"]}
    assert_equal_data(result, {"result": [expected] * len(df)})


@pytest.mark.parametrize(
    ("numerator", "expected"),
    list(
        zip(
            [*data["int"], *data["float"]],
            [*expected_truediv, *expected_truediv],
            strict=True,
        )
    ),
)
def test_expr_rtruediv_by_zero(
    constructor: Constructor, numerator: float, expected: float
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(result=numerator / nw.col("denominator"))
    assert_equal_data(result, {"result": [expected] * len(data["denominator"])})
    assert_equal_data(result.select((~nw.all().is_finite()).all()), {"result": [True]})


@pytest.mark.parametrize("numerator", data["int"])
def test_series_rfloordiv_by_zero(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest, numerator: float
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = {"result": numerator // df["denominator"]}
    assert_equal_data(result, {"result": expected_floordiv})


@pytest.mark.parametrize("numerator", data["int"])
def test_expr_rfloordiv_by_zero(
    constructor: Constructor, request: pytest.FixtureRequest, numerator: float
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if "cudf" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))

    result = df.select(result=numerator // nw.col("denominator"))
    expected = {"result": expected_floordiv}
    assert_equal_data(result, expected)


def test_floordiv_by_mixed_divisor(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip(reason="broadcast requires `over`, which requires DuckDB 1.3.0")
    if "cudf" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    data = {"i": [0, 1, 2, 3], "a": [7, 8, 7, None], "b": [0, 2, None, 0]}
    df = nw.from_native(constructor(data))
    result = df.select(
        "i",
        floordiv=nw.col("a") // nw.col("b"),
        rfloordiv=7 // nw.col("b"),
        broadcast_dividend=nw.col("a").max() // nw.col("b"),
        scalar_zero=nw.col("a") // 0,
    ).sort("i")
    expected = {
        "i": [0, 1, 2, 3],
        "floordiv": [None, 4, None, None],
        "rfloordiv": [None, 3, None, None],
        "broadcast_dividend": [None, 4, None, None],
        "scalar_zero": [None, None, None, None],
    }
    assert_equal_data(result, expected)

    if df.implementation.is_pandas_like():
        # `Int64` and `int64[pyarrow]` are both `nw.Int64`, so compare native dtypes.
        input_dtype = cast("pd.DataFrame", nw.to_native(df))["a"].dtype
        native_result = cast("pd.DataFrame", nw.to_native(result))
        assert all(
            native_result[name].dtype == input_dtype for name in expected if name != "i"
        )


def test_floordiv_by_null_scalar(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip(reason="broadcast requires `over`, which requires DuckDB 1.3.0")

    df = nw.from_native(constructor({"a": [7, 8], "b": [1, 2]}))
    null_scalar = nw.when(nw.col("b") > 99).then(nw.col("b")).max()
    result = df.select(nw.col("a") // null_scalar)
    assert_equal_data(result, {"a": [None, None]})


def test_floordiv_by_zero_keeps_dtype(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if "cudf" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor({"a": [6.0, 7.0]}))
    a = nw.col("a").cast(nw.Float32)
    schema = df.select(by_zero=a // 0, by_two=a // 2).collect_schema()
    assert schema["by_zero"] == schema["by_two"]
