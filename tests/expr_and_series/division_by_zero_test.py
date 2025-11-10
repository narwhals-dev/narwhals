from __future__ import annotations

from typing import Any, Callable

import pytest

import narwhals as nw
from narwhals._utils import zip_strict
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

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
    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if any(x in str(constructor_eager) for x in ("pandas", "modin", "cudf")):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    denominator = get_denominator(df)
    result = {"result": df["int"] // denominator}
    expected = {"result": expected_floordiv}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("denominator", [0, nw.lit(0), nw.col("denominator")])
def test_expr_floordiv_by_zero(
    constructor: Constructor, request: pytest.FixtureRequest, denominator: int | nw.Expr
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if any(x in str(constructor) for x in ("pandas", "modin", "cudf")):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))

    result = df.select(result=nw.col("int") // denominator)
    expected = {"result": expected_floordiv}
    assert_equal_data(result, expected)
    assert_equal_data(result.select(nw.col("result").is_null().all()), {"result": [True]})


@pytest.mark.parametrize(
    ("numerator", "expected"),
    list(
        zip_strict([*data["int"], *data["float"]], [*expected_truediv, *expected_truediv])
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
        zip_strict([*data["int"], *data["float"]], [*expected_truediv, *expected_truediv])
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
    if any(x in str(constructor_eager) for x in ("pandas_pyarrow", "modin_pyarrow")) or (
        any(
            x in str(constructor_eager)
            for x in ("pandas_nullable", "pandas_constructor", "modin_constructor")
        )
        and numerator != 0
    ):
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

    if any(x in str(constructor) for x in ("pandas_pyarrow", "modin_pyarrow")) or (
        any(
            x in str(constructor)
            for x in ("pandas_nullable", "pandas_constructor", "modin_constructor")
        )
        and numerator != 0
    ):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(result=numerator // nw.col("denominator"))
    expected = {"result": expected_floordiv}
    assert_equal_data(result, expected)
