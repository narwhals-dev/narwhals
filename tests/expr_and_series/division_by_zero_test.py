from __future__ import annotations

from typing import Any, Callable

import pytest

import narwhals as nw
from tests.utils import (
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    zip_strict,
)

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
    assert (~result["int"].is_finite()).all()


@pytest.mark.parametrize("denominator", [0, nw.lit(0), nw.col("denominator")])
def test_expr_truediv_by_zero(
    constructor: Constructor, denominator: int | nw.Expr
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("int", "float") / denominator)
    expected = {"int": expected_truediv, "float": expected_truediv}
    assert_equal_data(result, expected)
    assert_equal_data(
        result.select((~nw.all().is_finite()).all()), {"int": [True], "float": [True]}
    )


@pytest.mark.parametrize("get_denominator", [lambda _: 0, lambda df: df["denominator"]])
def test_series_floordiv_by_zero(
    constructor_eager: ConstructorEager,
    get_denominator: Callable[[nw.DataFrame[Any]], int | nw.Series[Any]],
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")

    df = nw.from_native(constructor_eager(data), eager_only=True)
    denominator = get_denominator(df)
    result = {"int": df["int"] // denominator}
    expected = {"int": expected_floordiv}
    assert_equal_data(result, expected)
    assert result["int"].is_null().all()


@pytest.mark.parametrize("denominator", [0, nw.lit(0), nw.col("denominator")])
def test_expr_floordiv_by_zero(
    constructor: Constructor, denominator: int | nw.Expr
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("int") // denominator)
    expected = {"int": expected_floordiv}
    assert_equal_data(result, expected)
    assert_equal_data(result.select(nw.col("int").is_null().all()), {"int": [True]})


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
    result = {"literal": numerator / df["denominator"]}
    assert_equal_data(result, {"literal": [expected] * len(df)})
    assert (~result["literal"].is_finite()).all()


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
    result = df.select((numerator / nw.col("denominator")).alias("result"))
    assert_equal_data(result, {"result": [expected] * len(data["denominator"])})
    assert_equal_data(result.select((~nw.all().is_finite()).all()), {"result": [True]})


@pytest.mark.parametrize("numerator", data["int"])
def test_series_rfloordiv_by_zero(
    constructor_eager: ConstructorEager, numerator: float
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 10):
        pytest.skip(reason="bug")

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {"literal": numerator // df["denominator"]}
    assert_equal_data(result, {"literal": expected_floordiv})
    assert result["literal"].is_null().all()


@pytest.mark.parametrize("numerator", data["int"])
def test_expr_rfloordiv_by_zero(constructor: Constructor, numerator: float) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip(reason="bug")

    df = nw.from_native(constructor(data))
    result = df.select(numerator // nw.col("denominator"))
    expected = {"literal": expected_floordiv}
    assert_equal_data(result, expected)
    assert_equal_data(
        result.select(nw.col("literal").is_null().all()), {"literal": [True]}
    )
