from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, POLARS_VERSION, Constructor, assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import PythonLiteral
    from tests.utils import ConstructorEager

data: dict[str, list[PythonLiteral]] = {
    "a": [8, 2, 1, None],
    "b": [58, 5, 6, 12],
    "c": [2.5, 1.0, 3.0, 0.9],
    "d": [2, 1, 4, 3],
    "idx": [0, 1, 2, 3],
}

single_cases = pytest.mark.parametrize(
    ("col", "expected_first", "expected_last"),
    [("a", 8, None), ("b", 58, 12), ("c", 2.5, 0.9)],
)


@single_cases
def test_first_last_series(
    constructor_eager: ConstructorEager,
    col: str,
    expected_first: PythonLiteral,
    expected_last: PythonLiteral,
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.first()
    assert_equal_data({col: [result]}, {col: [expected_first]})
    result = series.last()
    assert_equal_data({col: [result]}, {col: [expected_last]})


def test_first_last_series_empty(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    series = series.filter(series > 50)
    result = series.first()
    assert result is None
    result = series.last()
    assert result is None


@single_cases
def test_first_last_expr_select(
    constructor_eager: ConstructorEager,
    col: str,
    expected_first: PythonLiteral,
    expected_last: PythonLiteral,
) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(
        nw.col(col).first().name.suffix("_first"), nw.col(col).last().name.suffix("_last")
    )
    assert_equal_data(
        result, {f"{col}_first": [expected_first], f"{col}_last": [expected_last]}
    )


@single_cases
def test_first_expr_with_columns(
    constructor_eager: ConstructorEager,
    col: str,
    expected_first: PythonLiteral,
    expected_last: PythonLiteral,
) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.with_columns(
        nw.col(col).first().name.suffix("_first"), nw.col(col).last().name.suffix("_last")
    ).select(nw.selectors.matches("first|last"))
    assert_equal_data(
        result,
        {
            f"{col}_first": [expected_first] * len(data["a"]),
            f"{col}_last": [expected_last] * len(data["a"]),
        },
    )


def test_first_expr_over_order_by(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    frame = nw.from_native(
        constructor({"a": [1, 1, 2], "b": [4, 5, 6], "c": [None, 7, 8], "i": [0, 2, 1]})
    )
    result = frame.with_columns(nw.col("b", "c").first().over(order_by="i")).sort("i")
    expected = {"a": [1, 2, 1], "b": [4, 4, 4], "c": [None, None, None], "i": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_first_expr_over_order_by_partition_by(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if "pandas" in str(constructor) and PANDAS_VERSION < (2, 2, 1):
        pytest.skip()
    frame = nw.from_native(
        constructor({"a": [1, 1, 2], "b": [4, 5, 6], "c": [None, 7, 8], "i": [0, 1, 2]})
    )
    result = frame.with_columns(nw.col("b", "c").first().over("a", order_by="i")).sort(
        "i"
    )
    expected = {"a": [1, 1, 2], "b": [4, 4, 6], "c": [None, None, 8], "i": [0, 1, 2]}
    assert_equal_data(result, expected)
