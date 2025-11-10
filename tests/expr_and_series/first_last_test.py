from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    assert_equal_data,
)

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


def test_first_expr_over_order_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if any(x in str(constructor) for x in ("pyspark", "dask")):
        # Currently unsupported.
        request.applymarker(pytest.mark.xfail)
    if "ibis" in str(constructor):
        # https://github.com/ibis-project/ibis/issues/11656
        request.applymarker(pytest.mark.xfail)
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    frame = nw.from_native(
        constructor(
            {
                "a": [1, 1, 2],
                "b": [4, 5, 6],
                "c": [None, 7, 8],
                "d": ["x", "y", "z"],
                "i": [None, 2, 1],
            }
        )
    )
    result = frame.with_columns(
        nw.col("b", "c", "d").first().over(order_by="i").name.suffix("_first"),
        nw.col("b", "c", "d").last().over(order_by="i").name.suffix("_last"),
    ).sort("i")
    expected = {
        "a": [1, 2, 1],
        "b": [4, 6, 5],
        "c": [None, 8, 7],
        "d": ["x", "z", "y"],
        "i": [None, 1, 2],
        "b_first": [4, 4, 4],
        "c_first": [None, None, None],
        "d_first": ["x", "x", "x"],
        "b_last": [5, 5, 5],
        "c_last": [7, 7, 7],
        "d_last": ["y", "y", "y"],
    }
    assert_equal_data(result, expected)


def test_first_expr_over_order_by_partition_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if any(x in str(constructor) for x in ("pyspark", "dask")):
        # Currently unsupported.
        request.applymarker(pytest.mark.xfail)
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (14,):
        pytest.skip()
    if "ibis" in str(constructor):
        # https://github.com/ibis-project/ibis/issues/11656
        request.applymarker(pytest.mark.xfail)
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    frame = nw.from_native(
        constructor(
            {"a": [1, 1, 2], "b": [4, 5, 6], "c": [None, 7, 8], "i": [1, None, 2]}
        )
    )
    result = frame.with_columns(
        nw.col("b", "c").first().over("a", order_by="i").name.suffix("_first"),
        nw.col("b", "c").last().over("a", order_by="i").name.suffix("_last"),
    ).sort("i")
    expected = {
        "a": [1, 1, 2],
        "b": [5, 4, 6],
        "c": [7.0, None, 8.0],
        "i": [None, 1, 2],
        "b_first": [5, 5, 6],
        "c_first": [7.0, 7.0, 8.0],
        "b_last": [4, 4, 6],
        "c_last": [None, None, 8.0],
    }
    assert_equal_data(result, expected)
