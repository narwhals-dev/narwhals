from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
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


def test_first_last_expr_over_order_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()

    if any(x in str(constructor) for x in ("pyspark", "dask")):
        # Currently unsupported.
        request.applymarker(pytest.mark.xfail)
    if "ibis" in str(constructor):
        # https://github.com/ibis-project/ibis/issues/11656
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor):
        reason = "Need to pass dtype when passing pd.NA or None"
        request.applymarker(pytest.mark.xfail(reason=reason))

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
        nw.col("b", "c", "d")
        .first()
        .over(order_by="i")
        .name.suffix("_first_ordered_over"),
        nw.col("b", "c", "d").last().over(order_by="i").name.suffix("_last_ordered_over"),
        nw.col("b", "c", "d").first(order_by="i").name.suffix("_first"),
        nw.col("b", "c", "d").last(order_by="i").name.suffix("_last"),
    ).sort("i")
    expected = {
        "a": [1, 2, 1],
        "b": [4, 6, 5],
        "c": [None, 8, 7],
        "d": ["x", "z", "y"],
        "i": [None, 1, 2],
        "b_first_ordered_over": [4, 4, 4],
        "c_first_ordered_over": [None, None, None],
        "d_first_ordered_over": ["x", "x", "x"],
        "b_last_ordered_over": [5, 5, 5],
        "c_last_ordered_over": [7, 7, 7],
        "d_last_ordered_over": ["y", "y", "y"],
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
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (14,):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()

    if any(x in str(constructor) for x in ("pyspark", "dask")):
        # Currently unsupported.
        request.applymarker(pytest.mark.xfail)
    if "ibis" in str(constructor):
        # https://github.com/ibis-project/ibis/issues/11656
        request.applymarker(pytest.mark.xfail)

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


def test_first_expr_in_group_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("spark", "dask")):
        # ibis: https://github.com/ibis-project/ibis/issues/11656
        request.applymarker(pytest.mark.xfail)
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (14,):
        pytest.skip()
    data = {
        "grp": [1, 1, 1, 2],
        "a": [None, 4, 9, 3],
        "b": [9, 7, 10, 8],
        "c": [9, None, 10, 8],
        "idx": [9, None, None, 7],
        "idx2": [9, 8, 7, 7],
    }
    df = nw.from_native(constructor(data))
    result = (
        df.group_by("grp")
        .agg(
            nw.col("a", "b", "c").first(order_by="idx").name.suffix("_first"),
            nw.col("a", "b", "c").last(order_by="idx").name.suffix("_last"),
        )
        .sort("grp")
    )
    expected = {
        "grp": [1, 2],
        "a_first": [4, 3],
        "b_first": [7, 8],
        "c_first": [None, 8],
        "a_last": [None, 3],
        "b_last": [9, 8],
        "c_last": [9, 8],
    }
    assert_equal_data(result, expected)
    result = (
        df.group_by("grp")
        .agg(
            nw.col("a", "b", "c").first(order_by=["idx", "idx2"]).name.suffix("_first"),
            nw.col("a", "b", "c").last(order_by=["idx", "idx2"]).name.suffix("_last"),
        )
        .sort("grp")
    )
    expected = {
        "grp": [1, 2],
        "a_first": [9, 3],
        "b_first": [10, 8],
        "c_first": [10, 8],
        "a_last": [None, 3],
        "b_last": [9, 8],
        "c_last": [9, 8],
    }
    assert_equal_data(result, expected)


def test_first_expr_broadcasting(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("ibis", "spark", "dask")):
        # ibis: https://github.com/ibis-project/ibis/issues/11656
        request.applymarker(pytest.mark.xfail)
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()

    data = {
        "grp": [1, 1, 1, 2],
        "a": [None, 4, 9, 3],
        "b": [9, 7, 10, 8],
        "c": [9, None, 10, 8],
        "idx": [9, None, None, 7],
        "idx2": [9, 8, 7, 7],
    }
    df = nw.from_native(constructor(data))
    result = df.select(
        "idx", "idx2", nw.col("a", "b", "c").first(order_by="idx").name.suffix("_first")
    ).sort("idx", "idx2")
    expected = {
        "idx": [None, None, 7, 9],
        "idx2": [7, 8, 7, 9],
        "a_first": [4, 4, 4, 4],
        "b_first": [7, 7, 7, 7],
        "c_first": [None, None, None, None],
    }
    assert_equal_data(result, expected)
    result = df.select(
        "idx",
        "idx2",
        nw.col("a", "b", "c").first(order_by=["idx", "idx2"]).name.suffix("_first"),
    ).sort("idx", "idx2")
    expected = {
        "idx": [None, None, 7, 9],
        "idx2": [7, 8, 7, 9],
        "a_first": [9, 9, 9, 9],
        "b_first": [10, 10, 10, 10],
        "c_first": [10, 10, 10, 10],
    }
    assert_equal_data(result, expected)


def test_first_expr_invalid(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {
        "grp": [1, 1, 1, 2],
        "a": [None, 4, 9, 3],
        "b": [9, 7, 10, 8],
        "c": [9, None, 10, 8],
        "idx": [9, None, None, 7],
        "idx2": [9, 8, 7, 7],
    }
    df = nw.from_native(constructor(data))
    with pytest.raises(InvalidOperationError, match="expression which isn't orderable"):
        df.select("idx", "idx2", nw.col("a").first(order_by="idx").over(order_by="idx2"))


def test_first_last_different_orders(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if any(x in str(constructor) for x in ("pyspark", "dask")):
        # Currently unsupported.
        request.applymarker(pytest.mark.xfail)
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (14,):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()

    context = (
        pytest.raises(NotImplementedError, match="Only one `order_by` can")
        if "pandas" in str(constructor) or "pyarrow" in str(constructor)
        else does_not_raise()
    )
    frame = nw.from_native(
        constructor(
            {
                "a": [1, 1, 2],
                "b": [4, 5, 6],
                "c": [None, 7, 8],
                "i_0": [1, None, 2],
                "i_1": [2, None, 1],
            }
        )
    )
    with context:
        result = (
            frame.group_by("a")
            .agg(
                nw.col("b", "c").first(order_by="i_0").name.suffix("_first_i_0"),
                nw.col("b", "c").first(order_by="i_1").name.suffix("_first_i_1"),
                nw.col("b", "c").last(order_by="i_0").name.suffix("_last_i_0"),
                nw.col("b", "c").last(order_by="i_1").name.suffix("_last_i_1"),
            )
            .sort("a")
        )
        expected = {
            "a": [1, 2],
            "b_first_i_0": [5, 6],
            "c_first_i_0": [7, 8],
            "b_first_i_1": [5, 6],
            "c_first_i_1": [7, 8],
            "b_last_i_0": [4, 6],
            "c_last_i_0": [None, 8],
            "b_last_i_1": [4, 6],
            "c_last_i_1": [None, 8],
        }
        assert_equal_data(result, expected)
