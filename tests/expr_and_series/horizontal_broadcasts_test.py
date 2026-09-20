from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, Constructor, assert_equal_data


def test_sumh_broadcasting(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "i": [0, 1, 2]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(
        any=nw.any_horizontal(nw.sum("a", "b").cast(nw.Boolean), ignore_nulls=True),
        all=nw.all_horizontal(nw.sum("a", "b").cast(nw.Boolean), ignore_nulls=True),
        max=nw.max_horizontal(nw.sum("a"), nw.sum("b")),
        min=nw.min_horizontal(nw.sum("a"), nw.sum("b")),
        sum=nw.sum_horizontal(nw.sum("a"), nw.sum("b")),
        mean=nw.mean_horizontal(nw.sum("a"), nw.sum("b")),
    ).sort("i")
    expected = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "i": [0, 1, 2],
        "any": [True, True, True],
        "all": [True, True, True],
        "max": [15, 15, 15],
        "min": [6, 6, 6],
        "sum": [21, 21, 21],
        "mean": [10.5, 10.5, 10.5],
    }
    assert_equal_data(result, expected)


SCALAR_OPERAND_CASES: dict[str, tuple[nw.Expr, list[object]]] = {
    "sum_lit_first": (nw.sum_horizontal(nw.lit(10), "a", "b"), [15, 15, 13]),
    "sum_agg_first": (nw.sum_horizontal(nw.col("a").sum(), "b"), [8, 9, 4]),
    "sum_all_scalar": (nw.sum_horizontal(nw.lit(1), nw.col("a").sum()), [5, 5, 5]),
    "mean_lit_first": (nw.mean_horizontal(nw.lit(10), "a", "b"), [5.0, 7.5, 6.5]),
    "min_lit_first": (nw.min_horizontal(nw.lit(2), "a", "b"), [1, 2, 2]),
    "max_lit_first": (nw.max_horizontal(nw.lit(2), "a", "b"), [4, 5, 3]),
    "all_lit_first": (
        nw.all_horizontal(nw.lit(True), nw.col("i") > 0, ignore_nulls=True),
        [False, True, True],
    ),
    "any_lit_first": (
        nw.any_horizontal(nw.lit(False), nw.col("i") > 0, ignore_nulls=True),
        [False, True, True],
    ),
    "coalesce_agg_fallback": (nw.coalesce("a", nw.col("b").max()), [1, 5, 3]),
    "coalesce_lit_first": (nw.coalesce(nw.lit(7), "a"), [7, 7, 7]),
}


@pytest.mark.parametrize("name", SCALAR_OPERAND_CASES)
def test_scalar_operand_broadcasting(constructor: Constructor, name: str) -> None:
    aggregates = name in {"sum_agg_first", "sum_all_scalar", "coalesce_agg_fallback"}
    if aggregates and "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip("aggregations broadcast via window functions, DuckDB>=1.3 only")
    expr, expected = SCALAR_OPERAND_CASES[name]
    data = {"a": [1, None, 3], "b": [4, 5, None], "i": [0, 1, 2]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(result=expr).sort("i").select("result")
    assert_equal_data(result, {"result": expected})
