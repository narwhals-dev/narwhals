from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, Constructor, assert_equal_data


def test_over_pushdown(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    if any(x in str(constructor) for x in ("pyarrow", "pandas", "dask", "cudf", "modin")):
        # non-trivial aggregation
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, -4, 5, 6, -1], "b": [1, 1, 1, 2, 2, 2], "i": list(range(6))}
    df = nw.from_native(constructor(data))
    result = df.select(
        "i",
        a=nw.col("a").sum().abs().over("b"),
        b=nw.col("a").abs().sum().over("b"),
        c=nw.col("a").sum().over("b"),
        d=(nw.col("a").sum() + nw.col("a").sum()).over("b"),
        e=(nw.col("a").sum().abs() + nw.col("a").abs().sum()).over("b"),
    ).sort("i")
    expected = {
        "i": [0, 1, 2, 3, 4, 5],
        "a": [1, 1, 1, 10, 10, 10],
        "b": [7, 7, 7, 12, 12, 12],
        "c": [-1, -1, -1, 10, 10, 10],
        "d": [-2, -2, -2, 20, 20, 20],
        "e": [8, 8, 8, 22, 22, 22],
    }
    assert_equal_data(result, expected)
