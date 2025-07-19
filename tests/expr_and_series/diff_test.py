from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {"i": [0, 1, 2, 3, 4], "b": [1, 2, 3, 5, 3], "c": [5, 4, 3, 2, 1]}


def test_diff(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.with_columns(c_diff=nw.col("c").diff()).filter(nw.col("i") > 0)
    expected = {
        "i": [1, 2, 3, 4],
        "b": [2, 3, 5, 3],
        "c": [4, 3, 2, 1],
        "c_diff": [-1, -1, -1, -1],
    }
    assert_equal_data(result, expected)


def test_diff_lazy(constructor: Constructor) -> None:
    data = {"i": [None, 1, 2, 3, 4], "b": [1, 2, 3, 5, 3], "c": [5, 4, 3, 2, 1]}
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df = nw.from_native(constructor(data))
    result = df.with_columns(c_diff=nw.col("c").diff().over(order_by="i")).filter(
        ~nw.col("i").is_null()
    )
    expected = {
        "i": [1, 2, 3, 4],
        "b": [2, 3, 5, 3],
        "c": [4, 3, 2, 1],
        "c_diff": [-1, -1, -1, -1],
    }
    assert_equal_data(result, expected)


def test_diff_lazy_grouped(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    if any(x in str(constructor) for x in ("dask", "pyarrow_table", "cudf")):
        # https://github.com/dask/dask/issues/11806
        # https://github.com/rapidsai/cudf/issues/18160
        # wooah their issue numbers use exactly the same digits but in a different order
        request.applymarker(pytest.mark.xfail)
    data = {"i": [0, 1, 2, 3, 4], "b": [1, 1, 1, 2, 2], "c": [5, 4, 3, 2, 1]}
    df = nw.from_native(constructor(data))
    result = (
        df.with_columns(c_diff=nw.col("c").diff().over("b", order_by="i"))
        .filter(nw.col("i") > 0)
        .sort("i")
    )
    expected = {
        "i": [1, 2, 3, 4],
        "b": [1, 1, 2, 2],
        "c": [4, 3, 2, 1],
        "c_diff": [-1, -1, None, -1],
    }
    assert_equal_data(result, expected)


def test_diff_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = {
        "i": [1, 2, 3, 4],
        "b": [2, 3, 5, 3],
        "c": [4, 3, 2, 1],
        "c_diff": [-1, -1, -1, -1],
    }
    result = df.with_columns(c_diff=df["c"].diff())[1:]
    assert_equal_data(result, expected)
