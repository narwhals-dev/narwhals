from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pyarrow as pa
import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {
    "i": [0, 1, 2, 3, 4],
    "a": [0, 1, 2, 3, 4],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_shift(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.with_columns(nw.col("a", "b", "c").shift(2)).filter(nw.col("i") > 1)
    expected = {"i": [2, 3, 4], "a": [0, 1, 2], "b": [1, 2, 3], "c": [5, 4, 3]}
    assert_equal_data(result, expected)


def test_shift_lazy(constructor: Constructor) -> None:
    data = {
        "i": [None, 1, 2, 3, 4],
        "a": [0, 1, 2, 3, 4],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
    }
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df = nw.from_native(constructor(data))
    result = df.with_columns(nw.col("a", "b", "c").shift(2).over(order_by="i")).filter(
        nw.col("i") > 1, ~nw.col("i").is_null()
    )
    expected = {"i": [2, 3, 4], "a": [0, 1, 2], "b": [1, 2, 3], "c": [5, 4, 3]}
    assert_equal_data(result, expected)


def test_shift_lazy_grouped(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("dask", "pyarrow_table", "cudf")):
        # https://github.com/dask/dask/issues/11806
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df = nw.from_native(constructor(data))
    result = df.with_columns(nw.col("a").shift(1).over("b", order_by="i")).sort("i")
    expected = {
        "i": [0, 1, 2, 3, 4],
        "a": [None, None, None, None, 2],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
    }
    assert_equal_data(result, expected)


def test_shift_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.with_columns(df["a"].shift(2), df["b"].shift(2), df["c"].shift(2)).filter(
        nw.col("i") > 1
    )
    expected = {"i": [2, 3, 4], "a": [0, 1, 2], "b": [1, 2, 3], "c": [5, 4, 3]}
    assert_equal_data(result, expected)


def test_shift_multi_chunk_pyarrow() -> None:
    tbl = pa.table({"a": [1, 2, 3]})
    tbl = pa.concat_tables([tbl, tbl, tbl])
    df = nw.from_native(tbl, eager_only=True)

    result = df.select(nw.col("a").shift(1))
    expected = {"a": [None, 1, 2, 3, 1, 2, 3, 1, 2]}
    assert_equal_data(result, expected)

    result = df.select(nw.col("a").shift(-1))
    expected = {"a": [2, 3, 1, 2, 3, 1, 2, 3, None]}
    assert_equal_data(result, expected)

    result = df.select(nw.col("a").shift(0))
    expected = {"a": [1, 2, 3, 1, 2, 3, 1, 2, 3]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("n", "context"),
    [
        (1.0, pytest.raises(TypeError, match=r"Expected '.+?', got: '.+?'\s+n=")),
        ("1", pytest.raises(TypeError, match=r"Expected '.+?', got: '.+?'\s+n=")),
        (None, pytest.raises(TypeError, match=r"Expected '.+?', got: '.+?'\s+n=")),
        (1, nullcontext()),
        (0, nullcontext()),
    ],
)
def test_shift_expr_invalid_params(n: Any, context: Any) -> None:
    with context:
        nw.col("a").shift(n)


@pytest.mark.parametrize(
    ("n", "context"),
    [
        (1.0, pytest.raises(TypeError, match=r"Expected '.+?', got: '.+?'\s+n=")),
        ("1", pytest.raises(TypeError, match=r"Expected '.+?', got: '.+?'\s+n=")),
        (None, pytest.raises(TypeError, match=r"Expected '.+?', got: '.+?'\s+n=")),
        (1, nullcontext()),
        (0, nullcontext()),
    ],
)
def test_shift_series_invalid_params(
    constructor_eager: ConstructorEager, n: Any, context: Any
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with context:
        df["a"].shift(n)
