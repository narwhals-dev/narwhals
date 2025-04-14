from __future__ import annotations

import pyarrow as pa
import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION
from tests.utils import POLARS_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "i": [0, 1, 2, 3, 4],
    "a": [0, 1, 2, 3, 4],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_shift(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.with_columns(nw.col("a", "b", "c").shift(2)).filter(nw.col("i") > 1)
    expected = {
        "i": [2, 3, 4],
        "a": [0, 1, 2],
        "b": [1, 2, 3],
        "c": [5, 4, 3],
    }
    assert_equal_data(result, expected)


def test_shift_lazy(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df = nw.from_native(constructor(data))
    result = df.with_columns(nw.col("a", "b", "c").shift(2).over(order_by="i")).filter(
        nw.col("i") > 1
    )
    expected = {
        "i": [2, 3, 4],
        "a": [0, 1, 2],
        "b": [1, 2, 3],
        "c": [5, 4, 3],
    }
    assert_equal_data(result, expected)


def test_shift_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.with_columns(
        df["a"].shift(2),
        df["b"].shift(2),
        df["c"].shift(2),
    ).filter(nw.col("i") > 1)
    expected = {
        "i": [2, 3, 4],
        "a": [0, 1, 2],
        "b": [1, 2, 3],
        "c": [5, 4, 3],
    }
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
