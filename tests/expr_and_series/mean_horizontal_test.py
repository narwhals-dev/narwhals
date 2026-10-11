from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_meanh(constructor: Constructor) -> None:
    data = {"a": [1, 3, None, None], "b": [4, None, 6, None]}
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_mean=nw.mean_horizontal(nw.col("a"), nw.col("b")))
    expected = {"horizontal_mean": [2.5, 3.0, 6.0, None]}
    assert_equal_data(result, expected)


def test_meanh_with_literal(constructor: Constructor) -> None:
    data = {"a": [1, 3, None, None], "b": [4, None, 6, None]}
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_mean=nw.mean_horizontal(nw.lit(1), "a", nw.col("b")))
    expected = {"horizontal_mean": [2.0, 2.0, 3.5, 1.0]}
    assert_equal_data(result, expected)


def test_meanh_all(constructor: Constructor) -> None:
    data = {"a": [2, 4, 6], "b": [10, 20, 30]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.mean_horizontal(nw.all()))
    expected = {"a": [6, 12, 18]}
    assert_equal_data(result, expected)
    result = df.select(c=nw.mean_horizontal(nw.all()))
    expected = {"c": [6, 12, 18]}
    assert_equal_data(result, expected)


def test_meanh_all_null(constructor: Constructor) -> None:
    data = {"a": [None, 1.0, None], "b": [None, 3.0, 5.0]}
    df = nw.from_native(constructor(data))
    result = df.select(
        mean=nw.mean_horizontal("a", "b"), is_null=nw.mean_horizontal("a", "b").is_null()
    )
    expected = {"mean": [None, 2.0, 5.0], "is_null": [True, False, False]}
    assert_equal_data(result, expected)


def test_meanh_all_null_duckdb() -> None:
    pytest.importorskip("duckdb")
    data = {"a": [None, 1.0, None], "b": [None, 3.0, 5.0]}
    schema = {"a": nw.Float64(), "b": nw.Float64()}
    lf = nw.from_dict(data, schema=schema, backend="pyarrow").lazy("duckdb")
    result = lf.select(
        mean=nw.mean_horizontal("a", "b"), is_null=nw.mean_horizontal("a", "b").is_null()
    ).collect()
    expected = {"mean": [None, 2.0, 5.0], "is_null": [True, False, False]}
    assert_equal_data(result, expected)
