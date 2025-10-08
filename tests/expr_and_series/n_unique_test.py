from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [1.0, None, None, 3.0], "b": [1.0, None, 4.0, 5.0]}


def test_n_unique(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df = nw.from_native(constructor(data))
    result = df.select(nw.all().n_unique())
    expected = {"a": [3], "b": [4]}
    assert_equal_data(result, expected)


def test_n_unique_over(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/10550
        request.applymarker(pytest.mark.xfail)
    if "pyspark" in str(constructor) and "sqlframe" not in str(constructor):
        # "Distinct window functions are not supported"
        request.applymarker(pytest.mark.xfail)
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    data = {"a": [1, None, None, 1, 2, 2, 2, None, 3], "b": [1, 1, 1, 1, 1, 1, 1, 2, 2]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(
        nw.all().n_unique(), a_over_b=nw.col("a").n_unique().over("b")
    ).sort("a_over_b")
    expected = {"a": [4] * 9, "b": [2] * 9, "a_over_b": [2, 2, 3, 3, 3, 3, 3, 3, 3]}
    assert_equal_data(result, expected)


def test_n_unique_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = {"a": [3], "b": [4]}
    result_series = {"a": [df["a"].n_unique()], "b": [df["b"].n_unique()]}
    assert_equal_data(result_series, expected)
