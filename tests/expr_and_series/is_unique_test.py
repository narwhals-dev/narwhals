from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, Constructor, ConstructorEager, assert_equal_data


def test_is_unique_expr(nw_frame_constructor: Constructor) -> None:
    if "duckdb" in str(nw_frame_constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()

    data = {"a": [1, 1, 2], "b": [1, 2, 3], "index": [0, 1, 2]}
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(nw.col("a", "b").is_unique(), "index").sort("index")
    expected = {"a": [False, False, True], "b": [True, True, True], "index": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_is_unique_expr_grouped(
    nw_frame_constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(nw_frame_constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    if any(
        x in str(nw_frame_constructor)
        for x in ("pandas", "dask", "modin", "cudf", "pyarrow")
    ):
        # non-trivial aggregation
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 1, 2, 4, 4], "b": [1, 1, 1, 2, 2], "index": [0, 1, 2, 3, 4]}
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(nw.col("a").is_unique().over("b"), "index").sort("index")
    expected = {"a": [False, False, True, False, False], "index": [0, 1, 2, 3, 4]}
    assert_equal_data(result, expected)


def test_is_unique_w_nulls_expr(nw_frame_constructor: Constructor) -> None:
    if "duckdb" in str(nw_frame_constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()

    data = {"a": [None, 1, 2], "b": [None, 2, None], "index": [0, 1, 2]}
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(nw.col("a", "b").is_unique(), "index").sort("index")
    expected = {"a": [True, True, True], "b": [False, True, False], "index": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_is_unique_series(nw_eager_constructor: ConstructorEager) -> None:
    data = {"a": [1, 1, 2], "b": [1, 2, 3], "index": [0, 1, 2]}
    series = nw.from_native(nw_eager_constructor(data), eager_only=True)["a"]
    result = series.is_unique()
    expected = {"a": [False, False, True]}
    assert_equal_data({"a": result}, expected)
