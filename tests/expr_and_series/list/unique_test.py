from __future__ import annotations

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [[2, 2, 3, None, None]]}
expected = {"a": [[None, 2, 3]]}
expected_nulls_first= {"a": [None, 2, 3]}
expected_nulls_last= {'a': [[2, 3, None]]}


def test_unique_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("dask", "modin", "cudf")):
        request.applymarker(pytest.mark.xfail)

    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.unique()
    )
    if any(backend in str(constructor) for backend in ("duckdb", "sqlframe", "pyspark", "polars", "ibis")):
        result = result.explode("a").sort("a")
        assert_equal_data(result, expected_nulls_first)
    elif any(backend in str(constructor) for backend in ("pandas", "pyarrow")):
        assert_equal_data(result, expected_nulls_last)
    else:
        assert_equal_data(result, expected)


def test_unique_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in ("modin", "cudf")):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = df["a"].cast(nw.List(nw.Int32())).list.unique()
    if any(backend in str(constructor_eager) for backend in ("pandas", "pyarrow")):
        assert_equal_data({"a": result}, expected_nulls_last)
    else:
        assert_equal_data({"a": result}, expected)