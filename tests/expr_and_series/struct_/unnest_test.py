from __future__ import annotations

from datetime import datetime

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {
    "id": [0, 1],
    "name": ["john", "jane"],
    "hash": ["fake-hash-1", "fake-hash-2"],
    "ts": [datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 2, 0, 0)],
}

user_dtype = nw.Struct({"id": nw.Int16(), "name": nw.String()})
psw_dtype = nw.Struct({"hash": nw.String(), "ts": nw.Datetime()})

user_expr = nw.struct("id", "name").cast(user_dtype).alias("user")
psw_expr = nw.struct("hash", "ts").cast(psw_dtype).alias("user")


def test_unnest_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("dask",)):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor) and (
        PANDAS_VERSION < (2, 2, 0) or PYARROW_VERSION == (0, 0, 0)
    ):
        pytest.skip()

    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 30):
        pytest.skip()

    df = nw.from_native(constructor(data)).select(user=user_expr, psw=psw_expr)

    result = df.select(nw.col("user").struct.unnest())
    expected = {"id": [0, 1], "name": ["john", "jane"]}
    assert_equal_data(result, expected)


def test_unnest_expr_multi(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(backend in str(constructor) for backend in ("dask",)):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor) and (
        PANDAS_VERSION < (2, 2, 0) or PYARROW_VERSION == (0, 0, 0)
    ):
        pytest.skip()

    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 30):
        pytest.skip()

    df = nw.from_native(constructor(data)).select(user=user_expr, psw=psw_expr)

    result = df.select(nw.col("user", "psw").struct.unnest())
    expected = {
        "id": [0, 1],
        "name": ["john", "jane"],
        "hash": ["fake-hash-1", "fake-hash-2"],
        "ts": [datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 2, 0, 0)],
    }
    assert_equal_data(result, expected)


def test_unnest_series(constructor_eager: ConstructorEager) -> None:
    if "pandas" in str(constructor_eager) and (
        PANDAS_VERSION < (2, 2, 0) or PYARROW_VERSION == (0, 0, 0)
    ):
        pytest.skip()

    df = nw.from_native(constructor_eager(data), eager_only=True).select(user=user_expr)

    result = df.get_column("user").struct.unnest()
    expected = {"id": [0, 1], "name": ["john", "jane"]}
    assert_equal_data(result, expected)
