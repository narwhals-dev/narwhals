from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)


def test_unnest_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("dask",)):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor) and (
        PANDAS_VERSION < (2, 2, 0) or PYARROW_VERSION == (0, 0, 0)
    ):
        pytest.skip()

    data = {"user": [{"id": 0, "name": "john"}, {"id": 1, "name": "jane"}]}
    dtype = nw.Struct({"id": nw.Int16(), "name": nw.String()})
    df = nw.from_native(constructor(data)).select(user=nw.col("user").cast(dtype))

    result = df.select(nw.col("user").struct.unnest())
    expected = {"id": [0, 1], "name": ["john", "jane"]}
    assert_equal_data(result, expected)


def test_unnest_series(constructor_eager: ConstructorEager) -> None:
    if "pandas" in str(constructor_eager) and (
        PANDAS_VERSION < (2, 2, 0) or PYARROW_VERSION == (0, 0, 0)
    ):
        pytest.skip()

    data = {"user": [{"id": 0, "name": "john"}, {"id": 1, "name": "jane"}]}
    dtype = nw.Struct({"id": nw.Int16(), "name": nw.String()})
    df = nw.from_native(constructor_eager(data), eager_only=True).select(
        user=nw.col("user").cast(dtype)
    )

    result = df.get_column("user").struct.unnest()
    expected = {"id": [0, 1], "name": ["john", "jane"]}
    assert_equal_data(result, expected)
