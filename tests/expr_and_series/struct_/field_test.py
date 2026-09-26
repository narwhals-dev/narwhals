from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    PANDAS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)


def test_get_field_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    pytest.importorskip("pyarrow")

    if any(backend in str(constructor) for backend in ("dask",)):
        request.applymarker(pytest.mark.xfail)
    if ("pandas" in str(constructor) and PANDAS_VERSION < (2, 2, 0)) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3, 0)
    ):
        pytest.skip()

    data = {"id": ["0", "1"], "name": ["john", "jane"]}
    expected = data.copy()
    df = constructor(data, nw).select(user=nw.struct("id", "name"))

    result = nw.from_native(df).select(
        nw.col("user").struct.field("id"), nw.col("user").struct.field("name")
    )
    assert_equal_data(result, expected)
    result = nw.from_native(df).select(nw.col("user").struct.field("id").name.keep())
    expected = {"user": ["0", "1"]}
    assert_equal_data(result, expected)


def test_get_field_series(constructor_eager: ConstructorEager) -> None:
    pytest.importorskip("pyarrow")

    if "pandas" in str(constructor_eager) and PANDAS_VERSION < (2, 2, 0):
        pytest.skip()
    data = {"id": ["0", "1"], "name": ["john", "jane"]}
    expected = data.copy()
    df = constructor_eager(data, nw).select(user=nw.struct("id", "name"))

    result = nw.from_native(df).select(
        df["user"].struct.field("id"), df["user"].struct.field("name")
    )
    assert_equal_data(result, expected)


def test_pandas_object_series() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s_native = pd.Series(data=[{"id": "0", "name": "john"}, {"id": "1", "name": "jane"}])
    s = nw.from_native(s_native, series_only=True)

    with pytest.raises(TypeError):
        s.struct.field("name")
