from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data


def test_get_field_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("dask", "modin")):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor) and PANDAS_VERSION < (2, 2, 0):
        pytest.skip()
    data = {"user": [{"id": "0", "name": "john"}, {"id": "1", "name": "jane"}]}

    df_native = constructor(data)

    if "pandas" in str(constructor):
        df_native = df_native.assign(  # type: ignore[union-attr]
            user=pd.Series(
                data["user"],
                dtype=pd.ArrowDtype(
                    pa.struct([("id", pa.string()), ("name", pa.string())])
                ),
            )
        )

    df = nw.from_native(df_native)

    result = nw.from_native(df).select(
        nw.col("user").struct.field("id"), nw.col("user").struct.field("name")
    )
    expected = {"id": ["0", "1"], "name": ["john", "jane"]}
    assert_equal_data(result, expected)
    result = nw.from_native(df).select(nw.col("user").struct.field("id").name.keep())
    expected = {"user": ["0", "1"]}
    assert_equal_data(result, expected)


def test_get_field_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in ("modin",)):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor_eager) and PANDAS_VERSION < (2, 2, 0):
        pytest.skip()
    data = {"user": [{"id": "0", "name": "john"}, {"id": "1", "name": "jane"}]}
    expected = {"id": ["0", "1"], "name": ["john", "jane"]}

    _expected = expected.copy()
    df_native = constructor_eager(data)

    if "pandas" in str(constructor_eager):
        df_native = df_native.assign(  # type: ignore[union-attr]
            user=pd.Series(
                data["user"],
                dtype=pd.ArrowDtype(
                    pa.struct([("id", pa.string()), ("name", pa.string())])
                ),
            )
        )

    df = nw.from_native(df_native, eager_only=True)

    result = nw.from_native(df).select(
        df["user"].struct.field("id"), df["user"].struct.field("name")
    )
    expected = {"id": ["0", "1"], "name": ["john", "jane"]}
    assert_equal_data(result, _expected)


def test_pandas_object_series() -> None:
    s_native = pd.Series(data=[{"id": "0", "name": "john"}, {"id": "1", "name": "jane"}])
    s = nw.from_native(s_native, series_only=True)

    with pytest.raises(TypeError):
        s.struct.field("name")
