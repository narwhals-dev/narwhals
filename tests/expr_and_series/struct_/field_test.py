from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"user": [{"id": "0", "name": "john"}, {"id": "1", "name": "jane"}]}
expected = {"id": ["0", "1"], "name": ["john", "jane"]}


@pytest.mark.skipif(PANDAS_VERSION < (2, 2, 0), reason="old pandas")
def test_get_field(
    request: pytest.FixtureRequest,
    constructor: Constructor,
) -> None:
    if any(
        backend in str(constructor) for backend in ("dask", "modin", "cudf", "sqlframe")
    ):
        request.applymarker(pytest.mark.xfail)

    _expected = expected.copy()
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
        nw.col("user").struct.field("id"),
        nw.col("user").struct.field("name"),
    )

    assert_equal_data(result, _expected)


def test_polars_series_get_field() -> None:
    import polars as pl

    import narwhals as nw

    s_native = pl.Series(
        [
            {"id": "0", "name": "john"},
            {"id": "1", "name": "jane"},
        ]
    )
    s = nw.from_native(s_native, series_only=True)
    assert s.struct.field("name").to_list() == ["john", "jane"]


def test_pandas_series_get_field() -> None:
    import pandas as pd

    import narwhals as nw

    s_native = pd.Series(
        data=[
            {"id": "0", "name": "john"},
            {"id": "1", "name": "jane"},
        ]
    )
    s = nw.from_native(s_native, series_only=True)

    with pytest.raises(TypeError):
        s.struct.field("name")
