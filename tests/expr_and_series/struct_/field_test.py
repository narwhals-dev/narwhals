from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"user": [{"id": "0", "name": "john"}, {"id": "1", "name": "jane"}]}
expected = {"id": ["0", "1"], "name": ["john", "jane"]}


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

    # Pandas may use 2 different methods depending on the data type
    if "pandas" in str(constructor):
        df_native.assign(  # type: ignore[union-attr]
            user_arrow=pd.Series(
                data["user"],
                dtype=pd.ArrowDtype(
                    pa.struct([("id", pa.string()), ("name", pa.string())])
                ),
            )
        )

    df = nw.from_native(df_native)
    selects = [
        nw.col("user").struct.field("id"),
        nw.col("user").struct.field("name"),
    ]
    if "pandas" in str(constructor):
        selects += [
            nw.col("user").struct.field("name").alias("name_arrow"),
        ]
        _expected["name_arrow"] = _expected["name"]

    result = nw.from_native(df).select(*selects)

    assert_equal_data(result, _expected)
