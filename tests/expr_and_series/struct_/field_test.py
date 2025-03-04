from __future__ import annotations

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
    if any(backend in str(constructor) for backend in ("dask", "modin", "cudf")):
        request.applymarker(pytest.mark.xfail)

    df_native = constructor(data)
    df = nw.from_native(df_native)
    result = nw.from_native(df).select(
        nw.col("user").struct.field("id"),
        nw.col("user").struct.field("name"),
    )

    if "pandas" in str(constructor):
        _df = result.to_native()

    assert_equal_data(result, expected)
