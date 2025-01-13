from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_clone(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    expected = {"a": [1, 2], "b": [3, 4]}
    df = nw.from_native(constructor(expected))
    df_clone = df.clone()
    assert df is not df_clone
    assert df._compliant_frame is not df_clone._compliant_frame
    assert_equal_data(df_clone, expected)
