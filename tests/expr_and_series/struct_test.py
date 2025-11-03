from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, Constructor, assert_equal_data

pytest.importorskip("pyarrow")

data = {"a": [1, 2, 3], "b": ["dogs", "cats", None], "c": ["play", "swim", "walk"]}

def test_struct(constructor: Constructor, *, request: pytest.FixtureRequest) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 0, 0):
        request.applymarker(pytest.mark.xfail)
    if any(
        x in str(constructor) for x in ("dask", "duckdb", "ibis", "pyspark", "sqlframe")
    ):
        request.applymarker(pytest.mark.xfail(reason="Not supported / not implemented"))

    df = nw.from_native(constructor(data))
    result = df.select(nw.struct([nw.col("a"), nw.col("b"), nw.col("c")]).alias("struct"))

    expected = {
        "struct": [
            {"a": 1, "b": "dogs", "c": "play"},
            {"a": 2, "b": "cats", "c": "swim"},
            {"a": 3, "b": None,  "c": "walk"},
        ]
    }

    assert_equal_data(result, expected)