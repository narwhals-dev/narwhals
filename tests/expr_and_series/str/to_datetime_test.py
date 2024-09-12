from typing import Any

import pytest

import narwhals.stable.v1 as nw

data = {"a": ["2020-01-01T12:34:56"]}


def test_to_datetime(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_datetime(format="%Y-%m-%dT%H:%M:%S"))
        .collect()
        .item(row=0, column="b")
    )
    assert str(result) == "2020-01-01 12:34:56"
