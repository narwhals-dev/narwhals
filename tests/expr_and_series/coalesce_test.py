from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_coalesce_numeric(constructor: Constructor) -> None:
    data = {
        "a": [0, None, None, None, None],
        "b": [1, None, None, 5, 3],
        "c": [5, None, 3, 2, 1],
    }
    result = nw.from_native(constructor(data)).select(result=nw.coalesce(*data, -1))

    assert_equal_data(result, {"result": [0, -1, 3, 5, 3]})


def test_coalesce_strings(constructor: Constructor) -> None:
    data = {
        "a": ["0", None, None, None, None],
        "b": ["1", None, None, "5", "3"],
        "c": ["5", None, "3", "2", "1"],
    }
    result = nw.from_native(constructor(data)).select(
        result=nw.coalesce(*data, nw.lit("xyz"))
    )

    assert_equal_data(result, {"result": ["0", "xyz", "3", "5", "3"]})


def test_coalesce_upcast(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(
        x in str(constructor)
        for x in [
            "pandas_pyarrow",
            "pandas_nullable",
            "modin_pyarrow",
            "pyarrow_table",
            "ibis",
        ]
    ):
        # backends handle upcasting independently, most of them fail
        #   due to strict preferences against automatic upcasting
        request.applymarker(pytest.mark.xfail)

    data = {
        "a": [0, None, None, None, None],
        "b": [1, None, None, 5, 3],
        "c": [5, None, 3, 2, 1],
    }
    result = nw.from_native(constructor(data)).select(
        result=nw.coalesce(*data, nw.lit("xyz"))
    )

    schema = result.collect_schema()
    if schema["result"] == nw.String:
        assert_equal_data(result, {"result": ["0", "xyz", "3", "5", "3"]})
    elif schema["result"] == nw.Object:
        assert_equal_data(result, {"result": [0, "xyz", 3, 5, 3]})
