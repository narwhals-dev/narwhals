from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_cast(constructor: Constructor) -> None:
    data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]}
    df = nw.from_native(constructor(data))
    original = df.collect_schema()

    result = df.cast({"a": nw.Float64, "b": nw.Int32})
    schema = result.collect_schema()
    assert schema["a"] == nw.Float64
    assert schema["b"] == nw.Int32
    # A column that is not in the mapping keeps its dtype.
    assert schema["c"] == original["c"]
    assert_equal_data(
        result, {"a": [1.0, 2.0, 3.0], "b": [4, 5, 6], "c": ["x", "y", "z"]}
    )


def test_cast_empty_mapping(constructor: Constructor) -> None:
    data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    df = nw.from_native(constructor(data))
    result = df.cast({})
    assert result.collect_schema() == df.collect_schema()
    assert_equal_data(result, data)


def test_cast_nonexistent_column(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    with pytest.raises(nw.exceptions.ColumnNotFoundError):
        df.cast({"b": nw.Int64})
