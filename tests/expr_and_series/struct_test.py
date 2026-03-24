from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {"a": [1, 2, 3], "b": ["dogs", "cats", None], "c": ["play", "swim", "walk"]}

UNSUPPORTED_BACKENDS = ("dask", "ibis")


def test_struct(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.struct([nw.col("a"), nw.col("b"), nw.col("c")]).alias("struct"))

    expected = {
        "struct": [
            {"a": 1, "b": "dogs", "c": "play"},
            {"a": 2, "b": "cats", "c": "swim"},
            {"a": 3, "b": None, "c": "walk"},
        ]
    }

    assert_equal_data(result, expected)


def test_struct_positional_args(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.struct(nw.col("a"), nw.col("b")).alias("struct"))

    expected = {
        "struct": [{"a": 1, "b": "dogs"}, {"a": 2, "b": "cats"}, {"a": 3, "b": None}]
    }

    assert_equal_data(result, expected)


def test_struct_named_exprs(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.struct(x="a", y="b").alias("struct"))

    expected = {
        "struct": [{"x": 1, "y": "dogs"}, {"x": 2, "y": "cats"}, {"x": 3, "y": None}]
    }

    assert_equal_data(result, expected)


def test_struct_subset_of_columns(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.struct("a", "c").alias("struct"))

    expected = {
        "struct": [{"a": 1, "c": "play"}, {"a": 2, "c": "swim"}, {"a": 3, "c": "walk"}]
    }

    assert_equal_data(result, expected)


def test_struct_with_expressions(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.struct(nw.col("a") * 2, nw.col("c")).alias("struct"))

    expected = {
        "struct": [{"a": 2, "c": "play"}, {"a": 4, "c": "swim"}, {"a": 6, "c": "walk"}]
    }

    assert_equal_data(result, expected)


def test_struct_single_column(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.struct("a").alias("struct"))

    expected = {"struct": [{"a": 1}, {"a": 2}, {"a": 3}]}

    assert_equal_data(result, expected)
