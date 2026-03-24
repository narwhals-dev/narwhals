from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 2, 3], "b": ["dogs", "cats", None], "c": ["play", "swim", "walk"]}

UNSUPPORTED_BACKENDS = ("dask",)


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


def test_struct_with_schema(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}))
    result = df.select(
        nw.struct("a", "b", schema={"a": nw.Float64(), "b": nw.Float32()}).alias("struct")
    )
    schema = result.collect_schema()
    struct_dtype = schema["struct"]
    assert isinstance(struct_dtype, nw.Struct)
    inner = struct_dtype.fields
    assert inner[0].dtype == nw.Float64()
    assert inner[1].dtype == nw.Float32()

    expected = {
        "struct": [{"a": 1.0, "b": 4.0}, {"a": 2.0, "b": 5.0}, {"a": 3.0, "b": 6.0}]
    }
    assert_equal_data(result, expected)


def test_struct_with_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    s_a = df.get_column("a")
    s_b = df.get_column("b")
    result = df.select(nw.struct(s_a, s_b).alias("struct"))

    expected = {
        "struct": [{"a": 1, "b": "dogs"}, {"a": 2, "b": "cats"}, {"a": 3, "b": None}]
    }

    assert_equal_data(result, expected)


def test_struct_mixed_series_and_exprs(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    s_a = df.get_column("a")
    result = df.select(nw.struct(s_a, nw.col("c")).alias("struct"))

    expected = {
        "struct": [{"a": 1, "c": "play"}, {"a": 2, "c": "swim"}, {"a": 3, "c": "walk"}]
    }

    assert_equal_data(result, expected)


def test_struct_named_with_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    s_a = df.get_column("a")
    result = df.select(nw.struct(x=s_a, y="b").alias("struct"))

    expected = {
        "struct": [{"x": 1, "y": "dogs"}, {"x": 2, "y": "cats"}, {"x": 3, "y": None}]
    }

    assert_equal_data(result, expected)
