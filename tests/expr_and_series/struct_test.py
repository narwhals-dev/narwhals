from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {"a": [1, 2, 3], "b": ["dogs", "cats", None], "c": ["play", "swim", "walk"]}

UNSUPPORTED_BACKENDS = ("dask",)


def skip_pandas(constructor: Constructor | ConstructorEager) -> None:
    if "pandas" in str(constructor) and (
        PANDAS_VERSION < (2, 2, 0) or PYARROW_VERSION == (0, 0, 0)
    ):
        reason = "too old or pyarrow not installed"
        pytest.skip(reason=reason)


@pytest.mark.parametrize(
    "exprs",
    [
        (nw.col("a"), nw.col("b"), nw.col("c")),
        ([nw.col("a"), nw.col("b"), nw.col("c")]),
        (nw.all(),),
        ("a", "b", "c"),
    ],
)
def test_struct_positional_exprs(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    exprs: tuple[nw.Expr | list[nw.Expr], ...],
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(nw.struct(*exprs))

    expected = {
        "a": [
            {"a": 1, "b": "dogs", "c": "play"},
            {"a": 2, "b": "cats", "c": "swim"},
            {"a": 3, "b": None, "c": "walk"},
        ]
    }

    assert_equal_data(result, expected)


def test_struct_named_exprs(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    skip_pandas(constructor=constructor)

    df = nw.from_native(constructor(data))
    result = df.select(nw.struct(x="a", y="b").alias("struct"))

    expected = {
        "struct": [{"x": 1, "y": "dogs"}, {"x": 2, "y": "cats"}, {"x": 3, "y": None}]
    }

    assert_equal_data(result, expected)


def test_struct_positional_and_named(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    skip_pandas(constructor=constructor)

    df = nw.from_native(constructor(data))
    result = df.select(nw.struct("a", z="c").alias("struct"))

    expected = {
        "struct": [{"a": 1, "z": "play"}, {"a": 2, "z": "swim"}, {"a": 3, "z": "walk"}]
    }

    assert_equal_data(result, expected)


def test_struct_with_expressions(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    skip_pandas(constructor=constructor)

    df = nw.from_native(constructor(data))
    result = df.select(
        nw.struct(nw.col("a") * 2, nw.col("c").str.len_chars()).alias("struct")
    )

    expected = {"struct": [{"a": 2, "c": 4}, {"a": 4, "c": 4}, {"a": 6, "c": 4}]}

    assert_equal_data(result, expected)


def test_struct_with_schema(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    skip_pandas(constructor=constructor)

    data_numeric = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    schema = {"a": nw.Float64(), "b": nw.Float32()}
    df = nw.from_native(constructor(data_numeric))
    result = df.select(nw.struct("a", "b", schema=schema).alias("struct"))
    assert result.collect_schema()["struct"] == nw.Struct(schema)

    expected = {
        "struct": [{"a": 1.0, "b": 4.0}, {"a": 2.0, "b": 5.0}, {"a": 3.0, "b": 6.0}]
    }
    assert_equal_data(result, expected)


def test_struct_with_series(constructor_eager: ConstructorEager) -> None:
    skip_pandas(constructor=constructor_eager)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    s_a, s_b = df.get_column("a"), df.get_column("b")
    result = df.select(nw.struct(s_a, s_b).alias("struct"))

    expected = {
        "struct": [{"a": 1, "b": "dogs"}, {"a": 2, "b": "cats"}, {"a": 3, "b": None}]
    }

    assert_equal_data(result, expected)


def test_struct_mixed_series_and_exprs(constructor_eager: ConstructorEager) -> None:
    skip_pandas(constructor=constructor_eager)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    s_a = df.get_column("a")
    result = df.select(nw.struct(s_a, nw.col("c")).alias("struct"))

    expected = {
        "struct": [{"a": 1, "c": "play"}, {"a": 2, "c": "swim"}, {"a": 3, "c": "walk"}]
    }

    assert_equal_data(result, expected)


def test_struct_named_with_series(constructor_eager: ConstructorEager) -> None:
    skip_pandas(constructor=constructor_eager)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    s_a = df.get_column("a")
    result = df.select(nw.struct(x=s_a, y="b").alias("struct"))

    expected = {
        "struct": [{"x": 1, "y": "dogs"}, {"x": 2, "y": "cats"}, {"x": 3, "y": None}]
    }

    assert_equal_data(result, expected)
