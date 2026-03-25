from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    PANDAS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {"a": [1, 2, 3], "b": ["dogs", "cats", None], "c": ["play", "swim", "walk"]}

UNSUPPORTED_BACKENDS = ("dask",)


def maybe_skip(constructor: Constructor | ConstructorEager) -> None:
    if "pandas" in str(constructor) and (
        PANDAS_VERSION < (2, 2, 0) or PYARROW_VERSION == (0, 0, 0)
    ):
        reason = "pandas is too old or pyarrow not installed"
        pytest.skip(reason=reason)

    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        reason = "duckdb requires v1.3.0"
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

    maybe_skip(constructor=constructor)

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

    maybe_skip(constructor=constructor)

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

    maybe_skip(constructor=constructor)

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

    maybe_skip(constructor=constructor)

    df = nw.from_native(constructor(data))
    result = df.select(
        nw.struct(nw.col("a") * 2, nw.col("c").str.len_chars()).alias("struct")
    )

    expected = {"struct": [{"a": 2, "c": 4}, {"a": 4, "c": 4}, {"a": 6, "c": 4}]}

    assert_equal_data(result, expected)


def test_struct_with_literals(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    maybe_skip(constructor=constructor)

    df = nw.from_native(constructor(data))
    result = df.select(nw.struct("a", x="c", y=False).alias("struct"))

    expected = {
        "struct": [
            {"a": 1, "x": "play", "y": False},
            {"a": 2, "x": "swim", "y": False},
            {"a": 3, "x": "walk", "y": False},
        ]
    }

    assert_equal_data(result, expected)


def test_struct_raise_no_exprs(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises(ValueError, match="expected at least 1 expression in 'struct'"):
        df.select(nw.struct().alias("struct"))

    with pytest.raises(ValueError, match="expected at least 1 expression in 'struct'"):
        df.select(nw.struct(schema={"x": nw.Float32()}).alias("struct"))


def test_struct_with_schema(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    maybe_skip(constructor=constructor)

    data_numeric = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    schema = {"a": nw.Float64(), "b": nw.Float32()}
    df = nw.from_native(constructor(data_numeric))
    result = df.select(nw.struct("a", "b", schema=schema).alias("struct"))
    assert result.collect_schema()["struct"] == nw.Struct(schema)

    expected = {
        "struct": [{"a": 1.0, "b": 4.0}, {"a": 2.0, "b": 5.0}, {"a": 3.0, "b": 6.0}]
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (
            {"a": nw.Float32(), "x": nw.Float32()},
            [{"a": 1.0, "x": None}, {"a": 2.0, "x": None}, {"a": 3.0, "x": None}],
        ),
        ({"x": nw.Float32()}, [{"x": None}, {"x": None}, {"x": None}]),
    ],
)
def test_struct_schema_mismatch(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    schema: dict[str, nw.dtypes.DType],
    expected: list[dict[str, Any]],
) -> None:
    if any(x in str(constructor) for x in UNSUPPORTED_BACKENDS):
        request.applymarker(pytest.mark.xfail)

    maybe_skip(constructor=constructor)

    df = nw.from_native(constructor(data))
    result = df.select(nw.struct(nw.all(), schema=schema).alias("struct"))

    assert_equal_data(result, {"struct": expected})


def test_struct_with_series(constructor_eager: ConstructorEager) -> None:
    maybe_skip(constructor=constructor_eager)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    s_a, s_b = df.get_column("a"), df.get_column("b")
    result = df.select(nw.struct(s_a, s_b).alias("struct"))

    expected = {
        "struct": [{"a": 1, "b": "dogs"}, {"a": 2, "b": "cats"}, {"a": 3, "b": None}]
    }

    assert_equal_data(result, expected)


def test_struct_mixed_series_and_exprs(constructor_eager: ConstructorEager) -> None:
    maybe_skip(constructor=constructor_eager)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    s_a = df.get_column("a")
    result = df.select(nw.struct(s_a, nw.col("c")).alias("struct"))

    expected = {
        "struct": [{"a": 1, "c": "play"}, {"a": 2, "c": "swim"}, {"a": 3, "c": "walk"}]
    }

    assert_equal_data(result, expected)


def test_struct_named_with_series(constructor_eager: ConstructorEager) -> None:
    maybe_skip(constructor=constructor_eager)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    s_a = df.get_column("a")
    result = df.select(nw.struct(x=s_a, y="b").alias("struct"))

    expected = {
        "struct": [{"x": 1, "y": "dogs"}, {"x": 2, "y": "cats"}, {"x": 3, "y": None}]
    }

    assert_equal_data(result, expected)
