from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"pets": ["cat", "dog", "rabbit and parrot", "dove", "Parrot|dove", None]}
"""Test data for string literal pattern tests"""

EXPR_PATTERN_UNSUPPORTED = ("dask", "pyarrow", "pandas", "modin", "cudf")
"""Backends that don't support expression/series patterns"""

# Test cases for string literal patterns: (pattern, literal, expected_with_null, expected_without_null)
contains_str_pattern_data = [
    # Case insensitive regex
    (
        "(?i)parrot|Dove",
        False,
        {"match": [False, False, True, True, True, None]},
        {"match": [False, False, True, True, True, False]},
    ),
    # Case sensitive regex
    (
        "parrot|Dove",
        False,
        {"match": [False, False, True, False, False, None]},
        {"match": [False, False, True, False, False, False]},
    ),
    # Literal match (not regex)
    (
        "Parrot|dove",
        True,
        {"match": [False, False, False, False, True, None]},
        {"match": [False, False, False, False, True, False]},
    ),
]

# Test cases for expression patterns: (data, literal, expected)
contains_expr_pattern_data = [
    # Basic literal match
    (
        {"text": ["x", "y", "z"], "pattern": ["x", "z", "z"]},
        True,
        {"result": [True, False, True]},
    ),
    # Regex patterns with literal=False
    (
        {
            "text": ["hello world", "foo bar", "test123", "HELLO"],
            "pattern": ["^hello", "bar$", r"\d+", "(?i)hello"],
        },
        False,
        {"result": [True, True, True, True]},
    ),
    # Substrings and partial matches
    (
        {
            "text": ["abcdef", "xyz", "abc", "ABCDEF", ""],
            "pattern": ["cd", "xy", "abc", "abc", ""],
        },
        True,
        {"result": [True, True, True, False, True]},
    ),
    # Special regex characters with literal=True
    (
        {
            "text": ["a.b", "a*b", "a+b", "a?b", "a[b"],
            "pattern": [".", "*", "+", "?", "["],
        },
        True,
        {"result": [True, True, True, True, True]},
    ),
]


@pytest.mark.parametrize(
    ("pattern", "literal", "expected_with_null", "expected_without_null"),
    contains_str_pattern_data,
)
def test_expr_contains_str_pattern(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    pattern: str,
    *,
    literal: bool,
    expected_with_null: dict[str, list[Any]],
    expected_without_null: dict[str, list[Any]],
) -> None:
    if "cudf" in str(constructor) and "(?i)" in pattern:
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("pets").str.contains(pattern, literal=literal).alias("match")
    )

    if "pandas_constructor" in str(constructor) and PANDAS_VERSION >= (3,):
        expected = expected_without_null
    else:
        expected = expected_with_null
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("pattern", "literal", "expected_with_null", "expected_without_null"),
    contains_str_pattern_data,
)
def test_series_contains_str_pattern(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    pattern: str,
    *,
    literal: bool,
    expected_with_null: dict[str, list[Any]],
    expected_without_null: dict[str, list[Any]],
) -> None:
    if "cudf" in str(constructor_eager) and "(?i)" in pattern:
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(match=df["pets"].str.contains(pattern, literal=literal))

    if "pandas_constructor" in str(constructor_eager) and PANDAS_VERSION >= (3,):
        expected = expected_without_null
    else:
        expected = expected_with_null
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("expr_data", "literal", "expected"), contains_expr_pattern_data)
def test_expr_contains_expr_pattern(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    expr_data: dict[str, list[str]],
    *,
    literal: bool,
    expected: dict[str, list[Any]],
) -> None:
    if any(x in str(constructor) for x in EXPR_PATTERN_UNSUPPORTED):
        request.applymarker(pytest.mark.xfail(reason="Not supported", raises=TypeError))

    df = nw.from_native(constructor(expr_data))
    result = df.select(
        nw.col("text").str.contains(nw.col("pattern"), literal=literal).alias("result")
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("expr_data", "literal", "expected"), contains_expr_pattern_data)
def test_series_contains_series_pattern(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    expr_data: dict[str, list[str]],
    *,
    literal: bool,
    expected: dict[str, list[Any]],
) -> None:
    if any(x in str(constructor_eager) for x in EXPR_PATTERN_UNSUPPORTED):
        request.applymarker(pytest.mark.xfail(reason="Not supported", raises=TypeError))

    df = nw.from_native(constructor_eager(expr_data), eager_only=True)
    result = df.select(result=df["text"].str.contains(df["pattern"], literal=literal))
    assert_equal_data(result, expected)


def test_expr_contains_literal_vs_regex(constructor: Constructor) -> None:
    """Test that literal=True vs literal=False behaves differently for regex patterns."""
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("pets").str.contains("Parrot|dove", literal=False).alias("regex_match"),
        nw.col("pets").str.contains("Parrot|dove", literal=True).alias("literal_match"),
    )
    if "pandas_constructor" in str(constructor) and PANDAS_VERSION >= (3,):
        expected: dict[str, Any] = {
            "regex_match": [False, False, False, True, True, False],
            "literal_match": [False, False, False, False, True, False],
        }
    else:
        expected = {
            "regex_match": [False, False, False, True, True, None],
            "literal_match": [False, False, False, False, True, None],
        }
    assert_equal_data(result, expected)


def test_series_contains_literal_vs_regex(constructor_eager: ConstructorEager) -> None:
    """Test that literal=True vs literal=False behaves differently for regex patterns."""
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        regex_match=df["pets"].str.contains("Parrot|dove", literal=False),
        literal_match=df["pets"].str.contains("Parrot|dove", literal=True),
    )
    if "pandas_constructor" in str(constructor_eager) and PANDAS_VERSION >= (3,):
        expected: dict[str, Any] = {
            "regex_match": [False, False, False, True, True, False],
            "literal_match": [False, False, False, False, True, False],
        }
    else:
        expected = {
            "regex_match": [False, False, False, True, True, None],
            "literal_match": [False, False, False, False, True, None],
        }
    assert_equal_data(result, expected)
