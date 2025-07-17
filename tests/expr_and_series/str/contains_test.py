from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"pets": ["cat", "dog", "rabbit and parrot", "dove", "Parrot|dove", None]}


def test_contains_case_insensitive(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("pets").str.contains("(?i)parrot|Dove").alias("case_insensitive_match")
    )
    if "pandas_constructor" in str(constructor) and PANDAS_VERSION >= (3,):
        # pandas uses 'str' type, and the result is 'bool', which can't contain missing values.
        expected: dict[str, list[Any]] = {
            "case_insensitive_match": [False, False, True, True, True, False]
        }
    else:
        expected = {"case_insensitive_match": [False, False, True, True, True, None]}
    assert_equal_data(result, expected)


def test_contains_series_case_insensitive(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(case_insensitive_match=df["pets"].str.contains("(?i)parrot|Dove"))
    if "pandas_constructor" in str(constructor_eager) and PANDAS_VERSION >= (3,):
        # pandas uses 'str' type, and the result is 'bool', which can't contain missing values.
        expected: dict[str, Any] = {
            "case_insensitive_match": [False, False, True, True, True, False]
        }
    else:
        expected = {"case_insensitive_match": [False, False, True, True, True, None]}
    assert_equal_data(result, expected)


def test_contains_case_sensitive(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("pets").str.contains("parrot|Dove").alias("default_match"))
    if "pandas_constructor" in str(constructor) and PANDAS_VERSION >= (3,):
        # pandas uses 'str' type, and the result is 'bool', which can't contain missing values.
        expected: dict[str, Any] = {
            "default_match": [False, False, True, False, False, False]
        }
    else:
        expected = {"default_match": [False, False, True, False, False, None]}
    assert_equal_data(result, expected)


def test_contains_series_case_sensitive(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(default_match=df["pets"].str.contains("parrot|Dove"))
    if "pandas_constructor" in str(constructor_eager) and PANDAS_VERSION >= (3,):
        # pandas uses 'str' type, and the result is 'bool', which can't contain missing values.
        expected: dict[str, Any] = {
            "default_match": [False, False, True, False, False, False]
        }
    else:
        expected = {"default_match": [False, False, True, False, False, None]}
    assert_equal_data(result, expected)


def test_contains_literal(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("pets").str.contains("Parrot|dove").alias("default_match"),
        nw.col("pets").str.contains("Parrot|dove", literal=True).alias("literal_match"),
    )
    if "pandas_constructor" in str(constructor) and PANDAS_VERSION >= (3,):
        # pandas uses 'str' type, and the result is 'bool', which can't contain missing values.
        expected: dict[str, Any] = {
            "default_match": [False, False, False, True, True, False],
            "literal_match": [False, False, False, False, True, False],
        }
    else:
        expected = {
            "default_match": [False, False, False, True, True, None],
            "literal_match": [False, False, False, False, True, None],
        }
    assert_equal_data(result, expected)


def test_contains_series_literal(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        default_match=df["pets"].str.contains("Parrot|dove"),
        literal_match=df["pets"].str.contains("Parrot|dove", literal=True),
    )
    if "pandas_constructor" in str(constructor_eager) and PANDAS_VERSION >= (3,):
        # pandas uses 'str' type, and the result is 'bool', which can't contain missing values.
        expected: dict[str, Any] = {
            "default_match": [False, False, False, True, True, False],
            "literal_match": [False, False, False, False, True, False],
        }
    else:
        expected = {
            "default_match": [False, False, False, True, True, None],
            "literal_match": [False, False, False, False, True, None],
        }
    assert_equal_data(result, expected)
