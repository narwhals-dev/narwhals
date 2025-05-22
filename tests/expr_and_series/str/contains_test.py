from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

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
    expected = {"case_insensitive_match": [False, False, True, True, True, None]}
    assert_equal_data(result, expected)


def test_contains_series_case_insensitive(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(case_insensitive_match=df["pets"].str.contains("(?i)parrot|Dove"))
    expected = {"case_insensitive_match": [False, False, True, True, True, None]}
    assert_equal_data(result, expected)


def test_contains_case_sensitive(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("pets").str.contains("parrot|Dove").alias("default_match"))
    expected = {"default_match": [False, False, True, False, False, None]}
    assert_equal_data(result, expected)


def test_contains_series_case_sensitive(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(default_match=df["pets"].str.contains("parrot|Dove"))
    expected = {"default_match": [False, False, True, False, False, None]}
    assert_equal_data(result, expected)


def test_contains_literal(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("pets").str.contains("Parrot|dove").alias("default_match"),
        nw.col("pets").str.contains("Parrot|dove", literal=True).alias("literal_match"),
    )
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
    expected = {
        "default_match": [False, False, False, True, True, None],
        "literal_match": [False, False, False, False, True, None],
    }
    assert_equal_data(result, expected)
