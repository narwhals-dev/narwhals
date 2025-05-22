from __future__ import annotations

import narwhals as nw

# Don't move this into typechecking block, for coverage
# purposes
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["fdas", "edfas"]}


def test_ends_with(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.ends_with("das"))
    expected = {"a": [True, False]}
    assert_equal_data(result, expected)


def test_ends_with_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].str.ends_with("das"))
    expected = {"a": [True, False]}
    assert_equal_data(result, expected)


def test_starts_with(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data)).lazy()
    result = df.select(nw.col("a").str.starts_with("fda"))
    expected = {"a": [True, False]}
    assert_equal_data(result, expected)


def test_starts_with_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].str.starts_with("fda"))
    expected = {"a": [True, False]}
    assert_equal_data(result, expected)
