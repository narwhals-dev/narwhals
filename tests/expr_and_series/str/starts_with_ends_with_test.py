from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.testing.typing import Constructor, ConstructorEager

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
