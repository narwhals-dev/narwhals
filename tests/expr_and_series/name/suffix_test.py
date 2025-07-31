from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, Constructor, assert_equal_data

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}
suffix = "_with_suffix"


def test_suffix(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo", "BAR") * 2).name.suffix(suffix))
    expected = {"foo_with_suffix": [2, 4, 6], "BAR_with_suffix": [8, 10, 12]}
    assert_equal_data(result, expected)


def test_suffix_after_alias(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 32):
        pytest.skip(reason="https://github.com/pola-rs/polars/issues/23765")
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo")).alias("alias_for_foo").name.suffix(suffix))
    expected = {"alias_for_foo_with_suffix": [1, 2, 3]}
    assert_equal_data(result, expected)
    result = df.select(
        nw.col("foo").alias("bar").name.prefix("prefix_").name.suffix("_suffix")
    )
    expected = {"prefix_bar_suffix": [1, 2, 3]}
    assert_equal_data(result, expected)


def test_suffix_anonymous(constructor: Constructor) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.select(nw.all().name.suffix(suffix))
    expected = {"foo_with_suffix": [1, 2, 3], "BAR_with_suffix": [4, 5, 6]}
    assert_equal_data(result, expected)
