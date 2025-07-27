from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, Constructor, assert_equal_data

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}


def test_to_lowercase(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo", "BAR") * 2).name.to_lowercase())
    expected = {"foo": [2, 4, 6], "bar": [8, 10, 12]}
    assert_equal_data(result, expected)


def test_to_lowercase_after_alias(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 32):
        pytest.skip(reason="https://github.com/pola-rs/polars/issues/23765")
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("BAR")).alias("ALIAS_FOR_BAR").name.to_lowercase())
    expected = {"alias_for_bar": [4, 5, 6]}
    assert_equal_data(result, expected)


def test_to_lowercase_raise_anonymous(constructor: Constructor) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.select(nw.all().name.to_lowercase())
    expected = {"foo": [1, 2, 3], "bar": [4, 5, 6]}
    assert_equal_data(result, expected)
