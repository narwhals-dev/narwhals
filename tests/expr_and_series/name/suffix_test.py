from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import polars as pl
import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}
suffix = "_with_suffix"


def test_suffix(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo", "BAR") * 2).name.suffix(suffix))
    expected = {str(k) + suffix: [e * 2 for e in v] for k, v in data.items()}
    assert_equal_data(result, expected)


def test_suffix_after_alias(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo")).alias("alias_for_foo").name.suffix(suffix))
    expected = {"foo" + suffix: data["foo"]}
    assert_equal_data(result, expected)


def test_suffix_raise_anonymous(constructor: Constructor) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    context = (
        does_not_raise()
        if df.implementation.is_polars() or df.implementation.is_dask() or df.implementation.is_pyspark()
        else pytest.raises(
            ValueError,
            match="Anonymous expressions are not supported in `.name.suffix`.",
        )
    )

    with context:
        df.select(nw.all().name.suffix(suffix))
