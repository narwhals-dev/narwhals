from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals.stable.v1 as nw
from narwhals.exceptions import AnonymousExprError
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}


def test_to_uppercase(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo", "BAR") * 2).name.to_uppercase())
    expected = {k.upper(): [e * 2 for e in v] for k, v in data.items()}
    assert_equal_data(result, expected)


def test_to_uppercase_after_alias(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo")).alias("alias_for_foo").name.to_uppercase())
    expected = {"FOO": data["foo"]}
    assert_equal_data(result, expected)


def test_to_uppercase_raise_anonymous(constructor: Constructor) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    context = (
        does_not_raise()
        if df.implementation.is_polars() or df.implementation.is_dask() or df.implementation.is_pyspark()
        else pytest.raises(
            AnonymousExprError,
            match="Anonymous expressions are not supported in `.name.to_uppercase`.",
        )
    )

    with context:
        df.select(nw.all().name.to_uppercase())
