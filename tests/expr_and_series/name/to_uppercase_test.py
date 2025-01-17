from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import polars as pl
import pytest

import narwhals.stable.v1 as nw
from narwhals.exceptions import AnonymousExprError
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}


def test_to_uppercase(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark")):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo", "BAR") * 2).name.to_uppercase())
    expected = {k.upper(): [e * 2 for e in v] for k, v in data.items()}
    assert_equal_data(result, expected)


def test_to_uppercase_after_alias(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark")):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo")).alias("alias_for_foo").name.to_uppercase())
    expected = {"FOO": data["foo"]}
    assert_equal_data(result, expected)


def test_to_uppercase_raise_anonymous(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark")):
        request.applymarker(pytest.mark.xfail)
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    context = (
        does_not_raise()
        if isinstance(df_raw, (pl.LazyFrame, pl.DataFrame))
        else pytest.raises(
            AnonymousExprError,
            match="Anonymous expressions are not supported in `.name.to_uppercase`.",
        )
    )

    with context:
        df.select(nw.all().name.to_uppercase())
