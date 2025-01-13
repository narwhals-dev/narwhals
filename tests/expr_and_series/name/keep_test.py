from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import polars as pl
import pytest

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}


def test_keep(request: pytest.FixtureRequest, constructor: ConstructorEager) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo", "BAR") * 2).name.keep())
    expected = {k: [e * 2 for e in v] for k, v in data.items()}
    assert_equal_data(result, expected)


def test_keep_after_alias(
    request: pytest.FixtureRequest, constructor: ConstructorEager
) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo")).alias("alias_for_foo").name.keep())
    expected = {"foo": data["foo"]}
    assert_equal_data(result, expected)


def test_keep_raise_anonymous(
    request: pytest.FixtureRequest, constructor: ConstructorEager
) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    context = (
        does_not_raise()
        if isinstance(df_raw, (pl.LazyFrame, pl.DataFrame))
        else pytest.raises(
            ValueError,
            match="Anonymous expressions are not supported in `.name.keep`.",
        )
    )

    with context:
        df.select(nw.all().name.keep())
