from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["foo", "bars"]}


def test_str_head(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.head(3))
    expected = {"a": ["foo", "bar"]}
    assert_equal_data(result, expected)


def test_str_head_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = {"a": ["foo", "bar"]}
    result = df.select(df["a"].str.head(3))
    assert_equal_data(result, expected)


def test_str_head_negative_n(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "polars" in str(constructor):
        reason = (
            "narwhals maps `head` onto `str.slice`, and polars rejects a negative length "
            "there even though `polars.Expr.str.head` accepts one"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))
    if any(
        backend in str(constructor)
        for backend in ("duckdb", "sqlframe", "pyspark", "ibis")
    ):
        reason = "`substr` either rejects a negative length or returns an empty string"
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.head(-1))
    assert_equal_data(result, {"a": ["fo", "bar"]})
