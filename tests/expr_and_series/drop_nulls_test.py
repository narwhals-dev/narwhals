from __future__ import annotations

import polars as pl
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "a": [1, 2, None],
    "b": [3, 4, 5],
    "c": [None, None, None],
    "d": [6, None, None],
}


def test_drop_nulls(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))

    result_a = df.select(nw.col("a").drop_nulls())
    result_b = df.select(nw.col("b").drop_nulls())
    result_c = df.select(nw.col("c").drop_nulls())
    result_d = df.select(nw.col("d").drop_nulls())

    expected_a = {"a": [1.0, 2.0]}
    expected_b = {"b": [3, 4, 5]}
    expected_c = {"c": []}  # type: ignore[var-annotated]
    expected_d = {"d": [6]}

    assert_equal_data(result_a, expected_a)
    assert_equal_data(result_b, expected_b)
    assert_equal_data(result_c, expected_c)
    assert_equal_data(result_d, expected_d)


def test_drop_nulls_broadcast(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "dask" in str(constructor) or (
        "polars" in str(constructor) and parse_version(pl.__version__) >= (1, 7, 0)
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").drop_nulls(), nw.col("d").drop_nulls())
    expected = {"a": [1.0, 2.0], "d": [6, 6]}
    assert_equal_data(result, expected)


def test_drop_nulls_invalid(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data)).lazy()

    with pytest.raises(Exception):  # noqa: B017, PT011
        df.select(nw.col("a").drop_nulls(), nw.col("b").drop_nulls()).collect()


def test_drop_nulls_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_a = df.select(df["a"].drop_nulls())
    result_b = df.select(df["b"].drop_nulls())
    result_c = df.select(df["c"].drop_nulls())
    result_d = df.select(df["d"].drop_nulls())
    expected_a = {"a": [1.0, 2.0]}
    expected_b = {"b": [3, 4, 5]}
    expected_c = {"c": []}  # type: ignore[var-annotated]
    expected_d = {"d": [6]}

    assert_equal_data(result_a, expected_a)
    assert_equal_data(result_b, expected_b)
    assert_equal_data(result_c, expected_c)
    assert_equal_data(result_d, expected_d)
