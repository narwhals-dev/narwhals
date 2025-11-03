from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.exceptions import NarwhalsError
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from narwhals.dtypes import DType

polars_lt_v1 = POLARS_VERSION < (1, 0, 0)
skip_reason = "replace_strict only available after 1.0"


def xfail_lazy_non_polars_constructor(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    lazy_non_polars_constructors = ("dask", "duckdb", "ibis", "pyspark", "sqlframe")
    if any(x in str(constructor) for x in lazy_non_polars_constructors):
        request.applymarker(pytest.mark.xfail)


@pytest.mark.parametrize("return_dtype", [nw.String(), None])
def test_replace_strict(
    constructor: Constructor, request: pytest.FixtureRequest, return_dtype: DType | None
) -> None:
    xfail_lazy_non_polars_constructor(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(
        nw.col("a").replace_strict(
            [1, 2, 3], ["one", "two", "three"], return_dtype=return_dtype
        )
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


@pytest.mark.parametrize("return_dtype", [nw.String(), None])
def test_replace_strict_series(
    constructor_eager: ConstructorEager, return_dtype: DType | None
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    result = df.select(
        df["a"].replace_strict(
            [1, 2, 3], ["one", "two", "three"], return_dtype=return_dtype
        )
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


def test_replace_non_full(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    xfail_lazy_non_polars_constructor(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    if isinstance(df, nw.LazyFrame):
        with pytest.raises((ValueError, NarwhalsError)):
            df.select(
                nw.col("a").replace_strict([1, 3], [3, 4], return_dtype=nw.Int64)
            ).collect()
    else:
        with pytest.raises((ValueError, NarwhalsError)):
            df.select(nw.col("a").replace_strict([1, 3], [3, 4], return_dtype=nw.Int64))


def test_replace_strict_mapping(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    xfail_lazy_non_polars_constructor(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(
        nw.col("a").replace_strict(
            {1: "one", 2: "two", 3: "three"}, return_dtype=nw.String()
        )
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


def test_replace_strict_series_mapping(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    result = df.select(
        df["a"].replace_strict({1: "one", 2: "two", 3: "three"}, return_dtype=nw.String())
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


def test_replace_strict_invalid(constructor: Constructor) -> None:
    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    with pytest.raises(
        TypeError,
        match="`new` argument is required if `old` argument is not a Mapping type",
    ):
        df.select(nw.col("a").replace_strict(old=[1, 2, 3]))


def test_replace_strict_series_invalid(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    with pytest.raises(
        TypeError,
        match="`new` argument is required if `old` argument is not a Mapping type",
    ):
        df["a"].replace_strict([1, 2, 3])


def test_replace_strict_pandas_unnamed_series() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    ser = nw.from_native(pd.Series([1, 2, 3]), series_only=True)
    result = ser.replace_strict([1, 2, 3], ["one", "two", "three"])
    assert result.name is None


@pytest.mark.parametrize("return_dtype", [nw.String(), None])
def test_replace_strict_with_default(
    constructor: Constructor, request: pytest.FixtureRequest, return_dtype: DType | None
) -> None:
    xfail_lazy_non_polars_constructor(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    df = nw.from_native(constructor({"a": [1, 2, 3, 4]}))
    result = df.select(
        nw.col("a").replace_strict(
            [1, 2], ["one", "two"], default="other", return_dtype=return_dtype
        )
    )
    assert_equal_data(result, {"a": ["one", "two", "other", "other"]})


@pytest.mark.parametrize("return_dtype", [nw.String(), None])
def test_replace_strict_series_with_default(
    constructor_eager: ConstructorEager, return_dtype: DType | None
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    df = nw.from_native(constructor_eager({"a": [1, 2, 3, 4]}))
    result = df.select(
        df["a"].replace_strict(
            [1, 2], ["one", "two"], default="other", return_dtype=return_dtype
        )
    )
    assert_equal_data(result, {"a": ["one", "two", "other", "other"]})


def test_replace_strict_with_default_and_nulls(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    xfail_lazy_non_polars_constructor(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    df = nw.from_native(constructor({"a": [1, 2, None, 4]}))
    result = df.select(
        nw.col("a").replace_strict([1, 2], [10, 20], default=99, return_dtype=nw.Int64)
    )
    assert_equal_data(result, {"a": [10, 20, 99, 99]})


def test_replace_strict_with_default_mapping(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    xfail_lazy_non_polars_constructor(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    df = nw.from_native(constructor({"a": [1, 2, 3, 4]}))
    result = df.select(
        nw.col("a").replace_strict(
            {1: "one", 2: "two", 3: None}, default="other", return_dtype=nw.String()
        )
    )
    assert_equal_data(result, {"a": ["one", "two", None, "other"]})


def test_replace_strict_with_default_as_expr(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    xfail_lazy_non_polars_constructor(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    data = {"a": [1, 2, 3, 4], "b": ["beluga", "narwhal", "orca", "vaquita"]}
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("a").replace_strict(
            {1: "one", 2: "two"}, default=nw.col("b"), return_dtype=nw.String
        )
    )

    assert_equal_data(result, {"a": ["one", "two", "orca", "vaquita"]})


def test_replace_strict_with_default_as_series(
    constructor_eager: ConstructorEager,
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=skip_reason)

    data = {"a": [1, 2, 3, 4], "b": ["beluga", "narwhal", "orca", "vaquita"]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    series = df["a"]
    default = df["b"]
    result = series.replace_strict(
        {1: "one", 2: "two"}, default=default, return_dtype=nw.String
    )

    assert_equal_data({"a": result}, {"a": ["one", "two", "orca", "vaquita"]})
