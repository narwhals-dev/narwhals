from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.exceptions import NarwhalsError
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from narwhals.dtypes import DType


@pytest.mark.skipif(
    POLARS_VERSION < (1, 0), reason="replace_strict only available after 1.0"
)
@pytest.mark.parametrize("return_dtype", [nw.String(), None])
def test_replace_strict(
    constructor: Constructor, request: pytest.FixtureRequest, return_dtype: DType | None
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if (
        ("pyspark" in str(constructor))
        or "duckdb" in str(constructor)
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(
        nw.col("a").replace_strict(
            [1, 2, 3], ["one", "two", "three"], return_dtype=return_dtype
        )
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


@pytest.mark.skipif(
    POLARS_VERSION < (1, 0), reason="replace_strict only available after 1.0"
)
@pytest.mark.parametrize("return_dtype", [nw.String(), None])
def test_replace_strict_series(
    constructor_eager: ConstructorEager, return_dtype: DType | None
) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    result = df.select(
        df["a"].replace_strict(
            [1, 2, 3], ["one", "two", "three"], return_dtype=return_dtype
        )
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


@pytest.mark.skipif(
    POLARS_VERSION < (1, 0), reason="replace_strict only available after 1.0"
)
def test_replace_non_full(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if (
        ("pyspark" in str(constructor))
        or "duckdb" in str(constructor)
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    if isinstance(df, nw.LazyFrame):
        with pytest.raises((ValueError, NarwhalsError)):
            df.select(
                nw.col("a").replace_strict([1, 3], [3, 4], return_dtype=nw.Int64)
            ).collect()
    else:
        with pytest.raises((ValueError, NarwhalsError)):
            df.select(nw.col("a").replace_strict([1, 3], [3, 4], return_dtype=nw.Int64))


@pytest.mark.skipif(
    POLARS_VERSION < (1, 0), reason="replace_strict only available after 1.0"
)
def test_replace_strict_mapping(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if (
        ("pyspark" in str(constructor))
        or "duckdb" in str(constructor)
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(
        nw.col("a").replace_strict(
            {1: "one", 2: "two", 3: "three"}, return_dtype=nw.String()
        )
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


@pytest.mark.skipif(
    POLARS_VERSION < (1, 0), reason="replace_strict only available after 1.0"
)
def test_replace_strict_series_mapping(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    result = df.select(
        df["a"].replace_strict({1: "one", 2: "two", 3: "three"}, return_dtype=nw.String())
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


@pytest.mark.skipif(
    POLARS_VERSION < (1, 0), reason="replace_strict only available after 1.0"
)
def test_replace_strict_invalid(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    with pytest.raises(
        TypeError,
        match="`new` argument is required if `old` argument is not a Mapping type",
    ):
        df.select(nw.col("a").replace_strict(old=[1, 2, 3]))


@pytest.mark.skipif(
    POLARS_VERSION < (1, 0), reason="replace_strict only available after 1.0"
)
def test_replace_strict_series_invalid(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    with pytest.raises(
        TypeError,
        match="`new` argument is required if `old` argument is not a Mapping type",
    ):
        df["a"].replace_strict([1, 2, 3])
