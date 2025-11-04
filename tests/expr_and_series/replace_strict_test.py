from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from narwhals.dtypes import DType

polars_lt_v1 = POLARS_VERSION < (1, 0, 0)
pl_skip_reason = "replace_strict only available after 1.0"
sqlframe_xfail_reason = (
    "AttributeError: module 'sqlframe.duckdb.functions' has no attribute 'map_keys'"
)


def xfail_if_no_default(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    lazy_non_polars_constructors = ("dask", "duckdb", "ibis", "pyspark", "sqlframe")
    if any(x in str(constructor) for x in lazy_non_polars_constructors):
        request.applymarker(pytest.mark.xfail)


@pytest.mark.parametrize(
    ("old", "new", "return_dtype"),
    [
        (["one", "two", "three"], [1, 2, 3], nw.Int8()),
        (["one", "two", "three"], [1, 2, 3], None),
        ({"one": 1, "two": 2, "three": 3}, None, nw.Float32()),
    ],
)
def test_replace_strict_expr_basic(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    old: Sequence[Any] | Mapping[Any, Any],
    new: Sequence[Any] | None,
    return_dtype: DType | None,
) -> None:
    xfail_if_no_default(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df = nw.from_native(constructor({"a": ["one", "two", "three"]}))
    result = df.select(nw.col("a").replace_strict(old, new, return_dtype=return_dtype))
    assert_equal_data(result, {"a": [1, 2, 3]})
    if return_dtype is not None:
        assert result.collect_schema() == {"a": return_dtype}


@pytest.mark.parametrize(
    ("old", "new", "return_dtype"),
    [
        ([1, 2, 3], ["one", "two", "three"], nw.String()),
        ([1, 2, 3], ["one", "two", "three"], None),
        ({1: "one", 2: "two", 3: "three"}, None, nw.String()),
    ],
)
def test_replace_strict_series_basic(
    constructor_eager: ConstructorEager,
    old: Sequence[Any] | Mapping[Any, Any],
    new: Sequence[Any] | None,
    return_dtype: DType | None,
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    result = df["a"].replace_strict(old, new, return_dtype=return_dtype)
    assert_equal_data({"a": result}, {"a": ["one", "two", "three"]})


def test_replace_strict_non_full(constructor: Constructor) -> None:
    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    expr = nw.col("a").replace_strict([1, 3], [3, 4], return_dtype=nw.Int64)
    if isinstance(df, nw.LazyFrame):
        # NOTE: non-lazy polars backends raise ValueError since `default=no_default`
        with pytest.raises((ValueError, InvalidOperationError)):
            df.select(expr).collect()
    else:
        with pytest.raises((ValueError, InvalidOperationError)):
            df.select(expr)


def test_replace_strict_invalid_expr(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    msg = "`new` argument is required if `old` argument is not a Mapping type"
    with pytest.raises(TypeError, match=msg):
        df.select(nw.col("a").replace_strict(old=[1, 2, 3]))


def test_replace_strict_invalid_series(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))

    msg = "`new` argument is required if `old` argument is not a Mapping type"
    with pytest.raises(TypeError, match=msg):
        df["a"].replace_strict([1, 2, 3])


def test_replace_strict_pandas_unnamed_series() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    ser = nw.from_native(pd.Series([1, 2, 3]), series_only=True)
    result = ser.replace_strict([1, 2, 3], ["one", "two", "three"])
    assert result.name is None


@pytest.mark.parametrize("return_dtype", [nw.String(), None])
def test_replace_strict_expr_with_default(
    constructor: Constructor, request: pytest.FixtureRequest, return_dtype: DType | None
) -> None:
    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    if "sqlframe" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason=sqlframe_xfail_reason))

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
        pytest.skip(reason=pl_skip_reason)

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
    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    if "sqlframe" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason=sqlframe_xfail_reason))

    df = nw.from_native(constructor({"a": [1, 2, None, 4]}))
    result = df.select(
        nw.col("a").replace_strict([1, 2], [10, 20], default=99, return_dtype=nw.Int64)
    )
    assert_equal_data(result, {"a": [10, 20, 99, 99]})


def test_replace_strict_with_default_mapping(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    if "sqlframe" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason=sqlframe_xfail_reason))

    df = nw.from_native(constructor({"a": [1, 2, 3, 4]}))
    result = df.select(
        nw.col("a").replace_strict(
            {1: "one", 2: "two", 3: None}, default="other", return_dtype=nw.String()
        )
    )
    assert_equal_data(result, {"a": ["one", "two", None, "other"]})


def test_replace_strict_with_expressified_default(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    if "sqlframe" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason=sqlframe_xfail_reason))

    data = {"a": [1, 2, 3, 4], "b": ["beluga", "narwhal", "orca", "vaquita"]}
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("a").replace_strict(
            {1: "one", 2: "two"}, default=nw.col("b"), return_dtype=nw.String
        )
    )

    assert_equal_data(result, {"a": ["one", "two", "orca", "vaquita"]})


def test_replace_strict_with_series_default(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    data = {"a": [1, 2, 3, 4], "b": ["beluga", "narwhal", "orca", "vaquita"]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    series, default = df["a"], df["b"]
    result = series.replace_strict(
        {1: "one", 2: "two"}, default=default, return_dtype=nw.String
    )

    assert_equal_data({"a": result}, {"a": ["one", "two", "orca", "vaquita"]})


def test_mapping_key_not_in_expr(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    if "sqlframe" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason=sqlframe_xfail_reason))

    data = {"a": [1, 2]}
    df = nw.from_native(constructor(data))

    result = df.select(
        nw.col("a").replace_strict({1: "one", 2: "two", 3: "three"}, default="hundred")
    )
    assert_equal_data(result, {"a": ["one", "two"]})


def test_mapping_key_not_in_series(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    data = {"a": [1, 2]}
    df = nw.from_native(constructor_eager(data))

    result = df["a"].replace_strict({1: "one", 2: "two", 3: "three"})
    assert_equal_data({"a": result}, {"a": ["one", "two"]})
