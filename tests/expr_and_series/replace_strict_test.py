from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.conftest import dask_pyarrow_constructor
from tests.utils import (
    DASK_PYARROW_DTYPES,
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    maybe_collect,
    xfail_if_builtin_dask,
    xfail_if_pyspark_connect,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from narwhals.dtypes import DType

polars_lt_v1 = POLARS_VERSION < PkgVersion("1")
skip_reason = "replace_strict only available after 1.0"
pl_skip_reason = "replace_strict only available after 1.0"


def xfail_if_no_default(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    lazy_non_polars_constructors = ("dask", "duckdb", "ibis", "pyspark", "sqlframe")
    if any(x in str(constructor) for x in lazy_non_polars_constructors):
        reason = "non-polars lazy backends require default parameter to be provided"
        request.applymarker(pytest.mark.xfail(reason=reason))


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
    spark_connect_reason = (
        "`mapping_expr[expr]` raises: pyspark.errors.exceptions.base.PySparkTypeError: "
        "[UNSUPPORTED_DATA_TYPE] Unsupported DataType `Column`."
    )
    xfail_if_pyspark_connect(constructor, request, reason=spark_connect_reason)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df = nw.from_native(constructor({"a": [1, 2, 3, 4]}))
    xfail_if_builtin_dask(df, request, when=DASK_PYARROW_DTYPES)
    result = df.select(
        nw.col("a").replace_strict(
            [1, 2], ["one", "two"], default="other", return_dtype=return_dtype
        )
    )
    assert_equal_data(result, {"a": ["one", "two", "other", "other"]})


@pytest.mark.parametrize(
    ("mapping", "default", "dtype", "expected"),
    [
        ({1: 10}, 0, nw.Int64(), [10, 0, 0]),
        ({1: 10}, None, nw.Int64(), [10, None, None]),
        ({None: 99}, -1, nw.Int64(), [-1, -1, 99]),
        ({1.1: 10, 1.9: 20, 2: 30}, -1, nw.Int64(), [-1, 30, -1]),
        ({1: None}, None, nw.Unknown(), [None, None, None]),
        ({1: "one", 2: None}, "other", nw.String(), ["one", None, "other"]),
        ({1: 10}, nw.col("b"), nw.Float64(), [10.0, 2.5, 3.5]),
        ({1: 10}, "other", nw.String(), ["10", "other", "other"]),
        ({1: "one"}, 3, nw.String(), ["one", "3", "3"]),
    ],
    ids=[
        "integer",
        "null-default",
        "null-key",
        "unmatched-fractional-keys",
        "all-null",
        "string",
        "float-default",
        "string-default",
        "integer-default",
    ],
)
def test_replace_strict_dask_arrow_inferred_dtype(
    mapping: dict[float | None, Any],
    default: int | str | nw.Expr | None,
    dtype: DType,
    expected: list[Any],
    request: pytest.FixtureRequest,
) -> None:
    df = nw.from_native(
        dask_pyarrow_constructor({"a": [0, 1, 2, None], "b": [0.0, 1.5, 2.5, 3.5]})
    ).filter(nw.col("b") > 0)
    if dtype != nw.Int64:
        xfail_if_builtin_dask(df, request)

    result = df.select(nw.col("a").replace_strict(mapping, default=default))
    assert result.collect_schema() == {"a": dtype}
    collected = result.collect()
    assert collected.schema == {"a": dtype}
    assert_equal_data(collected, {"a": expected})


@pytest.mark.parametrize(
    ("replacement", "storage", "default", "expected"),
    [
        (9, "Int64", nw.col("b"), [9, 4, None]),
        (None, "Int64", nw.col("b"), [None, 4, None]),
        (None, "int64[pyarrow]", nw.col("b"), [None, 4, None]),
        (None, "Int64", 4, [None, 4, 4]),
    ],
)
def test_replace_strict_dask_arrow_integer_default(
    replacement: int | None,
    storage: str,
    default: nw.Expr | int,
    expected: list[int | None],
    request: pytest.FixtureRequest,
) -> None:
    dd = pytest.importorskip("dask.dataframe")
    pytest.importorskip("pyarrow")
    import pandas as pd

    frame = pd.DataFrame(
        {
            "a": pd.array([1, 2, 3], dtype="int64[pyarrow]"),
            "b": pd.Series([3, 4, None], dtype=storage),
        }
    )
    df = nw.from_native(dd.from_pandas(frame, npartitions=2))
    if replacement is None:
        xfail_if_builtin_dask(df, request)
    result = df.select(nw.col("a").replace_strict({1: replacement}, default=default))
    assert result.collect_schema() == {"a": nw.Int64}
    collected = result.collect()
    assert collected.schema == {"a": nw.Int64}
    assert_equal_data(collected, {"a": expected})


def test_replace_strict_dask_arrow_casts_only_used_defaults(
    request: pytest.FixtureRequest,
) -> None:
    df = nw.from_native(
        dask_pyarrow_constructor({"a": [1, 1, 2], "b": ["bad", "bad", "7"]})
    )
    if PANDAS_VERSION < PkgVersion("2.1"):
        xfail_if_builtin_dask(df, request)
    result = df.select(
        nw.col("a").replace_strict({1: 9}, default=nw.col("b"), return_dtype=nw.Int64)
    )
    assert_equal_data(result, {"a": [9, 9, 7]})


def test_replace_strict_with_grouped_default(arrow_constructor: Constructor) -> None:
    df = nw.from_native(arrow_constructor({"a": [1, 2, 3, 4], "g": ["x", "y"] * 2}))
    expr = nw.col("a").replace_strict({1: 9}, default=nw.col("a").sum().over("g"))
    result = df.select("a", replaced=expr).sort("a")
    assert_equal_data(result, {"a": [1, 2, 3, 4], "replaced": [9, 6, 4, 6]})


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (nw.lit(7, dtype=nw.Int64), nw.col("a").sum(), 6),
        (nw.col("a").sum(), nw.lit(7, dtype=nw.Int64), 7),
    ],
    ids=["literal", "reduction"],
)
def test_replace_strict_scalar_default(
    arrow_constructor: Constructor, value: nw.Expr, default: nw.Expr, expected: int
) -> None:
    df = nw.from_native(arrow_constructor({"a": [1, 2, 3]}))
    result = df.select(
        replaced=value.replace_strict({1: 9}, default=default, return_dtype=nw.Int64)
    )
    assert result.collect_schema() == {"replaced": nw.Int64}
    collected = maybe_collect(result)
    assert collected.schema == {"replaced": nw.Int64}
    assert_equal_data(collected, {"replaced": [expected]})


def test_dask_replace_strict_with_shuffled_duplicate_index(
    request: pytest.FixtureRequest,
) -> None:
    dd = pytest.importorskip("dask.dataframe")
    pytest.importorskip("pyarrow")
    import pandas as pd

    frame = pd.DataFrame(
        {"a": pd.array([1, 2, 3, 4], dtype="int64[pyarrow]"), "g": ["x", "y"] * 2},
        index=[3, 1, 3, 0],
    )
    df = nw.from_native(dd.from_pandas(frame, npartitions=2, sort=False))
    xfail_if_builtin_dask(df, request)
    expr = nw.col("a").replace_strict({1: 9}, default=nw.col("a").sum().over("g"))
    with pytest.raises(ValueError, match="duplicate"):
        df.select(expr).collect()


def test_replace_strict_arrow_large_integer_keys(arrow_constructor: Constructor) -> None:
    key = 2**53
    df = nw.from_native(arrow_constructor({"a": [key, key + 1, None, key + 2]}))
    result = df.select(
        nw.col("a").replace_strict({key: 10, key + 1: 20, None: 30}, default=-1)
    )
    assert_equal_data(result, {"a": [10, 20, 30, -1]})


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
    xfail_if_pyspark_connect(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df = nw.from_native(constructor({"a": [1, 2, None, 4]}))
    result = df.select(
        nw.col("a").replace_strict([1, 2], [10, 20], default=99, return_dtype=nw.Int64)
    )
    assert_equal_data(result, {"a": [10, 20, 99, 99]})


def test_replace_strict_with_default_mapping(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    xfail_if_pyspark_connect(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df = nw.from_native(constructor({"a": [1, 2, 3, 4]}))
    xfail_if_builtin_dask(df, request, when=DASK_PYARROW_DTYPES)
    result = df.select(
        nw.col("a").replace_strict(
            {1: "one", 2: "two", 3: None}, default="other", return_dtype=nw.String()
        )
    )
    assert_equal_data(result, {"a": ["one", "two", None, "other"]})


def test_replace_strict_with_expressified_default(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    xfail_if_pyspark_connect(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    data = {"a": [1, 2, 3, 4], "b": ["beluga", "narwhal", "orca", "vaquita"]}
    df = nw.from_native(constructor(data))
    xfail_if_builtin_dask(df, request, when=DASK_PYARROW_DTYPES)
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
    xfail_if_pyspark_connect(constructor, request)

    if "polars" in str(constructor) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    data = {"a": [1, 2]}
    df = nw.from_native(constructor(data))
    xfail_if_builtin_dask(df, request, when=DASK_PYARROW_DTYPES)

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


def test_replace_strict_dask_arrow_categorical_default() -> None:
    df = nw.from_native(dask_pyarrow_constructor({"a": ["a", "a", "b", "b"]}))
    result = df.select(
        nw.col("a").replace_strict({"a": "x"}, default="y", return_dtype=nw.Categorical)
    )
    assert result.collect_schema() == {"a": nw.Categorical}
    collected = result.collect()
    assert collected.schema == {"a": nw.Categorical}
    assert_equal_data(collected, {"a": ["x", "x", "y", "y"]})
