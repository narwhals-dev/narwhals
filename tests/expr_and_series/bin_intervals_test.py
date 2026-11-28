# Numeric binning cases adapted from Polars 2.1.0, under its MIT license:
# https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/LICENSE
# Copyright (c) 2025 Ritchie Vink
# Copyright (c) 2024 (Some portions) NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/py-polars/tests/unit/operations/test_bin_intervals.py
# https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/py-polars/tests/unit/operations/test_binning.py

from __future__ import annotations

from sys import float_info
from typing import Any, Literal

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from narwhals.exceptions import (
    ComputeError,
    DuplicateError,
    InvalidOperationError,
    ShapeError,
)
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    assert_equal_series,
)

inf = float("inf")
wide = 2**53 + 1
requires_polars_0208 = pytest.mark.xfail(
    POLARS_VERSION < PkgVersion("0.20.8"),
    reason="bin_intervals requires Polars 0.20.8",
    raises=NotImplementedError,
)
polars_precision = pytest.mark.xfail(
    POLARS_VERSION >= PkgVersion("0.20.8"),
    reason="Polars compares breakpoints as Float64",
    raises=(AssertionError, DuplicateError, OverflowError),
)


def xfail_polars(
    constructor: Constructor, request: pytest.FixtureRequest, *, precision: bool = False
) -> None:
    if "polars" in str(constructor):
        request.applymarker(requires_polars_0208)
        if precision:
            request.applymarker(polars_precision)


def test_bin_intervals_expr(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    xfail_polars(constructor, request)
    if any(
        x in str(constructor) for x in ("dask", "duckdb", "ibis", "pyspark", "sqlframe")
    ):
        reason = "`bin_intervals` is not implemented for Dask and the SQL backends"
        request.applymarker(pytest.mark.xfail(reason=reason, raises=NotImplementedError))
    df = nw.from_native(constructor({"a": [-inf, 0.0, 1.0, 2.0, inf]}))
    a = nw.col("a")
    labels = ["lo", "mid", "hi"]
    result = df.select(
        a.bin_intervals([0, 2], labels=labels),
        index=a.bin_intervals([0, 2], labels=False, right_closed=True),
    )
    label_dtype = nw.Categorical if df.implementation.is_pyarrow() else nw.Enum(labels)
    schema = result.collect_schema()
    assert schema == {"a": label_dtype, "index": nw.UInt32}
    assert_equal_data(
        result, {"a": ["lo", "mid", "mid", "hi", "hi"], "index": [0, 0, 1, 1, 2]}
    )


def test_bin_intervals_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    xfail_polars(constructor_eager, request)
    data = {"a": [-2, -1, 0, 1, 2]}
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    labels = ["a", "b", "c"]
    result = series.bin_intervals([-1, 1], labels=labels)
    assert result.dtype == (
        nw.Categorical if series.implementation.is_pyarrow() else nw.Enum(labels)
    )
    assert_equal_series(result, ["a", "b", "b", "c", "c"], "a")
    assert_equal_series(series.bin_intervals([], labels=["only"]), ["only"] * 5, "a")
    assert_equal_series(series.head(0).bin_intervals([], labels=["only"]), [], "a")


@pytest.mark.filterwarnings("ignore:Found complex group-by:UserWarning")
def test_bin_intervals_group_by(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    xfail_polars(constructor_eager, request)
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(
            pytest.mark.xfail(reason="non-elementary agg", raises=ValueError)
        )
    df = nw.from_native(
        constructor_eager({"g": [1, 1, 2], "a": [-1.0, 0.5, 2.0]}), eager_only=True
    )
    result = df.group_by("g").agg(nw.col("a").bin_intervals([0, 1], labels=False).max())
    assert_equal_data(result.sort("g"), {"g": [1, 2], "a": [1, 2]})


@pytest.mark.parametrize(
    ("dtype", "breaks", "expected"),
    [
        pytest.param(nw.Int8, [1.5], [0, 1, 1], id="int8"),
        pytest.param(nw.UInt64, [-1, 1.5], [1, 2, 2], id="uint64"),
        pytest.param(nw.Float32, [2.0], [0, 1, 1], id="float32"),
        pytest.param(nw.Int64, [], [0, 0, 0], id="single-bin"),
    ],
)
def test_bin_intervals_numeric_breakpoints(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    dtype: type[nw.dtypes.NumericType],
    breaks: list[float],
    expected: list[int],
) -> None:
    xfail_polars(constructor_eager, request)
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.select(nw.col("a").cast(dtype).bin_intervals(breaks, labels=False))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("breaks", "right_closed", "labels", "expected"),
    [
        pytest.param([wide - 1, wide], False, False, [1, 2, 2], id="adjacent-left"),
        pytest.param([wide - 1, wide], True, False, [0, 1, 2], id="adjacent-right"),
        pytest.param([0.5, wide], False, ["a", "b", "c"], ["b", "c", "c"], id="mixed"),
        pytest.param([float(wide - 1)], True, False, [0, 1, 1], id="float-edge"),
    ],
)
def test_bin_intervals_large_integers(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    breaks: list[float],
    *,
    right_closed: bool,
    labels: list[str] | Literal[False],
    expected: list[int] | list[str],
) -> None:
    xfail_polars(constructor_eager, request, precision=True)
    data = {"a": [wide - 1, wide, wide + 1]}
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.bin_intervals(breaks, labels=labels, right_closed=right_closed)
    assert_equal_series(result, expected, "a")


def test_bin_intervals_wide_breaks_on_floats(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    xfail_polars(constructor_eager, request, precision=True)
    data = {"a": [float(wide - 1), float(wide + 1), float(wide + 3)]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    a = nw.col("a")
    breaks = [wide, wide + 2]
    result = df.select(
        left=a.bin_intervals(breaks, labels=False),
        right=a.bin_intervals(breaks, labels=False, right_closed=True),
    )
    assert_equal_data(result, {"left": [0, 1, 2], "right": [0, 1, 2]})


def test_bin_intervals_breaks_outside_float64(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    xfail_polars(constructor_eager, request, precision=True)
    data = {"i": [0, 1, 2, 3], "f": [-inf, -float_info.max, float_info.max, inf]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    huge = 10**400
    result = df.select(
        integer=nw.col("i").bin_intervals([huge], labels=False),
        left=nw.col("f").bin_intervals([-huge, huge], labels=False),
        right=nw.col("f").bin_intervals([-huge, huge], labels=False, right_closed=True),
    )
    assert_equal_data(
        result, {"integer": [0, 0, 0, 0], "left": [0, 1, 1, 2], "right": [0, 1, 1, 2]}
    )


def test_bin_intervals_nan_and_null(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    xfail_polars(constructor_eager, request)
    data = {"a": [float("nan"), None, -1.0]}
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    expected = [None, None, 0] if series.implementation.is_pandas_like() else [1, None, 0]
    assert_equal_series(series.bin_intervals([0], labels=False), expected, "a")


def test_bin_intervals_pandas_missing_dtype() -> None:
    pd = pytest.importorskip("pandas")
    series = nw.from_native(
        pd.Series([1.0, float("nan")], dtype="float64"), series_only=True
    )
    assert series.bin_intervals([0], labels=False).dtype == nw.Float64


@pytest.mark.parametrize(
    "arrow",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                PANDAS_VERSION < PkgVersion("2"), reason="requires pandas Arrow dtypes"
            ),
        ),
    ],
)
def test_bin_intervals_pandas_unmasked_nan(*, arrow: bool) -> None:
    pd = pytest.importorskip("pandas")
    if arrow:
        pa = pytest.importorskip("pyarrow")
        data = pd.arrays.ArrowExtensionArray(pa.array([float("nan"), None]))
    else:
        np = pytest.importorskip("numpy")
        data = pd.arrays.FloatingArray(
            np.array([float("nan"), 0.0]), np.array([False, True])
        )
    series = nw.from_native(pd.Series(data, name="a", index=[8, 6]), series_only=True)
    result = series.bin_intervals([0], labels=["low", "high"])
    assert_equal_series(result, ["high", None], "a")
    assert result.to_native().index.equals(series.to_native().index)


@pytest.mark.parametrize(
    ("intervals", "labels", "exception", "match"),
    [
        ([1, 1], False, ComputeError, "intervals must be strictly increasing"),
        ([2, 1], False, ComputeError, "intervals must be strictly increasing"),
        ([float("nan")], False, ComputeError, "NaN or infinity"),
        ([inf], False, ComputeError, "NaN or infinity"),
        ([1], ["a"], ShapeError, "expected 2 labels"),
        ([1], ["a", "a"], DuplicateError, "labels are not unique"),
        ([1], True, TypeError, "labels"),
        ([1], None, TypeError, "labels"),
        ([1], "ab", TypeError, "labels"),
        ([1], ["a", 2], TypeError, "labels"),
        (2, False, TypeError, "intervals"),
    ],
)
def test_bin_intervals_invalid_arguments(
    intervals: Any, labels: Any, exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        nw.col("a").bin_intervals(intervals, labels=labels)


def test_bin_intervals_non_numeric() -> None:
    pa = pytest.importorskip("pyarrow")
    series = nw.from_native(pa.chunked_array([["a"]]), series_only=True)
    with pytest.raises(InvalidOperationError):
        series.bin_intervals([1], labels=False)


def test_bin_intervals_arrow_dictionary() -> None:
    pa = pytest.importorskip("pyarrow")
    values = pa.chunked_array([[-1.0, None], [3.0, float("nan")]])
    series = nw.from_native(values, series_only=True)
    labels = ["low", "unused", "high"]
    result = series.bin_intervals([0, 2], labels=labels)
    assert result.to_list() == ["low", None, "high", "high"]
    for chunk in result.to_native().chunks:
        assert chunk.dictionary.to_pylist() == labels


def test_bin_intervals_arrow_to_pandas() -> None:
    pa = pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")
    df = nw.from_native(pa.table({"a": [0, 2, None]}))
    labels = ["low", "unused", "high"]
    result = df.select(nw.col("a").bin_intervals([1, 2], labels=labels)).to_pandas()
    expected = pd.Categorical(["low", "high", None], categories=labels)
    pd.testing.assert_series_equal(result["a"], pd.Series(expected, name="a"))


@requires_polars_0208
@polars_precision
@pytest.mark.parametrize(
    ("prefix", "expected"), [([-1], [1, 2, 2]), ([-1, 0.5], [2, 3, 3])]
)
def test_bin_intervals_signed_breaks_on_uint64(
    prefix: list[float], expected: list[int]
) -> None:
    pl = pytest.importorskip("polars")
    edge = 2**63 + 1
    series = nw.from_native(
        pl.Series("a", [edge - 1, edge, edge + 1], dtype=pl.UInt64), series_only=True
    )
    assert series.bin_intervals([*prefix, edge], labels=False).to_list() == expected


@polars_precision
@pytest.mark.parametrize(
    ("dtype", "lower", "upper", "min_version"),
    [("Int128", -(2**127), 2**127 - 1, "1.18"), ("UInt128", 0, 2**128 - 1, "1.34")],
    ids=["signed", "unsigned"],
)
def test_bin_intervals_128_bit_limits(
    dtype: str, lower: int, upper: int, min_version: str
) -> None:
    pl = pytest.importorskip("polars", minversion=min_version)
    values = pl.Series(
        "a", [lower, lower + 1, upper - 1, upper], dtype=getattr(pl, dtype)
    )
    series = nw.from_native(values, series_only=True)
    result = series.bin_intervals([lower - 1, lower + 1, upper, upper + 1], labels=False)
    assert_equal_series(result, [1, 2, 2, 3], "a")


@requires_polars_0208
@pytest.mark.parametrize(
    "dtype",
    [
        "Float64",
        pytest.param(
            "Null",
            marks=pytest.mark.xfail(
                POLARS_VERSION < PkgVersion("1.44") or POLARS_VERSION >= PkgVersion("2"),
                reason="Polars rejects Null-typed input",
                raises=InvalidOperationError,
            ),
        ),
    ],
)
def test_bin_intervals_polars_null_dtype(dtype: str) -> None:
    pl = pytest.importorskip("polars")
    series = nw.from_native(
        pl.Series("a", [None, None], dtype=getattr(pl, dtype)), series_only=True
    )
    result = series.bin_intervals([wide], labels=["low", "high"])
    assert result.dtype == nw.Enum(["low", "high"])
    assert_equal_series(result, [None, None], "a")


@requires_polars_0208
def test_bin_intervals_polars_native_expr() -> None:
    pl = pytest.importorskip("polars")
    df = nw.from_native(pl.LazyFrame({"a": [None, 0.0, float("nan")], "b": [1, -1, 0]}))
    labels = ["low", "high"]
    result = df.select(
        nw.col("a", "b").bin_intervals([0], labels=labels),
        nw.col("b").alias("index").bin_intervals([0], labels=False, right_closed=True),
    )
    assert result.collect_schema() == {
        "a": nw.Enum(labels),
        "b": nw.Enum(labels),
        "index": nw.UInt32,
    }
    assert_equal_data(
        result,
        {"a": [None, "high", "high"], "b": ["high", "low", "high"], "index": [1, 0, 0]},
    )


@requires_polars_0208
def test_bin_intervals_polars_scalar_expr(request: pytest.FixtureRequest) -> None:
    pl = pytest.importorskip("polars")
    df = nw.from_native(pl.LazyFrame({"g": [0, 1, 0], "a": [2, 0, 3]}))
    bins = nw.col("a").min().bin_intervals([1], labels=False)
    assert_equal_data(df.select("g", bins), {"g": [0, 1, 0], "a": [0, 0, 0]})
    request.applymarker(
        pytest.mark.xfail(
            POLARS_VERSION < PkgVersion("1.35"),
            reason="Polars cut rejects an already aggregated column",
            raises=ComputeError,
        )
    )
    assert_equal_data(df.group_by("g").agg(bins).sort("g"), {"g": [0, 1], "a": [1, 0]})
