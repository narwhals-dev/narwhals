from __future__ import annotations

from typing import Any

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
import narwhals.stable.v1 as nw_v1
import narwhals.stable.v2 as nw_v2
from narwhals.exceptions import (
    ComputeError,
    DuplicateError,
    InvalidOperationError,
    ShapeError,
)
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

inf = float("inf")


def test_bin_intervals_expr(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(
        x in str(constructor) for x in ("dask", "duckdb", "ibis", "pyspark", "sqlframe")
    ):
        reason = "`bin_intervals` is not implemented for Dask and the SQL backends"
        request.applymarker(pytest.mark.xfail(reason=reason, raises=NotImplementedError))
    df = nw.from_native(constructor({"a": [-inf, -1.0, 0.0, 0.5, 2.0, 3.0, inf]}))
    a = nw.col("a")
    result = df.select(
        a.bin_intervals([0, 2], labels=["low", "mid", "high"]),
        index=a.bin_intervals([0, 2], labels=False, right_closed=True),
    )
    assert_equal_data(
        result,
        {
            "a": ["low", "low", "mid", "mid", "high", "high", "high"],
            "index": [0, 0, 0, 1, 1, 2, 2],
        },
    )
    assert result.collect_schema()["index"] == nw.UInt32


def test_bin_intervals_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager({"a": [1, 2, 2]}), eager_only=True)["a"]
    result = series.bin_intervals(
        [1, 2.5], labels=["low", "mid", "high"], right_closed=True
    )
    assert_equal_data({"a": result}, {"a": ["low", "mid", "mid"]})
    if series.implementation.is_pyarrow():
        assert result.dtype == nw.Categorical
    else:
        assert result.dtype == nw.Enum(["low", "mid", "high"])

    indices = series.bin_intervals([], labels=False)
    assert_equal_data({"a": indices}, {"a": [0, 0, 0]})
    assert indices.dtype == nw.UInt32
    assert series.head(0).bin_intervals([], labels=["all"]).len() == 0

    unsigned = series.cast(nw.UInt64)
    assert_equal_data(
        {"a": unsigned.bin_intervals([-1, 1.5], labels=False)}, {"a": [1, 2, 2]}
    )

    if series.implementation.is_pandas_like() or series.implementation.is_pyarrow():
        with pytest.raises(InvalidOperationError):
            series.cast(nw.String).bin_intervals([1], labels=False)


def test_bin_intervals_large_integers(constructor_eager: ConstructorEager) -> None:
    edge = 2**53 + 1
    series = nw.from_native(
        constructor_eager({"a": [edge - 1, edge, edge + 1]}), eager_only=True
    )["a"]
    assert_equal_data(
        {
            "left": series.bin_intervals([edge - 1, edge], labels=False),
            "mixed": series.bin_intervals([0.5, edge], labels=["low", "mid", "high"]),
            "right": series.bin_intervals(
                [edge - 1, edge], labels=False, right_closed=True
            ),
        },
        {"left": [1, 2, 2], "mixed": ["mid", "high", "high"], "right": [0, 1, 2]},
    )


@pytest.mark.parametrize(
    "backend",
    [
        "polars",
        "pyarrow",
        "pandas",
        "pandas-nullable",
        pytest.param(
            "pandas-arrow",
            marks=pytest.mark.skipif(
                PANDAS_VERSION < PkgVersion("2"), reason="requires pandas Arrow dtypes"
            ),
        ),
    ],
)
def test_bin_intervals_nan_and_null(backend: str) -> None:
    values = [None, float("nan"), -inf, inf]
    if backend == "polars":
        pl = pytest.importorskip("polars")
        series = nw.from_native(pl.Series("a", values), series_only=True)
    elif backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        series = nw.from_native(pa.chunked_array([values]), series_only=True)
    else:
        pd = pytest.importorskip("pandas")
        if backend == "pandas-nullable":
            np = pytest.importorskip("numpy")
            array = pd.arrays.FloatingArray(
                np.array(values, dtype=float), np.array([True, False, False, False])
            )
        elif backend == "pandas-arrow":
            pa = pytest.importorskip("pyarrow")
            array = pd.arrays.ArrowExtensionArray(pa.array(values))
        else:
            array = values
        series = nw.from_native(
            pd.Series(array, name="a", index=[8, 6, 4, 2]), series_only=True
        )
    # A NumPy-backed float column has already converted the first value to NaN.
    missing = 1 if backend == "pandas" else None
    result = series.bin_intervals([0], labels=False)
    assert_equal_data({"a": result}, {"a": [missing, 1, 0, 1]})
    labels = series.bin_intervals([0], labels=["lo", "hi"], right_closed=True)
    assert_equal_data(
        {"a": labels}, {"a": ["hi" if missing == 1 else None, "hi", "lo", "hi"]}
    )


@pytest.mark.parametrize(
    ("intervals", "labels", "exception", "match"),
    [
        ([1, 1], False, DuplicateError, "intervals are not unique"),
        ([2, 1], False, ComputeError, "intervals must increase monotonically"),
        (
            [float("nan")],
            False,
            ComputeError,
            "intervals must not contain NaN or infinity",
        ),
        ([0, inf], False, ComputeError, "intervals must not contain NaN or infinity"),
        ([1], ["a"], ShapeError, r"expected 2 labels \(len\(intervals\) \+ 1\), got 1"),
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


@pytest.mark.parametrize("stable", [nw_v1, nw_v2])
def test_bin_intervals_not_in_stable(stable: Any) -> None:
    with pytest.raises(NotImplementedError, match="unstable"):
        stable.col("a").bin_intervals([0], labels=False)
    pa = pytest.importorskip("pyarrow")
    series = stable.from_native(pa.chunked_array([[1.0]]), series_only=True)
    with pytest.raises(NotImplementedError, match="unstable"):
        series.bin_intervals([0], labels=False)


def test_bin_intervals_signed_breaks_on_uint64() -> None:
    pl = pytest.importorskip("polars")
    edge = 2**63 + 1
    series = nw.from_native(
        pl.Series("a", [edge - 1, edge, edge + 1], dtype=pl.UInt64), series_only=True
    )
    assert series.bin_intervals([-1, edge], labels=False).to_list() == [1, 2, 2]
    assert series.bin_intervals([-1, 0.5, edge], labels=False).to_list() == [2, 3, 3]
