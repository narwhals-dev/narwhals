from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Literal

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION
from tests.utils import POLARS_VERSION


@pytest.mark.parametrize("time_unit", ["us", "ns", "ms"])
@pytest.mark.parametrize("time_zone", ["Europe/Rome", timezone.utc, None])
def test_datetime_valid(
    time_unit: Literal["us", "ns", "ms"], time_zone: str | timezone | None
) -> None:
    dtype = nw.Datetime(time_unit=time_unit, time_zone=time_zone)

    assert dtype == nw.Datetime(time_unit=time_unit, time_zone=time_zone)
    assert dtype == nw.Datetime

    if time_zone:
        assert dtype != nw.Datetime(time_unit=time_unit)
    if time_unit != "ms":
        assert dtype != nw.Datetime(time_unit="ms")


@pytest.mark.parametrize("time_unit", ["abc"])
def test_datetime_invalid(time_unit: str) -> None:
    with pytest.raises(ValueError, match="invalid `time_unit`"):
        nw.Datetime(time_unit=time_unit)  # type: ignore[arg-type]


@pytest.mark.parametrize("time_unit", ["us", "ns", "ms"])
def test_duration_valid(time_unit: Literal["us", "ns", "ms"]) -> None:
    dtype = nw.Duration(time_unit=time_unit)

    assert dtype == nw.Duration(time_unit=time_unit)
    assert dtype == nw.Duration

    if time_unit != "ms":
        assert dtype != nw.Duration(time_unit="ms")


@pytest.mark.parametrize("time_unit", ["abc"])
def test_duration_invalid(time_unit: str) -> None:
    with pytest.raises(ValueError, match="invalid `time_unit`"):
        nw.Duration(time_unit=time_unit)  # type: ignore[arg-type]


def test_list_valid() -> None:
    dtype = nw.List(nw.Int64)
    assert dtype == nw.List(nw.Int64)
    assert dtype == nw.List
    assert dtype != nw.List(nw.Float32)
    assert dtype != nw.Duration
    assert repr(dtype) == "List(<class 'narwhals.dtypes.Int64'>)"
    dtype = nw.List(nw.List(nw.Int64))
    assert dtype == nw.List(nw.List(nw.Int64))
    assert dtype == nw.List
    assert dtype != nw.List(nw.List(nw.Float32))
    assert dtype in {nw.List(nw.List(nw.Int64))}


def test_array_valid() -> None:
    dtype = nw.Array(nw.Int64, 2)
    assert dtype == nw.Array(nw.Int64, 2)
    assert dtype == nw.Array
    assert dtype != nw.Array(nw.Float32, 2)
    assert dtype != nw.Duration
    assert repr(dtype) == "Array(<class 'narwhals.dtypes.Int64'>, 2)"
    dtype = nw.Array(nw.Array(nw.Int64, 2), 2)
    assert dtype == nw.Array(nw.Array(nw.Int64, 2), 2)
    assert dtype == nw.Array
    assert dtype != nw.Array(nw.Array(nw.Float32, 2), 2)
    assert dtype in {nw.Array(nw.Array(nw.Int64, 2), 2)}

    with pytest.raises(
        TypeError, match="`width` must be specified when initializing an `Array`"
    ):
        dtype = nw.Array(nw.Int64)


def test_struct_valid() -> None:
    dtype = nw.Struct([nw.Field("a", nw.Int64)])
    assert dtype == nw.Struct([nw.Field("a", nw.Int64)])
    assert dtype == nw.Struct
    assert dtype != nw.Struct([nw.Field("a", nw.Float32)])
    assert dtype != nw.Duration
    assert repr(dtype) == "Struct({'a': <class 'narwhals.dtypes.Int64'>})"

    dtype = nw.Struct({"a": nw.Int64, "b": nw.String})
    assert dtype == nw.Struct({"a": nw.Int64, "b": nw.String})
    assert dtype.to_schema() == nw.Struct({"a": nw.Int64, "b": nw.String}).to_schema()
    assert dtype == nw.Struct
    assert dtype != nw.Struct({"a": nw.Int32, "b": nw.String})
    assert dtype in {nw.Struct({"a": nw.Int64, "b": nw.String})}


def test_struct_reverse() -> None:
    dtype1 = nw.Struct({"a": nw.Int64, "b": nw.String})
    dtype1_reversed = nw.Struct([nw.Field(*field) for field in reversed(dtype1)])
    dtype2 = nw.Struct({"b": nw.String, "a": nw.Int64})
    assert dtype1_reversed == dtype2


def test_field_repr() -> None:
    dtype = nw.Field("a", nw.Int32)
    assert repr(dtype) == "Field('a', <class 'narwhals.dtypes.Int32'>)"


def test_struct_hashes() -> None:
    dtypes = (
        nw.Struct,
        nw.Struct([nw.Field("a", nw.Int64)]),
        nw.Struct([nw.Field("a", nw.Int64), nw.Field("b", nw.List(nw.Int64))]),
    )
    assert len({hash(tp) for tp in (dtypes)}) == 3


@pytest.mark.skipif(
    POLARS_VERSION < (1,) or PANDAS_VERSION < (2, 2),
    reason="`shape` is only available after 1.0",
)
def test_polars_2d_array() -> None:
    df = pl.DataFrame(
        {"a": [[[1, 2], [3, 4], [5, 6]]]}, schema={"a": pl.Array(pl.Int64, (3, 2))}
    )
    assert nw.from_native(df).collect_schema()["a"] == nw.Array(nw.Array(nw.Int64, 2), 3)
    assert nw.from_native(df.to_arrow()).collect_schema()["a"] == nw.Array(
        nw.Array(nw.Int64, 2), 3
    )
    assert nw.from_native(
        df.to_pandas(use_pyarrow_extension_array=True)
    ).collect_schema()["a"] == nw.Array(nw.Array(nw.Int64, 2), 3)


def test_second_time_unit() -> None:
    s = pd.Series(np.array([np.datetime64("2020-01-01", "s")]))
    result = nw.from_native(s, series_only=True)
    if PANDAS_VERSION < (2,):  # pragma: no cover
        assert result.dtype == nw.Datetime("ns")
    else:
        assert result.dtype == nw.Datetime("s")
    s = pa.chunked_array([pa.array([datetime(2020, 1, 1)], type=pa.timestamp("s"))])
    result = nw.from_native(s, series_only=True)
    assert result.dtype == nw.Datetime("s")
    s = pd.Series(np.array([np.timedelta64(1, "s")]))
    result = nw.from_native(s, series_only=True)
    if PANDAS_VERSION < (2,):  # pragma: no cover
        assert result.dtype == nw.Duration("ns")
    else:
        assert result.dtype == nw.Duration("s")
    s = pa.chunked_array([pa.array([timedelta(1)], type=pa.duration("s"))])
    result = nw.from_native(s, series_only=True)
    assert result.dtype == nw.Duration("s")


@pytest.mark.filterwarnings("ignore:Setting an item of incompatible")
def test_pandas_inplace_modification_1267(request: pytest.FixtureRequest) -> None:
    if PANDAS_VERSION >= (3,):
        # pandas 3.0+ won't allow this kind of inplace modification
        request.applymarker(pytest.mark.xfail)
    if PANDAS_VERSION < (1, 4):
        # pandas pre 1.4 wouldn't change the type?
        request.applymarker(pytest.mark.xfail)
    s = pd.Series([1, 2, 3])
    snw = nw.from_native(s, series_only=True)
    assert snw.dtype == nw.Int64
    s[0] = 999.5
    assert snw.dtype == nw.Float64


def test_pandas_fixed_offset_1302() -> None:
    result = nw.from_native(
        pd.Series(pd.to_datetime(["2020-01-01T00:00:00.000000000+01:00"])),
        series_only=True,
    ).dtype
    if PANDAS_VERSION >= (2,):
        assert result == nw.Datetime("ns", "UTC+01:00")
    else:  # pragma: no cover
        assert result == nw.Datetime("ns", "pytz.FixedOffset(60)")
    if PANDAS_VERSION >= (2,):
        result = nw.from_native(
            pd.Series(
                pd.to_datetime(["2020-01-01T00:00:00.000000000+01:00"])
            ).convert_dtypes(dtype_backend="pyarrow"),
            series_only=True,
        ).dtype
        assert result == nw.Datetime("ns", "+01:00")
    else:  # pragma: no cover
        pass


def test_huge_int() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})  # noqa: F841
    rel = duckdb.sql("""
        select cast(a as int128) as a
        from df
                     """)
    result = nw.from_native(rel).schema
    assert result["a"] == nw.Int128
