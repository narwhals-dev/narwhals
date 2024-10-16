from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version


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
    dtype = nw.Struct(nw.Struct([nw.Field("a", nw.Int64), nw.Field("b", nw.String)]))
    assert dtype == nw.Struct(
        nw.Struct([nw.Field("a", nw.Int64), nw.Field("b", nw.String)])
    )
    assert dtype == nw.Struct
    assert dtype != nw.Struct(
        nw.Struct([nw.Field("a", nw.Float32), nw.Field("b", nw.String)])
    )
    assert dtype in {
        nw.Struct(nw.Struct([nw.Field("a", nw.Int64), nw.Field("b", nw.String)]))
    }
    dtype = nw.Struct({"a": {"b": 1}})
    assert dtype == nw.Struct({"a": {"b": 1}})


def test_struct_reverse() -> None:
    dtype = nw.Struct({"a": {"b": 1}})

    (next(reversed(dtype)))


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
    parse_version(pl.__version__) < (1,) or parse_version(pd.__version__) < (2, 2),
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
    if parse_version(pd.__version__) < (2,):  # pragma: no cover
        assert result.dtype == nw.Datetime("ns")
    else:
        assert result.dtype == nw.Datetime("s")
    s = pa.chunked_array([pa.array([datetime(2020, 1, 1)], type=pa.timestamp("s"))])
    result = nw.from_native(s, series_only=True)
    assert result.dtype == nw.Datetime("s")
    s = pd.Series(np.array([np.timedelta64(1, "s")]))
    result = nw.from_native(s, series_only=True)
    if parse_version(pd.__version__) < (2,):  # pragma: no cover
        assert result.dtype == nw.Duration("ns")
    else:
        assert result.dtype == nw.Duration("s")
    s = pa.chunked_array([pa.array([timedelta(1)], type=pa.duration("s"))])
    result = nw.from_native(s, series_only=True)
    assert result.dtype == nw.Duration("s")
