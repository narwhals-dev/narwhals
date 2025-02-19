from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from typing_extensions import reveal_type

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION
from tests.utils import POLARS_VERSION
from tests.utils import PYARROW_VERSION

if TYPE_CHECKING:
    from narwhals.typing import IntoSeries
    from tests.utils import Constructor


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
        nw.Datetime(time_unit=time_unit)  # type: ignore[call-overload]


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
    assert dtype != nw.Array(nw.Int64, 3)
    assert dtype != nw.Array(nw.Float32, 2)
    assert dtype != nw.Duration
    assert repr(dtype) == "Array(<class 'narwhals.dtypes.Int64'>, shape=(2,))"
    dtype = nw.Array(nw.Array(nw.Int64, 2), 2)
    assert dtype == nw.Array(nw.Array(nw.Int64, 2), 2)
    assert dtype == nw.Array
    assert dtype != nw.Array(nw.Array(nw.Float32, 2), 2)
    assert dtype in {nw.Array(nw.Array(nw.Int64, 2), 2)}

    with pytest.raises(TypeError, match="invalid input for shape"):
        nw.Array(nw.Int64(), shape=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="invalid input for shape"):
        nw.Array(nw.Int64(), shape="invalid_type")  # type: ignore[arg-type]


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


@pytest.mark.skipif(PANDAS_VERSION < (2, 2), reason="old pandas")
def test_2d_array(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if any(x in str(constructor) for x in ("dask", "modin", "cudf", "pyspark")):
        request.applymarker(pytest.mark.xfail)
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (14,):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [[[1, 2], [3, 4], [5, 6]]]}
    df = nw.from_native(constructor(data)).with_columns(
        a=nw.col("a").cast(nw.Array(nw.Int64(), (3, 2)))
    )
    assert df.collect_schema()["a"] == nw.Array(nw.Int64(), (3, 2))
    assert df.collect_schema()["a"] == nw.Array(nw.Array(nw.Int64(), 2), 3)


def test_second_time_unit() -> None:
    s: IntoSeries = pd.Series(np.array([np.datetime64("2020-01-01", "s")]))
    result = nw.from_native(s, series_only=True)
    if PANDAS_VERSION < (2,):  # pragma: no cover
        assert result.dtype == nw.Datetime("ns")
    else:
        assert result.dtype == nw.Datetime("s")
    ts_sec = pa.timestamp("s")
    dur_sec = pa.duration("s")
    s = pa.chunked_array([pa.array([datetime(2020, 1, 1)], type=ts_sec)], type=ts_sec)
    result = nw.from_native(s, series_only=True)
    assert result.dtype == nw.Datetime("s")
    s = pd.Series(np.array([np.timedelta64(1, "s")]))
    result = nw.from_native(s, series_only=True)
    if PANDAS_VERSION < (2,):  # pragma: no cover
        assert result.dtype == nw.Duration("ns")
    else:
        assert result.dtype == nw.Duration("s")
    s = pa.chunked_array([pa.array([timedelta(1)], type=dur_sec)], type=dur_sec)
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
    duckdb = pytest.importorskip("duckdb")
    df = pl.DataFrame({"a": [1, 2, 3]})
    if POLARS_VERSION >= (1, 18):  # pragma: no cover
        result = nw.from_native(df.select(pl.col("a").cast(pl.Int128))).schema
        assert result["a"] == nw.Int128
    else:  # pragma: no cover
        # Int128 was not available yet
        pass
    rel = duckdb.sql("""
        select cast(a as int128) as a
        from df
                     """)
    result = nw.from_native(rel).schema
    assert result["a"] == nw.Int128
    rel = duckdb.sql("""
        select cast(a as uint128) as a
        from df
                     """)
    result = nw.from_native(rel).schema
    assert result["a"] == nw.UInt128

    # TODO(unassigned): once other libraries support Int128/UInt128,
    # add tests for them too


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_decimal() -> None:
    duckdb = pytest.importorskip("duckdb")
    df = pl.DataFrame({"a": [1]}, schema={"a": pl.Decimal})
    result = nw.from_native(df).schema
    assert result["a"] == nw.Decimal
    rel = duckdb.sql("""
        select *
        from df
                     """)
    result = nw.from_native(rel).schema
    assert result["a"] == nw.Decimal
    result = nw.from_native(df.to_pandas(use_pyarrow_extension_array=True)).schema
    assert result["a"] == nw.Decimal
    result = nw.from_native(df.to_arrow()).schema
    assert result["a"] == nw.Decimal


def test_dtype_is_x() -> None:
    dtypes = (
        nw.Array,
        nw.Boolean,
        nw.Categorical,
        nw.Date,
        nw.Datetime,
        nw.Decimal,
        nw.Duration,
        nw.Enum,
        nw.Float32,
        nw.Float64,
        nw.Int8,
        nw.Int16,
        nw.Int32,
        nw.Int64,
        nw.Int128,
        nw.List,
        nw.Object,
        nw.String,
        nw.Struct,
        nw.UInt8,
        nw.UInt16,
        nw.UInt32,
        nw.UInt64,
        nw.UInt128,
        nw.Unknown,
    )

    is_signed_integer = {nw.Int8, nw.Int16, nw.Int32, nw.Int64, nw.Int128}
    is_unsigned_integer = {nw.UInt8, nw.UInt16, nw.UInt32, nw.UInt64, nw.UInt128}
    is_float = {nw.Float32, nw.Float64}
    is_decimal = {nw.Decimal}
    is_temporal = {nw.Datetime, nw.Date, nw.Duration}
    is_nested = {nw.Array, nw.List, nw.Struct}

    for dtype in dtypes:
        assert dtype.is_numeric() == (
            dtype
            in is_signed_integer.union(is_unsigned_integer)
            .union(is_float)
            .union(is_decimal)
        )
        assert dtype.is_integer() == (
            dtype in is_signed_integer.union(is_unsigned_integer)
        )
        assert dtype.is_signed_integer() == (dtype in is_signed_integer)
        assert dtype.is_unsigned_integer() == (dtype in is_unsigned_integer)
        assert dtype.is_float() == (dtype in is_float)
        assert dtype.is_decimal() == (dtype in is_decimal)
        assert dtype.is_temporal() == (dtype in is_temporal)
        assert dtype.is_nested() == (dtype in is_nested)


def test_huge_int_to_native() -> None:
    duckdb = pytest.importorskip("duckdb")
    df = pl.DataFrame({"a": [1, 2, 3]})
    if POLARS_VERSION >= (1, 18):  # pragma: no cover
        df_casted = (
            nw.from_native(df)
            .with_columns(a_int=nw.col("a").cast(nw.Int128()))
            .to_native()
        )
        assert df_casted.schema["a_int"] == pl.Int128
    else:  # pragma: no cover
        # Int128 was not available yet
        pass
    rel = duckdb.sql("""
        select cast(a as int64) as a
        from df
                     """)
    result = (
        nw.from_native(rel)
        .with_columns(
            a_int=nw.col("a").cast(nw.Int128()), a_unit=nw.col("a").cast(nw.UInt128())
        )
        .select("a_int", "a_unit")
        .to_native()
    )
    type_a_int, type_a_unit = result.types
    assert type_a_int == "HUGEINT"
    assert type_a_unit == "UHUGEINT"


def test_cast_decimal_to_native() -> None:
    duckdb = pytest.importorskip("duckdb")
    data = {"a": [1, 2, 3]}

    df = pl.DataFrame(data)
    library_obj_to_test = [
        df,
        duckdb.sql("""
            select cast(a as INT1) as a
            from df
                         """),
        pd.DataFrame(data),
        pa.Table.from_arrays(
            [pa.array(data["a"])], schema=pa.schema([("a", pa.int64())])
        ),
    ]
    for obj in library_obj_to_test:
        with pytest.raises(
            NotImplementedError, match="Casting to Decimal is not supported yet."
        ):
            (
                nw.from_native(obj)
                .with_columns(a=nw.col("a").cast(nw.Decimal()))
                .to_native()
            )


def test_datetime_generic() -> None:
    import narwhals as unstable_nw

    dt_1 = unstable_nw.Datetime()
    dt_21 = unstable_nw.Datetime("ns")
    dt_22 = unstable_nw.Datetime(time_unit="ns")
    dt_3 = unstable_nw.Datetime("s", time_zone="zone")
    dt_4 = unstable_nw.Datetime("ns", timezone.utc)
    dt_5 = unstable_nw.Datetime(time_zone="Asia/Kathmandu")
    dt_6 = unstable_nw.Datetime(time_zone=timezone.utc)
    reveal_type(dt_1)
    reveal_type(dt_21)
    reveal_type(dt_22)
    reveal_type(dt_3)
    reveal_type(dt_4)
    reveal_type(dt_5)
    reveal_type(dt_6)
    reveal_type(dt_3.time_unit)
    assert dt_3.time_unit
