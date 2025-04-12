from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw_v1
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
    dtype = nw_v1.Datetime(time_unit=time_unit, time_zone=time_zone)

    assert dtype == nw_v1.Datetime(time_unit=time_unit, time_zone=time_zone)
    assert dtype == nw_v1.Datetime

    if time_zone:
        assert dtype != nw_v1.Datetime(time_unit=time_unit)
    if time_unit != "ms":
        assert dtype != nw_v1.Datetime(time_unit="ms")


@pytest.mark.parametrize("time_unit", ["abc"])
def test_datetime_invalid(time_unit: str) -> None:
    with pytest.raises(ValueError, match="invalid `time_unit`"):
        nw_v1.Datetime(time_unit=time_unit)  # type: ignore[arg-type]


@pytest.mark.parametrize("time_unit", ["us", "ns", "ms"])
def test_duration_valid(time_unit: Literal["us", "ns", "ms"]) -> None:
    dtype = nw_v1.Duration(time_unit=time_unit)

    assert dtype == nw_v1.Duration(time_unit=time_unit)
    assert dtype == nw_v1.Duration

    if time_unit != "ms":
        assert dtype != nw_v1.Duration(time_unit="ms")


@pytest.mark.parametrize("time_unit", ["abc"])
def test_duration_invalid(time_unit: str) -> None:
    with pytest.raises(ValueError, match="invalid `time_unit`"):
        nw_v1.Duration(time_unit=time_unit)  # type: ignore[arg-type]


def test_list_valid() -> None:
    dtype = nw_v1.List(nw_v1.Int64)
    assert dtype == nw_v1.List(nw_v1.Int64)
    assert dtype == nw_v1.List
    assert dtype != nw_v1.List(nw_v1.Float32)
    assert dtype != nw_v1.Duration
    assert repr(dtype) == "List(<class 'narwhals.dtypes.Int64'>)"
    dtype = nw_v1.List(nw_v1.List(nw_v1.Int64))
    assert dtype == nw_v1.List(nw_v1.List(nw_v1.Int64))
    assert dtype == nw_v1.List
    assert dtype != nw_v1.List(nw_v1.List(nw_v1.Float32))
    assert dtype in {nw_v1.List(nw_v1.List(nw_v1.Int64))}


def test_array_valid() -> None:
    dtype = nw_v1.Array(nw_v1.Int64, 2)
    assert dtype == nw_v1.Array(nw_v1.Int64, 2)
    assert dtype == nw_v1.Array
    assert dtype != nw_v1.Array(nw_v1.Int64, 3)
    assert dtype != nw_v1.Array(nw_v1.Float32, 2)
    assert dtype != nw_v1.Duration
    assert repr(dtype) == "Array(<class 'narwhals.dtypes.Int64'>, shape=(2,))"
    dtype = nw_v1.Array(nw_v1.Array(nw_v1.Int64, 2), 2)
    assert dtype == nw_v1.Array(nw_v1.Array(nw_v1.Int64, 2), 2)
    assert dtype == nw_v1.Array
    assert dtype != nw_v1.Array(nw_v1.Array(nw_v1.Float32, 2), 2)
    assert dtype in {nw_v1.Array(nw_v1.Array(nw_v1.Int64, 2), 2)}

    with pytest.raises(TypeError, match="invalid input for shape"):
        nw_v1.Array(nw_v1.Int64(), shape=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="invalid input for shape"):
        nw_v1.Array(nw_v1.Int64(), shape="invalid_type")  # type: ignore[arg-type]


def test_struct_valid() -> None:
    dtype = nw_v1.Struct([nw_v1.Field("a", nw_v1.Int64)])
    assert dtype == nw_v1.Struct([nw_v1.Field("a", nw_v1.Int64)])
    assert dtype == nw_v1.Struct
    assert dtype != nw_v1.Struct([nw_v1.Field("a", nw_v1.Float32)])
    assert dtype != nw_v1.Duration
    assert repr(dtype) == "Struct({'a': <class 'narwhals.dtypes.Int64'>})"

    dtype = nw_v1.Struct({"a": nw_v1.Int64, "b": nw_v1.String})
    assert dtype == nw_v1.Struct({"a": nw_v1.Int64, "b": nw_v1.String})
    assert (
        dtype.to_schema()
        == nw_v1.Struct({"a": nw_v1.Int64, "b": nw_v1.String}).to_schema()
    )
    assert dtype == nw_v1.Struct
    assert dtype != nw_v1.Struct({"a": nw_v1.Int32, "b": nw_v1.String})
    assert dtype in {nw_v1.Struct({"a": nw_v1.Int64, "b": nw_v1.String})}


def test_struct_reverse() -> None:
    dtype1 = nw_v1.Struct({"a": nw_v1.Int64, "b": nw_v1.String})
    dtype1_reversed = nw_v1.Struct([nw_v1.Field(*field) for field in reversed(dtype1)])
    dtype2 = nw_v1.Struct({"b": nw_v1.String, "a": nw_v1.Int64})
    assert dtype1_reversed == dtype2


def test_field_repr() -> None:
    dtype = nw_v1.Field("a", nw_v1.Int32)
    assert repr(dtype) == "Field('a', <class 'narwhals.dtypes.Int32'>)"


def test_struct_hashes() -> None:
    dtypes = (
        nw_v1.Struct,
        nw_v1.Struct([nw_v1.Field("a", nw_v1.Int64)]),
        nw_v1.Struct(
            [nw_v1.Field("a", nw_v1.Int64), nw_v1.Field("b", nw_v1.List(nw_v1.Int64))]
        ),
    )
    assert len({hash(tp) for tp in (dtypes)}) == 3


def test_2d_array(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    version_conditions = [
        (PANDAS_VERSION < (2, 2), "Requires pandas 2.2+ for 2D array support"),
        (
            "pyarrow_table" in str(constructor) and PYARROW_VERSION < (14,),
            "PyArrow 14+ required for 2D array support",
        ),
    ]
    for condition, reason in version_conditions:
        if condition:
            pytest.skip(reason)

    if any(x in str(constructor) for x in ("dask", "modin", "cudf", "pyspark")):
        request.applymarker(
            pytest.mark.xfail(
                reason="2D array operations not supported in these backends"
            )
        )

    data = {"a": [[[1, 2], [3, 4], [5, 6]]]}
    df = nw_v1.from_native(constructor(data)).with_columns(
        a=nw_v1.col("a").cast(nw_v1.Array(nw_v1.Int64(), (3, 2)))
    )
    assert df.collect_schema()["a"] == nw_v1.Array(nw_v1.Int64(), (3, 2))
    assert df.collect_schema()["a"] == nw_v1.Array(nw_v1.Array(nw_v1.Int64(), 2), 3)


def test_second_time_unit() -> None:
    s: IntoSeries = pd.Series(np.array([np.datetime64("2020-01-01", "s")]))
    result = nw_v1.from_native(s, series_only=True)
    expected_unit: Literal["ns", "us", "ms", "s"] = (
        "s" if PANDAS_VERSION >= (2,) else "ns"
    )
    assert result.dtype == nw_v1.Datetime(expected_unit)

    ts_sec = pa.timestamp("s")
    s = pa.chunked_array([pa.array([datetime(2020, 1, 1)], type=ts_sec)], type=ts_sec)
    result = nw_v1.from_native(s, series_only=True)
    assert result.dtype == nw_v1.Datetime("s")

    s = pd.Series(np.array([np.timedelta64(1, "s")]))
    result = nw_v1.from_native(s, series_only=True)
    assert result.dtype == nw_v1.Duration(expected_unit)

    dur_sec = pa.duration("s")
    s = pa.chunked_array([pa.array([timedelta(1)], type=dur_sec)], type=dur_sec)
    result = nw_v1.from_native(s, series_only=True)
    assert result.dtype == nw_v1.Duration("s")


@pytest.mark.skipif(
    PANDAS_VERSION >= (3,),
    reason="pandas 3.0+ disallows this kind of inplace modification",
)
@pytest.mark.skipif(
    PANDAS_VERSION < (1, 4),
    reason="pandas pre 1.4 doesn't change the type on inplace modification",
)
@pytest.mark.filterwarnings("ignore:Setting an item of incompatible")
def test_pandas_inplace_modification_1267() -> None:
    s = pd.Series([1, 2, 3])
    snw = nw_v1.from_native(s, series_only=True)
    assert snw.dtype == nw_v1.Int64
    s[0] = 999.5
    assert snw.dtype == nw_v1.Float64


def test_pandas_fixed_offset_1302() -> None:
    result = nw_v1.from_native(
        pd.Series(pd.to_datetime(["2020-01-01T00:00:00.000000000+01:00"])),
        series_only=True,
    ).dtype
    expected_timezone = "UTC+01:00" if PANDAS_VERSION >= (2,) else "pytz.FixedOffset(60)"
    assert result == nw_v1.Datetime("ns", expected_timezone)

    if PANDAS_VERSION >= (2,):
        result = nw_v1.from_native(
            pd.Series(
                pd.to_datetime(["2020-01-01T00:00:00.000000000+01:00"])
            ).convert_dtypes(dtype_backend="pyarrow"),
            series_only=True,
        ).dtype
        assert result == nw_v1.Datetime("ns", "+01:00")
    else:  # pragma: no cover
        pass


def test_huge_int() -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("polars")

    import duckdb
    import polars as pl

    df = pl.DataFrame({"a": [1, 2, 3]})

    if POLARS_VERSION >= (1, 18):
        result = nw_v1.from_native(df.select(pl.col("a").cast(pl.Int128))).schema
        assert result["a"] == nw_v1.Int128
    else:  # pragma: no cover
        # Int128 was not available yet
        pass

    rel = duckdb.sql("""
        select cast(a as int128) as a
        from df
                     """)
    result = nw_v1.from_native(rel).schema
    assert result["a"] == nw_v1.Int128

    rel = duckdb.sql("""
        select cast(a as uint128) as a
        from df
                     """)
    result = nw_v1.from_native(rel).schema
    assert result["a"] == nw_v1.UInt128

    # TODO(unassigned): once other libraries support Int128/UInt128,
    # add tests for them too


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_decimal() -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("polars")

    import duckdb
    import polars as pl

    df = pl.DataFrame({"a": [1]}, schema={"a": pl.Decimal})
    result = nw_v1.from_native(df).schema
    assert result["a"] == nw_v1.Decimal
    rel = duckdb.sql("""
        select *
        from df
                     """)
    result = nw_v1.from_native(rel).schema
    assert result["a"] == nw_v1.Decimal
    result = nw_v1.from_native(df.to_pandas(use_pyarrow_extension_array=True)).schema
    assert result["a"] == nw_v1.Decimal
    result = nw_v1.from_native(df.to_arrow()).schema
    assert result["a"] == nw_v1.Decimal


def test_dtype_is_x() -> None:
    dtypes = (
        nw_v1.Array,
        nw_v1.Boolean,
        nw_v1.Categorical,
        nw_v1.Date,
        nw_v1.Datetime,
        nw_v1.Decimal,
        nw_v1.Duration,
        nw_v1.Enum,
        nw_v1.Float32,
        nw_v1.Float64,
        nw_v1.Int8,
        nw_v1.Int16,
        nw_v1.Int32,
        nw_v1.Int64,
        nw_v1.Int128,
        nw_v1.List,
        nw_v1.Object,
        nw_v1.String,
        nw_v1.Struct,
        nw_v1.Time,
        nw_v1.UInt8,
        nw_v1.UInt16,
        nw_v1.UInt32,
        nw_v1.UInt64,
        nw_v1.UInt128,
        nw_v1.Unknown,
        nw_v1.Binary,
    )

    is_signed_integer = {nw_v1.Int8, nw_v1.Int16, nw_v1.Int32, nw_v1.Int64, nw_v1.Int128}
    is_unsigned_integer = {
        nw_v1.UInt8,
        nw_v1.UInt16,
        nw_v1.UInt32,
        nw_v1.UInt64,
        nw_v1.UInt128,
    }
    is_float = {nw_v1.Float32, nw_v1.Float64}
    is_decimal = {nw_v1.Decimal}
    is_temporal = {nw_v1.Datetime, nw_v1.Date, nw_v1.Duration, nw_v1.Time}
    is_nested = {nw_v1.Array, nw_v1.List, nw_v1.Struct}

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


@pytest.mark.skipif(POLARS_VERSION < (1, 18), reason="too old for Int128")
def test_huge_int_to_native() -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("polars")

    import duckdb
    import polars as pl

    df = pl.DataFrame({"a": [1, 2, 3]})
    df_casted = (
        nw_v1.from_native(df)
        .with_columns(a_int=nw_v1.col("a").cast(nw_v1.Int128()))
        .to_native()
    )
    assert df_casted.schema["a_int"] == pl.Int128

    rel = duckdb.sql("""
        select cast(a as int64) as a
        from df
                     """)
    result = (
        nw_v1.from_native(rel)
        .with_columns(
            a_int=nw_v1.col("a").cast(nw_v1.Int128()),
            a_unit=nw_v1.col("a").cast(nw_v1.UInt128()),
        )
        .select("a_int", "a_unit")
        .to_native()
    )
    type_a_int, type_a_unit = result.types
    assert type_a_int == "HUGEINT"
    assert type_a_unit == "UHUGEINT"


def test_cast_decimal_to_native() -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("polars")

    import duckdb
    import polars as pl

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
                nw_v1.from_native(obj)  # type: ignore[call-overload]
                .with_columns(a=nw_v1.col("a").cast(nw_v1.Decimal()))
                .to_native()
            )
