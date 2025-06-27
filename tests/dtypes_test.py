from __future__ import annotations

import enum
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, POLARS_VERSION, PYARROW_VERSION

if TYPE_CHECKING:
    from collections.abc import Iterable

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
    df = nw.from_native(constructor(data)).with_columns(
        a=nw.col("a").cast(nw.Array(nw.Int64(), (3, 2)))
    )
    assert df.collect_schema()["a"] == nw.Array(nw.Int64(), (3, 2))
    assert df.collect_schema()["a"] == nw.Array(nw.Array(nw.Int64(), 2), 3)


def test_second_time_unit() -> None:
    s: IntoSeries = pd.Series(np.array([np.datetime64("2020-01-01", "s")]))
    result = nw.from_native(s, series_only=True)
    expected_unit: Literal["ns", "us", "ms", "s"] = (
        "s" if PANDAS_VERSION >= (2,) else "ns"
    )
    assert result.dtype == nw.Datetime(expected_unit)

    ts_sec = pa.timestamp("s")
    s = pa.chunked_array([pa.array([datetime(2020, 1, 1)], type=ts_sec)], type=ts_sec)
    result = nw.from_native(s, series_only=True)
    assert result.dtype == nw.Datetime("s")

    s = pd.Series(np.array([np.timedelta64(1, "s")]))
    result = nw.from_native(s, series_only=True)
    assert result.dtype == nw.Duration(expected_unit)

    dur_sec = pa.duration("s")
    s = pa.chunked_array([pa.array([timedelta(1)], type=dur_sec)], type=dur_sec)
    result = nw.from_native(s, series_only=True)
    assert result.dtype == nw.Duration("s")


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
    snw = nw.from_native(s, series_only=True)
    assert snw.dtype == nw.Int64
    s[0] = 999.5
    assert snw.dtype == nw.Float64


def test_pandas_fixed_offset_1302() -> None:
    result = nw.from_native(
        pd.Series(pd.to_datetime(["2020-01-01T00:00:00.000000000+01:00"])),
        series_only=True,
    ).dtype
    expected_timezone = "UTC+01:00" if PANDAS_VERSION >= (2,) else "pytz.FixedOffset(60)"
    assert result == nw.Datetime("ns", expected_timezone)

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
    pytest.importorskip("duckdb")
    pytest.importorskip("polars")

    import duckdb
    import polars as pl

    df = pl.DataFrame({"a": [1, 2, 3]})

    if POLARS_VERSION >= (1, 18):
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
    pytest.importorskip("duckdb")
    pytest.importorskip("polars")

    import duckdb
    import polars as pl

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
        nw.Time,
        nw.UInt8,
        nw.UInt16,
        nw.UInt32,
        nw.UInt64,
        nw.UInt128,
        nw.Unknown,
        nw.Binary,
    )

    is_signed_integer = {nw.Int8, nw.Int16, nw.Int32, nw.Int64, nw.Int128}
    is_unsigned_integer = {nw.UInt8, nw.UInt16, nw.UInt32, nw.UInt64, nw.UInt128}
    is_float = {nw.Float32, nw.Float64}
    is_decimal = {nw.Decimal}
    is_temporal = {nw.Datetime, nw.Date, nw.Duration, nw.Time}
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


@pytest.mark.skipif(POLARS_VERSION < (1, 18), reason="too old for Int128")
def test_huge_int_to_native() -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("polars")

    import duckdb
    import polars as pl

    df = pl.DataFrame({"a": [1, 2, 3]})
    df_casted = (
        nw.from_native(df).with_columns(a_int=nw.col("a").cast(nw.Int128())).to_native()
    )
    assert df_casted.schema["a_int"] == pl.Int128

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
                nw.from_native(obj)  # type: ignore[call-overload]
                .with_columns(a=nw.col("a").cast(nw.Decimal()))
                .to_native()
            )


@pytest.mark.parametrize(
    "categories",
    [["a", "b"], [np.str_("a"), np.str_("b")], enum.Enum("Test", "a b"), [1, 2, 3]],
)
def test_enum_valid(categories: Iterable[Any] | type[enum.Enum]) -> None:
    dtype = nw.Enum(categories)
    assert dtype == nw.Enum
    assert len(dtype.categories) == len([*categories])


def test_enum_from_series() -> None:
    pytest.importorskip("polars")
    import polars as pl

    elements = "a", "d", "e", "b", "c"
    categories = pl.Series(elements)
    categories_nw = nw.from_native(categories, series_only=True)
    assert nw.Enum(categories_nw).categories == elements
    assert nw.Enum(categories).categories == elements


def test_enum_categories_immutable() -> None:
    dtype = nw.Enum(["a", "b"])
    with pytest.raises(TypeError, match="does not support item assignment"):
        dtype.categories[0] = "c"  # type: ignore[index]
    with pytest.raises(AttributeError):
        dtype.categories = "a", "b", "c"  # type: ignore[misc]


def test_enum_repr_pd() -> None:
    df = nw.from_native(
        pd.DataFrame(
            {"a": ["broccoli", "cabbage"]}, dtype=pd.CategoricalDtype(ordered=True)
        )
    )
    dtype = df.schema["a"]
    assert isinstance(dtype, nw.Enum)
    assert dtype.categories == ("broccoli", "cabbage")
    assert "Enum(categories=['broccoli', 'cabbage'])" in str(dtype)


def test_enum_repr_pl() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(
        pl.DataFrame(
            {"a": ["broccoli", "cabbage"]}, schema={"a": pl.Enum(["broccoli", "cabbage"])}
        )
    )
    dtype = df.schema["a"]
    assert isinstance(dtype, nw.Enum)
    assert dtype.categories == ("broccoli", "cabbage")
    assert "Enum(categories=['broccoli', 'cabbage'])" in repr(dtype)


def test_enum_repr() -> None:
    result = nw.Enum(["a", "b"])
    assert "Enum(categories=['a', 'b'])" in repr(result)
    result = nw.Enum(nw.Implementation)
    assert "Enum(categories=['pandas', 'modin', 'cudf'" in repr(result)


def test_enum_hash() -> None:
    assert nw.Enum(["a", "b"]) in {nw.Enum(["a", "b"])}
    assert nw.Enum(["a", "b"]) not in {nw.Enum(["a", "b", "c"])}


def test_datetime_w_tz_duckdb() -> None:
    pytest.importorskip("duckdb")
    import duckdb

    duckdb.sql("""set timezone = 'Europe/Amsterdam'""")
    df = nw.from_native(
        duckdb.sql("""select * from values (timestamptz '2020-01-01')df(a)""")
    )
    result = df.collect_schema()
    assert result["a"] == nw.Datetime("us", "Europe/Amsterdam")
    duckdb.sql("""set timezone = 'Asia/Kathmandu'""")
    result = df.collect_schema()
    assert result["a"] == nw.Datetime("us", "Asia/Kathmandu")

    df = nw.from_native(
        duckdb.sql(
            """select * from values (timestamptz '2020-01-01', [[timestamptz '2020-01-02']])df(a,b)"""
        )
    )
    result = df.collect_schema()
    assert result["a"] == nw.Datetime("us", "Asia/Kathmandu")
    assert result["b"] == nw.List(nw.List(nw.Datetime("us", "Asia/Kathmandu")))


def test_datetime_w_tz_pyspark(constructor: Constructor) -> None:  # pragma: no cover
    if "pyspark" not in str(constructor) or "sqlframe" in str(constructor):
        pytest.skip()
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    session = SparkSession.builder.config(
        "spark.sql.session.timeZone", "UTC"
    ).getOrCreate()

    df = nw.from_native(
        session.createDataFrame([(datetime(2020, 1, 1, tzinfo=timezone.utc),)], ["a"])
    )
    result = df.collect_schema()
    assert result["a"] == nw.Datetime("us", "UTC")
    df = nw.from_native(
        session.createDataFrame([([datetime(2020, 1, 1, tzinfo=timezone.utc)],)], ["a"])
    )
    result = df.collect_schema()
    assert result["a"] == nw.List(nw.Datetime("us", "UTC"))
