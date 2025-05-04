from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from narwhals.exceptions import UnsupportedDTypeError
from narwhals.utils import Implementation
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    from types import ModuleType

    import sqlframe.base.functions as sqlframe_functions
    import sqlframe.base.types as sqlframe_types
    from sqlframe.base.column import Column
    from typing_extensions import TypeAlias

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.expr import SparkLikeExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version

    _NativeDType: TypeAlias = sqlframe_types.DataType

UNITS_DICT = {
    "y": "year",
    "q": "quarter",
    "mo": "month",
    "d": "day",
    "h": "hour",
    "m": "minute",
    "s": "second",
    "ms": "millisecond",
    "us": "microsecond",
    "ns": "nanosecond",
}


class WindowInputs:
    __slots__ = ("expr", "order_by", "partition_by")

    def __init__(
        self,
        expr: Column,
        partition_by: Sequence[str] | Sequence[Column],
        order_by: Sequence[str],
    ) -> None:
        self.expr = expr
        self.partition_by = partition_by
        self.order_by = order_by


# NOTE: don't lru_cache this as `ModuleType` isn't hashable
def native_to_narwhals_dtype(  # noqa: C901, PLR0912
    dtype: _NativeDType, version: Version, spark_types: ModuleType
) -> DType:
    dtypes = version.dtypes
    if TYPE_CHECKING:
        native = sqlframe_types
    else:
        native = spark_types

    if isinstance(dtype, native.DoubleType):
        return dtypes.Float64()
    if isinstance(dtype, native.FloatType):
        return dtypes.Float32()
    if isinstance(dtype, native.LongType):
        return dtypes.Int64()
    if isinstance(dtype, native.IntegerType):
        return dtypes.Int32()
    if isinstance(dtype, native.ShortType):
        return dtypes.Int16()
    if isinstance(dtype, native.ByteType):
        return dtypes.Int8()
    if isinstance(dtype, (native.StringType, native.VarcharType, native.CharType)):
        return dtypes.String()
    if isinstance(dtype, native.BooleanType):
        return dtypes.Boolean()
    if isinstance(dtype, native.DateType):
        return dtypes.Date()
    if isinstance(dtype, native.TimestampNTZType):
        # TODO(marco): cover this
        return dtypes.Datetime()  # pragma: no cover
    if isinstance(dtype, native.TimestampType):
        # TODO(marco): is UTC correct, or should we be getting the connection timezone?
        # https://github.com/narwhals-dev/narwhals/issues/2165
        return dtypes.Datetime(time_zone="UTC")
    if isinstance(dtype, native.DecimalType):
        # TODO(marco): cover this
        return dtypes.Decimal()  # pragma: no cover
    if isinstance(dtype, native.ArrayType):
        return dtypes.List(
            inner=native_to_narwhals_dtype(
                dtype.elementType, version=version, spark_types=spark_types
            )
        )
    if isinstance(dtype, native.StructType):
        return dtypes.Struct(
            fields=[
                dtypes.Field(
                    name=field.name,
                    dtype=native_to_narwhals_dtype(
                        field.dataType, version=version, spark_types=spark_types
                    ),
                )
                for field in dtype
            ]
        )
    if isinstance(dtype, native.BinaryType):
        return dtypes.Binary()
    return dtypes.Unknown()  # pragma: no cover


def narwhals_to_native_dtype(  # noqa: C901, PLR0912
    dtype: DType | type[DType], version: Version, spark_types: ModuleType
) -> _NativeDType:
    dtypes = version.dtypes
    if TYPE_CHECKING:
        native = sqlframe_types
    else:
        native = spark_types

    if isinstance_or_issubclass(dtype, dtypes.Float64):
        return native.DoubleType()
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        return native.FloatType()
    if isinstance_or_issubclass(dtype, dtypes.Int64):
        return native.LongType()
    if isinstance_or_issubclass(dtype, dtypes.Int32):
        return native.IntegerType()
    if isinstance_or_issubclass(dtype, dtypes.Int16):
        return native.ShortType()
    if isinstance_or_issubclass(dtype, dtypes.Int8):
        return native.ByteType()
    if isinstance_or_issubclass(dtype, dtypes.String):
        return native.StringType()
    if isinstance_or_issubclass(dtype, dtypes.Boolean):
        return native.BooleanType()
    if isinstance_or_issubclass(dtype, dtypes.Date):
        return native.DateType()
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        dt_time_zone = dtype.time_zone
        if dt_time_zone is None:
            return native.TimestampNTZType()
        if dt_time_zone != "UTC":  # pragma: no cover
            msg = f"Only UTC time zone is supported for PySpark, got: {dt_time_zone}"
            raise ValueError(msg)
        return native.TimestampType()
    if isinstance_or_issubclass(dtype, (dtypes.List, dtypes.Array)):
        return native.ArrayType(
            elementType=narwhals_to_native_dtype(
                dtype.inner, version=version, spark_types=native
            )
        )
    if isinstance_or_issubclass(dtype, dtypes.Struct):  # pragma: no cover
        return native.StructType(
            fields=[
                native.StructField(
                    name=field.name,
                    dataType=narwhals_to_native_dtype(
                        field.dtype, version=version, spark_types=native
                    ),
                )
                for field in dtype.fields
            ]
        )
    if isinstance_or_issubclass(dtype, dtypes.Binary):
        return native.BinaryType()

    if isinstance_or_issubclass(
        dtype,
        (
            dtypes.UInt64,
            dtypes.UInt32,
            dtypes.UInt16,
            dtypes.UInt8,
            dtypes.Enum,
            dtypes.Categorical,
            dtypes.Time,
        ),
    ):  # pragma: no cover
        msg = "Unsigned integer, Enum, Categorical and Time types are not supported by spark-like backend"
        raise UnsupportedDTypeError(msg)

    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def evaluate_exprs(
    df: SparkLikeLazyFrame, /, *exprs: SparkLikeExpr
) -> list[tuple[str, Column]]:
    native_results: list[tuple[str, Column]] = []

    for expr in exprs:
        native_series_list = expr._call(df)
        output_names = expr._evaluate_output_names(df)
        if expr._alias_output_names is not None:
            output_names = expr._alias_output_names(output_names)
        if len(output_names) != len(native_series_list):  # pragma: no cover
            msg = f"Internal error: got output names {output_names}, but only got {len(native_series_list)} results"
            raise AssertionError(msg)
        native_results.extend(zip(output_names, native_series_list))

    return native_results


def _std(
    column: Column,
    ddof: int,
    np_version: tuple[int, ...],
    functions: ModuleType,
    implementation: Implementation,
) -> Column:
    if TYPE_CHECKING:
        F = sqlframe_functions  # noqa: N806
    else:
        F = functions  # noqa: N806
    if implementation is Implementation.PYSPARK and np_version < (2, 0):
        from pyspark.pandas.spark.functions import stddev

        return stddev(column, ddof)  # pyright: ignore[reportReturnType, reportArgumentType]
    if ddof == 0:
        return F.stddev_pop(column)
    if ddof == 1:
        return F.stddev_samp(column)
    n_rows = F.count(column)
    return F.stddev_samp(column) * F.sqrt((n_rows - 1) / (n_rows - ddof))


def _var(
    column: Column,
    ddof: int,
    np_version: tuple[int, ...],
    functions: ModuleType,
    implementation: Implementation,
) -> Column:
    if TYPE_CHECKING:
        F = sqlframe_functions  # noqa: N806
    else:
        F = functions  # noqa: N806
    if implementation is Implementation.PYSPARK and np_version < (2, 0):
        from pyspark.pandas.spark.functions import var

        return var(column, ddof)  # pyright: ignore[reportReturnType, reportArgumentType]
    if ddof == 0:
        return F.var_pop(column)
    if ddof == 1:
        return F.var_samp(column)

    n_rows = F.count(column)
    return F.var_samp(column) * (n_rows - 1) / (n_rows - ddof)


def import_functions(implementation: Implementation, /) -> ModuleType:
    if implementation is Implementation.PYSPARK:
        from pyspark.sql import functions

        return functions
    if implementation is Implementation.PYSPARK_CONNECT:
        from pyspark.sql.connect import functions

        return functions
    from sqlframe.base.session import _BaseSession

    return import_module(f"sqlframe.{_BaseSession().execution_dialect_name}.functions")


def import_native_dtypes(implementation: Implementation, /) -> ModuleType:
    if implementation is Implementation.PYSPARK:
        from pyspark.sql import types

        return types
    if implementation is Implementation.PYSPARK_CONNECT:
        from pyspark.sql.connect import types

        return types
    from sqlframe.base.session import _BaseSession

    return import_module(f"sqlframe.{_BaseSession().execution_dialect_name}.types")


def import_window(implementation: Implementation, /) -> type[Any]:
    if implementation is Implementation.PYSPARK:
        from pyspark.sql import Window

        return Window

    if implementation is Implementation.PYSPARK_CONNECT:
        from pyspark.sql.connect.window import Window

        return Window
    from sqlframe.base.session import _BaseSession

    return import_module(
        f"sqlframe.{_BaseSession().execution_dialect_name}.window"
    ).Window
