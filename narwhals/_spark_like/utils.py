from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from pyspark.sql import Column
from pyspark.sql import Window
from pyspark.sql import functions as F  # noqa: N812
from pyspark.sql import types as pyspark_types

from narwhals.exceptions import UnsupportedDTypeError
from narwhals.utils import import_dtypes_module
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    from pyspark.sql import Column

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.expr import SparkLikeExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version


@lru_cache(maxsize=16)
def native_to_narwhals_dtype(
    dtype: pyspark_types.DataType,
    version: Version,
) -> DType:  # pragma: no cover
    dtypes = import_dtypes_module(version=version)

    if isinstance(dtype, pyspark_types.DoubleType):
        return dtypes.Float64()
    if isinstance(dtype, pyspark_types.FloatType):
        return dtypes.Float32()
    if isinstance(dtype, pyspark_types.LongType):
        return dtypes.Int64()
    if isinstance(dtype, pyspark_types.IntegerType):
        return dtypes.Int32()
    if isinstance(dtype, pyspark_types.ShortType):
        return dtypes.Int16()
    if isinstance(dtype, pyspark_types.ByteType):
        return dtypes.Int8()
    string_types = [
        pyspark_types.StringType,
        pyspark_types.VarcharType,
        pyspark_types.CharType,
    ]
    if any(isinstance(dtype, t) for t in string_types):
        return dtypes.String()
    if isinstance(dtype, pyspark_types.BooleanType):
        return dtypes.Boolean()
    if isinstance(dtype, pyspark_types.DateType):
        return dtypes.Date()
    datetime_types = [
        pyspark_types.TimestampType,
        pyspark_types.TimestampNTZType,
    ]
    if any(isinstance(dtype, t) for t in datetime_types):
        return dtypes.Datetime()
    if isinstance(dtype, pyspark_types.DecimalType):  # pragma: no cover
        # TODO(unassigned): cover this in dtypes_test.py
        return dtypes.Decimal()
    return dtypes.Unknown()


def narwhals_to_native_dtype(
    dtype: DType | type[DType], version: Version
) -> pyspark_types.DataType:
    dtypes = import_dtypes_module(version)

    if isinstance_or_issubclass(dtype, dtypes.Float64):
        return pyspark_types.DoubleType()
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        return pyspark_types.FloatType()
    if isinstance_or_issubclass(dtype, dtypes.Int64):
        return pyspark_types.LongType()
    if isinstance_or_issubclass(dtype, dtypes.Int32):
        return pyspark_types.IntegerType()
    if isinstance_or_issubclass(dtype, dtypes.Int16):
        return pyspark_types.ShortType()
    if isinstance_or_issubclass(dtype, dtypes.Int8):
        return pyspark_types.ByteType()
    if isinstance_or_issubclass(dtype, dtypes.String):
        return pyspark_types.StringType()
    if isinstance_or_issubclass(dtype, dtypes.Boolean):
        return pyspark_types.BooleanType()
    if isinstance_or_issubclass(dtype, (dtypes.Date, dtypes.Datetime)):
        msg = "Converting to Date or Datetime dtype is not supported yet"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.List):  # pragma: no cover
        msg = "Converting to List dtype is not supported yet"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Struct):  # pragma: no cover
        msg = "Converting to Struct dtype is not supported yet"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Array):  # pragma: no cover
        msg = "Converting to Array dtype is not supported yet"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(
        dtype, (dtypes.UInt64, dtypes.UInt32, dtypes.UInt16, dtypes.UInt8)
    ):  # pragma: no cover
        msg = "Unsigned integer types are not supported by PySpark"
        raise UnsupportedDTypeError(msg)

    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def parse_exprs_and_named_exprs(
    df: SparkLikeLazyFrame,
) -> Callable[..., tuple[dict[str, Column], list[bool]]]:
    def func(
        *exprs: SparkLikeExpr, **named_exprs: SparkLikeExpr
    ) -> tuple[dict[str, Column], list[bool]]:
        native_results: dict[str, list[Column]] = {}

        # `returns_scalar` keeps track if an expression returns a scalar and is not lit.
        # Notice that lit is quite special case, since it gets broadcasted by pyspark
        # without the need of adding `.over(Window.partitionBy(F.lit(1)))`
        returns_scalar: list[bool] = []
        for expr in exprs:
            native_series_list = expr._call(df)
            output_names = expr._evaluate_output_names(df)
            if expr._alias_output_names is not None:
                output_names = expr._alias_output_names(output_names)
            if len(output_names) != len(native_series_list):  # pragma: no cover
                msg = f"Internal error: got output names {output_names}, but only got {len(native_series_list)} results"
                raise AssertionError(msg)
            native_results.update(zip(output_names, native_series_list))
            returns_scalar.extend(
                [
                    expr._returns_scalar
                    and expr._function_name.split("->", maxsplit=1)[0] != "lit"
                ]
                * len(output_names)
            )
        for col_alias, expr in named_exprs.items():
            native_series_list = expr._call(df)
            if len(native_series_list) != 1:  # pragma: no cover
                msg = "Named expressions must return a single column"
                raise ValueError(msg)
            native_results[col_alias] = native_series_list[0]
            returns_scalar.append(
                expr._returns_scalar
                and expr._function_name.split("->", maxsplit=1)[0] != "lit"
            )

        return native_results, returns_scalar

    return func


def maybe_evaluate(df: SparkLikeLazyFrame, obj: Any, *, returns_scalar: bool) -> Column:
    from narwhals._spark_like.expr import SparkLikeExpr

    if isinstance(obj, SparkLikeExpr):
        column_results = obj._call(df)
        if len(column_results) != 1:  # pragma: no cover
            msg = "Multi-output expressions (e.g. `nw.all()` or `nw.col('a', 'b')`) not supported in this context"
            raise NotImplementedError(msg)
        column_result = column_results[0]
        if (
            obj._returns_scalar
            and obj._function_name.split("->", maxsplit=1)[0] != "lit"
            and not returns_scalar
        ):
            # Returns scalar, but overall expression doesn't.
            # Let PySpark do its broadcasting
            return column_result.over(Window.partitionBy(F.lit(1)))
        return column_result
    return F.lit(obj)


def _std(_input: Column | str, ddof: int, np_version: tuple[int, ...]) -> Column:
    if np_version > (2, 0):
        if ddof == 1:
            return F.stddev_samp(_input)

        n_rows = F.count(_input)
        return F.stddev_samp(_input) * F.sqrt((n_rows - 1) / (n_rows - ddof))

    from pyspark.pandas.spark.functions import stddev

    input_col = F.col(_input) if isinstance(_input, str) else _input
    return stddev(input_col, ddof=ddof)


def _var(_input: Column | str, ddof: int, np_version: tuple[int, ...]) -> Column:
    if np_version > (2, 0):
        if ddof == 1:
            return F.var_samp(_input)

        n_rows = F.count(_input)
        return F.var_samp(_input) * (n_rows - 1) / (n_rows - ddof)

    from pyspark.pandas.spark.functions import var

    input_col = F.col(_input) if isinstance(_input, str) else _input
    return var(input_col, ddof=ddof)


def binary_operation_returns_scalar(lhs: SparkLikeExpr, rhs: SparkLikeExpr | Any) -> bool:
    # If `rhs` is a SparkLikeExpr, we look at `_returns_scalar`. If it isn't,
    # it means that it was a scalar (e.g. nw.col('a') + 1), and so we default
    # to `True`.
    return lhs._returns_scalar and getattr(rhs, "_returns_scalar", True)
