import polars as pl
from narwhals import dtypes

from narwhals._polars.utils import extract_args_kwargs


class PolarsNamespace:
    Int64 = dtypes.Int64
    Int32 = dtypes.Int32
    Int16 = dtypes.Int16
    Int8 = dtypes.Int8
    UInt64 = dtypes.UInt64
    UInt32 = dtypes.UInt32
    UInt16 = dtypes.UInt16
    UInt8 = dtypes.UInt8
    Float64 = dtypes.Float64
    Float32 = dtypes.Float32
    Boolean = dtypes.Boolean
    Object = dtypes.Object
    Unknown = dtypes.Unknown
    Categorical = dtypes.Categorical
    Enum = dtypes.Enum
    String = dtypes.String
    Datetime = dtypes.Datetime
    Duration = dtypes.Duration
    Date = dtypes.Date

    def _from_native_expr(self, expr):
        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(expr)

    def __getattr__(self, attr):
        def func(*args, **kwargs):
            args, kwargs = extract_args_kwargs(args, kwargs)
            return self._from_native_expr(getattr(pl, attr)(*args, **kwargs))

        return func
