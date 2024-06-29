from __future__ import annotations

from typing import Iterable

from narwhals import dtypes
from narwhals._arrow.expr import ArrowExpr
from narwhals._arrow.series import ArrowSeries
from narwhals.utils import flatten


class ArrowNamespace:
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
    Categorical = dtypes.Categorical
    String = dtypes.String
    Datetime = dtypes.Datetime
    Date = dtypes.Date

    # --- not in spec ---
    def __init__(self) -> None: ...

    # --- selection ---
    def col(self, *column_names: str | Iterable[str]) -> ArrowExpr:
        return ArrowExpr.from_column_names(
            *flatten(column_names)
        )

    def all(self) -> ArrowExpr:
        return ArrowExpr(
            lambda df: [
                ArrowSeries(
                    df._dataframe[column_name], name=column_name
                )
                for column_name in df.columns
            ],
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
        )
