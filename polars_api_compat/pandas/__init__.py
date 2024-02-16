from __future__ import annotations
from typing_extensions import Self
from polars_api_compat.utils import register_expression_call, flatten_into_expr, parse_into_exprs

import re
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any, Iterable
from typing import Callable
from typing import Literal
from typing import cast

import pandas as pd

from polars_api_compat.pandas.column_object import Series
from polars_api_compat.pandas.dataframe_object import DataFrame, LazyFrame
from polars_api_compat.spec import (
    DataFrame as DataFrameT,
    LazyFrame as LazyFrameT,
    Series as SeriesT,
    IntoExpr,
    Expr as ExprT,
    Namespace as NamespaceT,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars_api_compat.spec import (
        Expr as ExprT,
    )
else:
    NamespaceT = object
    BoolT = object
    DateT = object
    DatetimeT = object
    DurationT = object
    Float32T = object
    Float64T = object
    Int8T = object
    Int16T = object
    Int32T = object
    Int64T = object
    StringT = object
    UInt8T = object
    UInt16T = object
    UInt32T = object
    UInt64T = object
    AggregationT = object
    NullTypeT = object

SUPPORTED_VERSIONS = frozenset({"2023.11-beta"})


def map_pandas_dtype_to_standard_dtype(dtype: Any) -> DType:
    if dtype == "int64":
        return Namespace.Int64()
    if dtype == "Int64":
        return Namespace.Int64()
    if dtype == "int32":
        return Namespace.Int32()
    if dtype == "Int32":
        return Namespace.Int32()
    if dtype == "int16":
        return Namespace.Int16()
    if dtype == "Int16":
        return Namespace.Int16()
    if dtype == "int8":
        return Namespace.Int8()
    if dtype == "Int8":
        return Namespace.Int8()
    if dtype == "uint64":
        return Namespace.UInt64()
    if dtype == "UInt64":
        return Namespace.UInt64()
    if dtype == "uint32":
        return Namespace.UInt32()
    if dtype == "UInt32":
        return Namespace.UInt32()
    if dtype == "uint16":
        return Namespace.UInt16()
    if dtype == "UInt16":
        return Namespace.UInt16()
    if dtype == "uint8":
        return Namespace.UInt8()
    if dtype == "UInt8":
        return Namespace.UInt8()
    if dtype == "float64":
        return Namespace.Float64()
    if dtype == "Float64":
        return Namespace.Float64()
    if dtype == "float32":
        return Namespace.Float32()
    if dtype == "Float32":
        return Namespace.Float32()
    if dtype in ("bool", "boolean"):
        # Also for `pandas.core.arrays.boolean.BooleanDtype`
        return Namespace.Bool()
    if dtype == "object":
        return Namespace.String()
    if dtype == "string":
        return Namespace.String()
    if hasattr(dtype, "name"):
        # For types like `numpy.dtypes.DateTime64DType`
        dtype = dtype.name
    if dtype.startswith("datetime64["):
        match = re.search(r"datetime64\[(\w{1,2})", dtype)
        assert match is not None
        time_unit = cast(Literal["ms", "us"], match.group(1))
        return Namespace.Datetime(time_unit)
    if dtype.startswith("timedelta64["):
        match = re.search(r"timedelta64\[(\w{1,2})", dtype)
        assert match is not None
        time_unit = cast(Literal["ms", "us"], match.group(1))
        return Namespace.Duration(time_unit)
    msg = f"Unsupported dtype! {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def map_standard_dtype_to_pandas_dtype(dtype: DType) -> Any:
    if isinstance(dtype, Namespace.Int64):
        return "int64"
    if isinstance(dtype, Namespace.Int32):
        return "int32"
    if isinstance(dtype, Namespace.Int16):
        return "int16"
    if isinstance(dtype, Namespace.Int8):
        return "int8"
    if isinstance(dtype, Namespace.UInt64):
        return "uint64"
    if isinstance(dtype, Namespace.UInt32):
        return "uint32"
    if isinstance(dtype, Namespace.UInt16):
        return "uint16"
    if isinstance(dtype, Namespace.UInt8):
        return "uint8"
    if isinstance(dtype, Namespace.Float64):
        return "float64"
    if isinstance(dtype, Namespace.Float32):
        return "float32"
    if isinstance(dtype, Namespace.Bool):
        return "bool"
    if isinstance(dtype, Namespace.String):
        return "object"
    if isinstance(dtype, Namespace.Datetime):
        if dtype.time_zone is not None:  # pragma: no cover (todo)
            return f"datetime64[{dtype.time_unit}, {dtype.time_zone}]"
        return f"datetime64[{dtype.time_unit}]"
    if isinstance(dtype, Namespace.Duration):
        return f"timedelta64[{dtype.time_unit}]"
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def convert_to_standard_compliant_column(
    ser: pd.Series[Any],
    api_version: str | None = None,
) -> Series:
    if ser.name is not None and not isinstance(ser.name, str):
        msg = f"Expected column with string name, got: {ser.name}"
        raise ValueError(msg)
    if ser.name is None:
        ser = ser.rename("")
    return Series(
        ser,
        api_version=api_version or "2023.11-beta",
    )


def convert_to_standard_compliant_dataframe(
    df: pd.DataFrame,
    api_version: str | None = None,
) -> tuple[DataFrame, NamespaceT]:
    df = DataFrame(df, api_version=api_version or "2023.11-beta")
    return df, df.__dataframe_namespace__()


class Namespace(NamespaceT):
    def __init__(self, *, api_version: str) -> None:
        self.__dataframeapi_version__ = api_version
        self.api_version = api_version

    # --- horizontal reductions
    def sum_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> ExprT:
        return reduce(lambda x, y: x + y, parse_into_exprs(self, *exprs))

    def all_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> ExprT:
        return reduce(lambda x, y: x & y, parse_into_exprs(self, *exprs))

    def any_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> ExprT:
        return reduce(lambda x, y: x | y, parse_into_exprs(self, *exprs))

    def col(self, *column_names: str | Iterable[str]) -> Expr:
        return Expr.from_column_names(flatten_into_expr(*column_names))

    def sum(self, column_name: str) -> Expr:
        return Expr.from_column_names(column_name).sum()

    def mean(self, column_name: str) -> Expr:
        return Expr.from_column_names(column_name).mean()

    def len(self) -> Expr:
        return Expr(
            lambda df: [
                Series(
                    pd.Series([len(df.dataframe)], name="len", index=[0]),
                    api_version=df.api_version,
                )
            ]
        )

    def _create_expr_from_callable(self, call: Callable[[DataFrameT|LazyFrameT], list[SeriesT]]) -> ExprT:
        return Expr(call)

    def _create_series_from_scalar(self, value: Any, series: SeriesT) -> SeriesT:
        return Series(
            pd.Series([value], name=series.series.name, index=series.series.index[0:1]),
            api_version=self.api_version,
        )

    def _create_expr_from_series(self, series: SeriesT) -> ExprT:
        return Expr(lambda df: [series])

    def all(self) -> Expr:
        return Expr(
            lambda df: [
                Series(
                    df.dataframe.loc[:, column_name],
                    api_version=df.api_version,
                )
                for column_name in df.columns
            ],
        )


class Expr(ExprT):
    def __init__(self, call: Callable[[DataFrameT|LazyFrameT], list[SeriesT]]) -> None:
        self.call = call
        self.api_version = '0.20.0'  # todo

    @classmethod
    def from_column_names(cls: type[Expr], *column_names: str) -> Expr:
        return cls(
            lambda df: [
                Series(
                    df.dataframe.loc[:, column_name],
                    api_version=df.api_version,
                )
                for column_name in column_names
            ],
        )

    def __expr_namespace__(self) -> Namespace:
        return Namespace(api_version="2023.11-beta")

    def __eq__(self, other: Expr | Any) -> Self:  # type: ignore[override]
        return register_expression_call(self, "__eq__", other)

    def __ne__(self, other: Expr | Any) -> Self:  # type: ignore[override]
        return register_expression_call(self, "__ne__", other)

    def __ge__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__ge__", other)

    def __gt__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__gt__", other)

    def __le__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__le__", other)

    def __lt__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__lt__", other)

    def __and__(self, other: Expr | bool | Any) -> Self:
        return register_expression_call(self, "__and__", other)

    def __rand__(self, other: Series | Any) -> Self:
        return register_expression_call(self, "__rand__", other)

    def __or__(self, other: Expr | bool | Any) -> Self:
        return register_expression_call(self, "__or__", other)

    def __ror__(self, other: Series | Any) -> Self:
        return register_expression_call(self, "__ror__", other)

    def __add__(self, other: Expr | Any) -> Self:  # type: ignore[override]
        return register_expression_call(self, "__add__", other)

    def __radd__(self, other: Series | Any) -> Self:
        return register_expression_call(self, "__radd__", other)

    def __sub__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__sub__", other)

    def __rsub__(self, other: Series | Any) -> Self:
        return register_expression_call(self, "__rsub__", other)

    def __mul__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__mul__", other)

    def __rmul__(self, other: Series | Any) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__truediv__", other)

    def __rtruediv__(self, other: Series | Any) -> Self:
        raise NotImplementedError

    def __floordiv__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__floordiv__", other)

    def __rfloordiv__(self, other: Series | Any) -> Self:
        raise NotImplementedError

    def __pow__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__pow__", other)

    def __rpow__(self, other: Series | Any) -> Self:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__mod__", other)

    def __rmod__(self, other: Series | Any) -> Self:  # pragma: no cover
        raise NotImplementedError

    # Unary

    def __invert__(self) -> Self:
        return register_expression_call(self, "__invert__")

    # Reductions

    def sum(self) -> Expr:
        return register_expression_call(self, "sum")

    def mean(self) -> Expr:
        return register_expression_call(self, "mean")

    # Other

    def alias(self, name: str) -> Expr:
        return register_expression_call(self, "alias", name)
