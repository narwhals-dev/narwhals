from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import overload

from narwhals.dataframe import DataFrame as NwDataFrame
from narwhals.dataframe import LazyFrame as NwLazyFrame
from narwhals.expr import Expr as NwExpr
from narwhals.series import Series as NwSeries
from narwhals.stable.v1.dataframe import DataFrame
from narwhals.stable.v1.dataframe import LazyFrame
from narwhals.stable.v1.expr import Expr
from narwhals.stable.v1.series import Series
from narwhals.utils import Version

if TYPE_CHECKING:
    from typing_extensions import ParamSpec
    from typing_extensions import TypeVar

    from narwhals.typing import IntoFrameT
    from narwhals.typing import IntoSeries

    FrameT = TypeVar("FrameT", "DataFrame[Any]", "LazyFrame[Any]")
    DataFrameT = TypeVar("DataFrameT", bound="DataFrame[Any]")
    LazyFrameT = TypeVar("LazyFrameT", bound="LazyFrame[Any]")
    SeriesT = TypeVar("SeriesT", bound="Series[Any]")
    IntoSeriesT = TypeVar("IntoSeriesT", bound="IntoSeries", default=Any)
    T = TypeVar("T", default=Any)
    P = ParamSpec("P")
    R = TypeVar("R")
else:
    from typing import TypeVar

    IntoSeriesT = TypeVar("IntoSeriesT", bound="IntoSeries")
    T = TypeVar("T")


@overload
def _stableify(obj: NwDataFrame[IntoFrameT]) -> DataFrame[IntoFrameT]: ...
@overload
def _stableify(obj: NwLazyFrame[IntoFrameT]) -> LazyFrame[IntoFrameT]: ...
@overload
def _stableify(obj: NwSeries[IntoSeriesT]) -> Series[IntoSeriesT]: ...
@overload
def _stableify(obj: NwExpr) -> Expr: ...
@overload
def _stableify(obj: Any) -> Any: ...


def _stableify(
    obj: NwDataFrame[IntoFrameT]
    | NwLazyFrame[IntoFrameT]
    | NwSeries[IntoSeriesT]
    | NwExpr
    | Any,
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT] | Series[IntoSeriesT] | Expr | Any:
    if isinstance(obj, NwDataFrame):
        return DataFrame(obj._compliant_frame._with_version(Version.V1), level=obj._level)
    if isinstance(obj, NwLazyFrame):
        return LazyFrame(obj._compliant_frame._with_version(Version.V1), level=obj._level)
    if isinstance(obj, NwSeries):
        return Series(obj._compliant_series._with_version(Version.V1), level=obj._level)
    if isinstance(obj, NwExpr):
        return Expr(obj._to_compliant_expr, obj._metadata)
    return obj
