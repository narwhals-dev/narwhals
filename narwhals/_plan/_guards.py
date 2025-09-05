"""Common type guards, mostly with inline imports."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypeVar

from narwhals._utils import _hasattr_static

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan import expr
    from narwhals._plan.dummy import Expr, Series
    from narwhals._plan.protocols import CompliantSeries
    from narwhals._plan.typing import NativeSeriesT, Seq
    from narwhals.typing import NonNestedLiteral

    T = TypeVar("T")

_NON_NESTED_LITERAL_TPS = (
    int,
    float,
    str,
    dt.date,
    dt.time,
    dt.timedelta,
    bytes,
    Decimal,
)


def _dummy(*_: Any):  # type: ignore[no-untyped-def]  # noqa: ANN202
    from narwhals._plan import dummy

    return dummy


def _expr(*_: Any):  # type: ignore[no-untyped-def]  # noqa: ANN202
    from narwhals._plan import expr

    return expr


def is_non_nested_literal(obj: Any) -> TypeIs[NonNestedLiteral]:
    return obj is None or isinstance(obj, _NON_NESTED_LITERAL_TPS)


def is_expr(obj: Any) -> TypeIs[Expr]:
    return isinstance(obj, _dummy().Expr)


def is_column(obj: Any) -> TypeIs[Expr]:
    """Indicate if the given object is a basic/unaliased column."""
    return is_expr(obj) and obj.meta.is_column()


def is_series(obj: Series[NativeSeriesT] | Any) -> TypeIs[Series[NativeSeriesT]]:
    return isinstance(obj, _dummy().Series)


def is_compliant_series(
    obj: CompliantSeries[NativeSeriesT] | Any,
) -> TypeIs[CompliantSeries[NativeSeriesT]]:
    return _hasattr_static(obj, "__narwhals_series__")


def is_iterable_reject(obj: Any) -> TypeIs[str | bytes | Series | CompliantSeries]:
    return isinstance(obj, (str, bytes, _dummy().Series)) or is_compliant_series(obj)


def is_window_expr(obj: Any) -> TypeIs[expr.WindowExpr]:
    return isinstance(obj, _expr().WindowExpr)


def is_function_expr(obj: Any) -> TypeIs[expr.FunctionExpr[Any]]:
    return isinstance(obj, _expr().FunctionExpr)


def is_binary_expr(obj: Any) -> TypeIs[expr.BinaryExpr]:
    return isinstance(obj, _expr().BinaryExpr)


def is_agg_expr(obj: Any) -> TypeIs[expr.AggExpr]:
    return isinstance(obj, _expr().AggExpr)


def is_aggregation(obj: Any) -> TypeIs[expr.AggExpr | expr.FunctionExpr[Any]]:
    """Superset of `ExprIR.is_scalar`, excludes literals & len."""
    return is_agg_expr(obj) or (is_function_expr(obj) and obj.is_scalar)


def is_literal(obj: Any) -> TypeIs[expr.Literal[Any]]:
    return isinstance(obj, _expr().Literal)


def is_horizontal_reduction(obj: Any) -> TypeIs[expr.FunctionExpr[Any]]:
    return is_function_expr(obj) and obj.options.is_input_wildcard_expansion()


def is_tuple_of(obj: Any, tp: type[T]) -> TypeIs[Seq[T]]:
    return bool(isinstance(obj, tuple) and obj and isinstance(obj[0], tp))
