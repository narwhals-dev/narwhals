"""Common type guards, mostly with inline imports."""

from __future__ import annotations

import datetime as dt
import re  # `_utils` imports at module-level
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypeVar

from narwhals._utils import _hasattr_static
from narwhals.dtypes import DType

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan import expressions as ir
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.expr import Expr
    from narwhals._plan.selectors import Selector
    from narwhals._plan.series import Series
    from narwhals._plan.typing import (
        ColumnNameOrSelector,
        IntoExprColumn,
        NativeSeriesT,
        Seq,
    )
    from narwhals.typing import NonNestedLiteral, PythonLiteral

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
_PYTHON_LITERAL_TPS = (*_NON_NESTED_LITERAL_TPS, list, tuple, type(None))


def _ir(*_: Any):  # type: ignore[no-untyped-def]  # noqa: ANN202
    from narwhals._plan import expressions as ir

    return ir


def _expr(*_: Any):  # type: ignore[no-untyped-def]  # noqa: ANN202
    from narwhals._plan import expr

    return expr


def _selectors(*_: Any):  # type: ignore[no-untyped-def]  # noqa: ANN202
    from narwhals._plan import selectors

    return selectors


def _series(*_: Any):  # type: ignore[no-untyped-def]  # noqa: ANN202
    from narwhals._plan import series

    return series


def is_non_nested_literal(obj: Any) -> TypeIs[NonNestedLiteral]:
    return obj is None or isinstance(obj, _NON_NESTED_LITERAL_TPS)


def is_python_literal(obj: Any) -> TypeIs[PythonLiteral]:
    return isinstance(obj, _PYTHON_LITERAL_TPS)


def is_expr(obj: Any) -> TypeIs[Expr]:
    return isinstance(obj, _expr().Expr)


def is_selector(obj: Any) -> TypeIs[Selector]:
    return isinstance(obj, _selectors().Selector)


def is_expr_column(obj: Any) -> TypeIs[Expr]:
    """Indicate if the given object is a basic/unaliased column."""
    return is_expr(obj) and obj.meta.is_column()


def is_series(obj: Series[NativeSeriesT] | Any) -> TypeIs[Series[NativeSeriesT]]:
    return isinstance(obj, _series().Series)


def is_into_expr_column(obj: Any) -> TypeIs[IntoExprColumn]:
    return isinstance(obj, (str, _expr().Expr, _series().Series))


def is_column_name_or_selector(
    obj: Any, *, allow_expr: bool = False
) -> TypeIs[ColumnNameOrSelector]:
    tps = (str, _selectors().Selector) if not allow_expr else (str, _expr().Expr)
    return isinstance(obj, tps)


def is_compliant_series(
    obj: CompliantSeries[NativeSeriesT] | Any,
) -> TypeIs[CompliantSeries[NativeSeriesT]]:
    return _hasattr_static(obj, "__narwhals_series__")


def is_iterable_reject(obj: Any) -> TypeIs[str | bytes | Series | CompliantSeries]:
    return isinstance(obj, (str, bytes, _series().Series, DType)) or is_compliant_series(
        obj
    )


def is_over(obj: Any) -> TypeIs[ir.Over]:
    return isinstance(obj, _ir().Over)


def is_function_expr(obj: Any) -> TypeIs[ir.FunctionExpr[Any]]:
    return isinstance(obj, _ir().FunctionExpr)


def is_binary_expr(obj: Any) -> TypeIs[ir.BinaryExpr]:
    return isinstance(obj, _ir().BinaryExpr)


def is_agg_expr(obj: Any) -> TypeIs[ir.AggExpr]:
    return isinstance(obj, _ir().AggExpr)


def is_aggregation(obj: Any) -> TypeIs[ir.AggExpr | ir.FunctionExpr[Any]]:
    """Superset of `ExprIR.is_scalar`, excludes literals & len."""
    return is_agg_expr(obj) or (is_function_expr(obj) and obj.is_scalar)


def is_literal(obj: Any) -> TypeIs[ir.Literal[Any]]:
    return isinstance(obj, _ir().Literal)


def is_tuple_of(obj: Any, tp: type[T]) -> TypeIs[Seq[T]]:
    """Return True if the **first** element of the tuple `obj` is an instance of `tp`."""
    return bool(isinstance(obj, tuple) and obj and isinstance(obj[0], tp))


def is_re_pattern(obj: Any) -> TypeIs[re.Pattern[str]]:
    return isinstance(obj, re.Pattern)


def is_seq_column(exprs: Seq[ir.ExprIR], /) -> TypeIs[Seq[ir.Column]]:
    """Return True if **every** element is a `Column`.

    Use this for detecting fastpaths in sub-expressions, that can rely on
    every element in `exprs` having a resolved `name` attribute.
    """
    Column = _ir().Column  # noqa: N806
    return all(isinstance(e, Column) for e in exprs)
