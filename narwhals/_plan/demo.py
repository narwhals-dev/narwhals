from __future__ import annotations

import builtins
import typing as t

from narwhals._plan import (
    aggregation as agg,
    boolean,
    expr_parsing as parse,
    functions as F,  # noqa: N812
)
from narwhals._plan.common import (
    ExprIR,
    IntoExpr,
    is_expr,
    is_non_nested_literal,
    py_to_narwhals_dtype,
)
from narwhals._plan.dummy import DummySeries
from narwhals._plan.expr import All, Column, Columns, IndexColumns, Len, Nth
from narwhals._plan.literal import ScalarLiteral, SeriesLiteral
from narwhals._plan.strings import ConcatHorizontal
from narwhals._plan.when_then import When
from narwhals._utils import Version, flatten
from narwhals.dtypes import DType
from narwhals.exceptions import OrderDependentExprError

if t.TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan.dummy import DummyExpr
    from narwhals._plan.expr import SortBy, WindowExpr
    from narwhals.typing import NonNestedLiteral


def col(*names: str | t.Iterable[str]) -> DummyExpr:
    flat_names = tuple(flatten(names))
    node = (
        Column(name=flat_names[0])
        if builtins.len(flat_names) == 1
        else Columns(names=flat_names)
    )
    return node.to_narwhals()


def nth(*indices: int | t.Sequence[int]) -> DummyExpr:
    flat_indices = tuple(flatten(indices))
    node = (
        Nth(index=flat_indices[0])
        if builtins.len(flat_indices) == 1
        else IndexColumns(indices=flat_indices)
    )
    return node.to_narwhals()


def lit(
    value: NonNestedLiteral | DummySeries, dtype: DType | type[DType] | None = None
) -> DummyExpr:
    if isinstance(value, DummySeries):
        return SeriesLiteral(value=value).to_literal().to_narwhals()
    if not is_non_nested_literal(value):
        msg = f"{type(value).__name__!r} is not supported in `nw.lit`, got: {value!r}."
        raise TypeError(msg)
    if dtype is None or not isinstance(dtype, DType):
        dtype = py_to_narwhals_dtype(value, Version.MAIN)
    return ScalarLiteral(value=value, dtype=dtype).to_literal().to_narwhals()


def len() -> DummyExpr:
    return Len().to_narwhals()


def all() -> DummyExpr:
    return All().to_narwhals()


def exclude(*names: str | t.Iterable[str]) -> DummyExpr:
    return all().exclude(*names)


def max(*columns: str) -> DummyExpr:
    return col(columns).max()


def mean(*columns: str) -> DummyExpr:
    return col(columns).mean()


def min(*columns: str) -> DummyExpr:
    return col(columns).min()


def median(*columns: str) -> DummyExpr:
    return col(columns).median()


def sum(*columns: str) -> DummyExpr:
    return col(columns).sum()


def all_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> DummyExpr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return boolean.AllHorizontal().to_function_expr(*it).to_narwhals()


def any_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> DummyExpr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return boolean.AnyHorizontal().to_function_expr(*it).to_narwhals()


def sum_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> DummyExpr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return F.SumHorizontal().to_function_expr(*it).to_narwhals()


def min_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> DummyExpr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MinHorizontal().to_function_expr(*it).to_narwhals()


def max_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> DummyExpr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MaxHorizontal().to_function_expr(*it).to_narwhals()


def mean_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> DummyExpr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MeanHorizontal().to_function_expr(*it).to_narwhals()


def concat_str(
    exprs: IntoExpr | t.Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    separator: str = "",
    ignore_nulls: bool = False,
) -> DummyExpr:
    it = parse.parse_into_seq_of_expr_ir(exprs, *more_exprs)
    return (
        ConcatHorizontal(separator=separator, ignore_nulls=ignore_nulls)
        .to_function_expr(*it)
        .to_narwhals()
    )


def when(*predicates: IntoExpr | t.Iterable[IntoExpr]) -> When:
    """Start a `when-then-otherwise` expression.

    Examples:
        >>> from narwhals._plan import demo as nwd

        >>> when_then_many = (
        ...     nwd.when(nwd.col("x") == "a")
        ...     .then(1)
        ...     .when(nwd.col("x") == "b")
        ...     .then(2)
        ...     .when(nwd.col("x") == "c")
        ...     .then(3)
        ...     .otherwise(4)
        ... )
        >>> when_then_many
        Narwhals DummyExpr (main):
        .when([(col('x')) == (lit(str: a))]).then(lit(int: 1)).otherwise(.when([(col('x')) == (lit(str: b))]).then(lit(int: 2)).otherwise(.when([(col('x')) == (lit(str: c))]).then(lit(int: 3)).otherwise(lit(int: 4))))
        >>>
        >>> nwd.when(nwd.col("y") == "b").then(1)
        Narwhals DummyExpr (main):
        .when([(col('y')) == (lit(str: b))]).then(lit(int: 1)).otherwise(lit(null))
    """
    if builtins.len(predicates) == 1 and is_expr(predicates[0]):
        expr = predicates[0]
    else:
        expr = all_horizontal(*predicates)
    return When._from_expr(expr)


def _is_order_enforcing_previous(obj: t.Any) -> TypeIs[SortBy]:
    """In theory, we could add other nodes to this check."""
    from narwhals._plan.expr import SortBy

    allowed = (SortBy,)
    return isinstance(obj, allowed)


def _is_order_enforcing_next(obj: t.Any) -> TypeIs[WindowExpr]:
    """Not sure how this one would work."""
    from narwhals._plan.expr import WindowExpr

    return isinstance(obj, WindowExpr) and obj.order_by is not None


def _order_dependent_error(node: agg.OrderableAgg) -> OrderDependentExprError:
    previous = node.expr
    method = repr(node).removeprefix(f"{previous!r}.")
    msg = (
        f"{method} is order-dependent and requires an ordering operation for lazy backends.\n"
        f"Hint:\nInstead of:\n"
        f"    {node!r}\n\n"
        "If you want to aggregate to a single value, try:\n"
        f"    {previous!r}.sort_by(...).{method}\n\n"
        "Otherwise, try:\n"
        f"    {node!r}.over(order_by=...)"
    )
    return OrderDependentExprError(msg)


def ensure_orderable_rules(*exprs: DummyExpr) -> tuple[DummyExpr, ...]:
    for expr in exprs:
        node = expr._ir
        if isinstance(node, agg.OrderableAgg):
            previous = node.expr
            if not _is_order_enforcing_previous(previous):
                raise _order_dependent_error(node)
    return exprs


def select_context(
    *exprs: IntoExpr | t.Iterable[IntoExpr], **named_exprs: IntoExpr
) -> tuple[ExprIR, ...]:
    return parse.parse_into_seq_of_expr_ir(*exprs, **named_exprs)
