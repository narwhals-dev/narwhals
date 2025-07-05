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
    into_dtype,
    is_non_nested_literal,
    is_series,
    py_to_narwhals_dtype,
)
from narwhals._plan.expr import All, Column, Columns, IndexColumns, Len, Nth
from narwhals._plan.literal import ScalarLiteral, SeriesLiteral
from narwhals._plan.ranges import IntRange
from narwhals._plan.strings import ConcatHorizontal
from narwhals._plan.when_then import When
from narwhals._utils import Version, flatten
from narwhals.exceptions import InvalidOperationError as OrderDependentExprError

if t.TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan.dummy import DummyExpr, DummySeries
    from narwhals._plan.expr import SortBy
    from narwhals._plan.typing import IntoExpr, IntoExprColumn, NativeSeriesT
    from narwhals.dtypes import IntegerType
    from narwhals.typing import IntoDType, NonNestedLiteral


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
    value: NonNestedLiteral | DummySeries[NativeSeriesT], dtype: IntoDType | None = None
) -> DummyExpr:
    if is_series(value):
        return SeriesLiteral(value=value).to_literal().to_narwhals()
    if not is_non_nested_literal(value):
        msg = f"{type(value).__name__!r} is not supported in `nw.lit`, got: {value!r}."
        raise TypeError(msg)
    if dtype is None:
        dtype = py_to_narwhals_dtype(value, Version.MAIN)
    else:
        dtype = into_dtype(dtype)
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


def when(
    *predicates: IntoExprColumn | t.Iterable[IntoExprColumn], **constraints: t.Any
) -> When:
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
    condition = parse.parse_predicates_constraints_into_expr_ir(
        *predicates, **constraints
    )
    return When._from_ir(condition)


def int_range(
    start: int | IntoExprColumn = 0,
    end: int | IntoExprColumn | None = None,
    step: int = 1,
    *,
    dtype: IntegerType | type[IntegerType] = Version.MAIN.dtypes.Int64,
    eager: bool = False,
) -> DummyExpr:
    if end is None:
        end = start
        start = 0
    if eager:
        msg = f"{eager=}"
        raise NotImplementedError(msg)
    return (
        IntRange(step=step, dtype=into_dtype(dtype))
        .to_function_expr(*parse.parse_into_seq_of_expr_ir(start, end))
        .to_narwhals()
    )


def _is_order_enforcing_previous(obj: t.Any) -> TypeIs[SortBy]:
    """In theory, we could add other nodes to this check."""
    from narwhals._plan.expr import SortBy

    allowed = (SortBy,)
    return isinstance(obj, allowed)


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
