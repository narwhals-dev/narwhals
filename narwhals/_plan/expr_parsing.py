from __future__ import annotations

# ruff: noqa: A002
from itertools import chain
from typing import TYPE_CHECKING, Iterable, Sequence, TypeVar

from narwhals._plan.common import IntoExprColumn, is_expr, is_iterable_reject
from narwhals._plan.exceptions import (
    invalid_into_expr_error,
    is_iterable_pandas_error,
    is_iterable_polars_error,
)
from narwhals.dependencies import get_polars, is_pandas_dataframe, is_pandas_series

if TYPE_CHECKING:
    from typing import Any, Iterator

    import polars as pl
    from typing_extensions import TypeAlias, TypeIs

    from narwhals._plan.common import ExprIR, IntoExpr, Seq
    from narwhals.dtypes import DType

T = TypeVar("T")

_RaisesInvalidIntoExprError: TypeAlias = "Any"
"""
Placeholder for multiple `Iterable[IntoExpr]`.

We only support cases `a`, `b`, but the typing for most contexts is more permissive:

>>> import polars as pl
>>> df = pl.DataFrame({"one": ["A", "B", "A"], "two": [1, 2, 3], "three": [4, 5, 6]})
>>> a = ("one", "two")
>>> b = (["one", "two"],)
>>>
>>> c = ("one", ["two"])
>>> d = (["one"], "two")
>>> [df.select(*into) for into in (a, b, c, d)]
[shape: (3, 2)
 ┌─────┬─────┐
 │ one ┆ two │
 │ --- ┆ --- │
 │ str ┆ i64 │
 ╞═════╪═════╡
 │ A   ┆ 1   │
 │ B   ┆ 2   │
 │ A   ┆ 3   │
 └─────┴─────┘,
 shape: (3, 2)
 ┌─────┬─────┐
 │ one ┆ two │
 │ --- ┆ --- │
 │ str ┆ i64 │
 ╞═════╪═════╡
 │ A   ┆ 1   │
 │ B   ┆ 2   │
 │ A   ┆ 3   │
 └─────┴─────┘,
 shape: (3, 2)
 ┌─────┬───────────┐
 │ one ┆ literal   │
 │ --- ┆ ---       │
 │ str ┆ list[str] │
 ╞═════╪═══════════╡
 │ A   ┆ ["two"]   │
 │ B   ┆ ["two"]   │
 │ A   ┆ ["two"]   │
 └─────┴───────────┘,
 shape: (3, 2)
 ┌───────────┬─────┐
 │ literal   ┆ two │
 │ ---       ┆ --- │
 │ list[str] ┆ i64 │
 ╞═══════════╪═════╡
 │ ["one"]   ┆ 1   │
 │ ["one"]   ┆ 2   │
 │ ["one"]   ┆ 3   │
 └───────────┴─────┘]
"""


def parse_into_expr_ir(
    input: IntoExpr, *, str_as_lit: bool = False, dtype: DType | None = None
) -> ExprIR:
    """Parse a single input into an `ExprIR` node."""
    from narwhals._plan import demo as nwd

    if is_expr(input):
        expr = input
    elif isinstance(input, str) and not str_as_lit:
        expr = nwd.col(input)
    else:
        expr = nwd.lit(input, dtype=dtype)
    return expr._ir


def parse_into_seq_of_expr_ir(
    first_input: IntoExpr | Iterable[IntoExpr] = (),
    *more_inputs: IntoExpr | _RaisesInvalidIntoExprError,
    **named_inputs: IntoExpr,
) -> Seq[ExprIR]:
    """Parse variadic inputs into a flat sequence of `ExprIR` nodes."""
    return tuple(_parse_into_iter_expr_ir(first_input, *more_inputs, **named_inputs))


def parse_predicates_constraints_into_expr_ir(
    first_predicate: IntoExprColumn | Iterable[IntoExprColumn] = (),
    *more_predicates: IntoExprColumn | _RaisesInvalidIntoExprError,
    **constraints: IntoExpr,
) -> ExprIR:
    """Parse variadic predicates and constraints into an `ExprIR` node.

    The result is an AND-reduction of all inputs.
    """
    all_predicates = _parse_into_iter_expr_ir(first_predicate, *more_predicates)
    if constraints:
        chained = chain(all_predicates, _parse_constraints(constraints))
        return _combine_predicates(chained)
    return _combine_predicates(all_predicates)


def _parse_into_iter_expr_ir(
    first_input: IntoExpr | Iterable[IntoExpr],
    *more_inputs: IntoExpr,
    **named_inputs: IntoExpr,
) -> Iterator[ExprIR]:
    if not _is_empty_sequence(first_input):
        # NOTE: These need to be separated to introduce an intersection type
        # Otherwise, `str | bytes` always passes through typing
        if _is_iterable(first_input) and not is_iterable_reject(first_input):
            if more_inputs:
                raise invalid_into_expr_error(first_input, more_inputs, named_inputs)
            else:
                yield from _parse_positional_inputs(first_input)
        else:
            yield parse_into_expr_ir(first_input)
    else:
        # NOTE: Passthrough case for no inputs - but gets skipped when calling next
        yield from ()
    if more_inputs:
        yield from _parse_positional_inputs(more_inputs)
    if named_inputs:
        yield from _parse_named_inputs(named_inputs)


def _parse_positional_inputs(inputs: Iterable[IntoExpr], /) -> Iterator[ExprIR]:
    for into in inputs:
        yield parse_into_expr_ir(into)


def _parse_named_inputs(named_inputs: dict[str, IntoExpr], /) -> Iterator[ExprIR]:
    from narwhals._plan.expr import Alias

    for name, input in named_inputs.items():
        yield Alias(expr=parse_into_expr_ir(input), name=name)


def _parse_constraints(constraints: dict[str, IntoExpr], /) -> Iterator[ExprIR]:
    from narwhals._plan import demo as nwd

    for name, value in constraints.items():
        yield (nwd.col(name) == value)._ir


def _combine_predicates(predicates: Iterator[ExprIR], /) -> ExprIR:
    from narwhals._plan.boolean import AllHorizontal

    first = next(predicates, None)
    if not first:
        msg = "at least one predicate or constraint must be provided"
        raise TypeError(msg)
    if second := next(predicates, None):
        return AllHorizontal().to_function_expr(first, second, *predicates)
    return first


def _is_iterable(obj: Iterable[T] | Any) -> TypeIs[Iterable[T]]:
    if is_pandas_dataframe(obj) or is_pandas_series(obj):
        raise is_iterable_pandas_error(obj)
    if _is_polars(obj):
        raise is_iterable_polars_error(obj)
    return isinstance(obj, Iterable)


def _is_empty_sequence(obj: Any) -> bool:
    return isinstance(obj, Sequence) and not obj


def _is_polars(obj: Any) -> TypeIs[pl.Series | pl.Expr | pl.DataFrame | pl.LazyFrame]:
    return (pl := get_polars()) is not None and isinstance(
        obj, (pl.Series, pl.Expr, pl.DataFrame, pl.LazyFrame)
    )
