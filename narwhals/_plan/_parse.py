from __future__ import annotations

from collections.abc import Iterable, Sequence

# ruff: noqa: A002
from itertools import chain
from typing import TYPE_CHECKING

from narwhals._plan._guards import (
    is_expr,
    is_into_expr_column,
    is_iterable_reject,
    is_selector,
)
from narwhals._plan.exceptions import (
    invalid_into_expr_error,
    is_iterable_pandas_error,
    is_iterable_polars_error,
)
from narwhals._utils import qualified_type_name
from narwhals.dependencies import get_polars, is_pandas_dataframe, is_pandas_series
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, TypeVar

    import polars as pl
    from typing_extensions import TypeAlias, TypeIs

    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions import ExprIR, SelectorIR
    from narwhals._plan.typing import (
        ColumnNameOrSelector,
        IntoExpr,
        IntoExprColumn,
        OneOrIterable,
        PartialSeries,
        Seq,
    )
    from narwhals.typing import IntoDType

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
    input: IntoExpr | list[Any],
    *,
    str_as_lit: bool = False,
    list_as_series: PartialSeries | None = None,
    dtype: IntoDType | None = None,
) -> ExprIR:
    """Parse a single input into an `ExprIR` node.

    Arguments:
        input: The input to be parsed as an expression.
        str_as_lit: Interpret string input as a string literal. If set to `False` (default),
            strings are parsed as column names.
        list_as_series: Interpret list input as a Series literal, using the provided constructor.
            If set to `None` (default), lists will raise when passed to `lit`.
        dtype: If the input is expected to resolve to a literal with a known dtype, pass
            this to the `lit` constructor.
    """
    from narwhals._plan import col, lit

    if is_expr(input):
        expr = input
    elif isinstance(input, str) and not str_as_lit:
        expr = col(input)
    elif isinstance(input, list):
        if list_as_series is None:
            raise TypeError(input)
        expr = lit(list_as_series(input))
    else:
        expr = lit(input, dtype=dtype)
    return expr._ir


def parse_into_selector_ir(input: ColumnNameOrSelector | Expr, /) -> SelectorIR:
    if is_selector(input):
        selector = input
    elif isinstance(input, str):
        from narwhals._plan import selectors as cs

        selector = cs.by_name(input)
    elif is_expr(input):
        selector = input.meta.as_selector()
    else:
        msg = f"cannot turn {qualified_type_name(input)!r} into selector"
        raise TypeError(msg)
    return selector._ir


def parse_into_seq_of_expr_ir(
    first_input: OneOrIterable[IntoExpr] = (),
    *more_inputs: IntoExpr | _RaisesInvalidIntoExprError,
    **named_inputs: IntoExpr,
) -> Seq[ExprIR]:
    """Parse variadic inputs into a flat sequence of `ExprIR` nodes."""
    return tuple(
        _parse_into_iter_expr_ir(
            first_input, *more_inputs, _list_as_series=None, **named_inputs
        )
    )


def parse_predicates_constraints_into_expr_ir(
    first_predicate: OneOrIterable[IntoExprColumn] | list[bool] = (),
    *more_predicates: IntoExprColumn | list[bool] | _RaisesInvalidIntoExprError,
    _list_as_series: PartialSeries | None = None,
    **constraints: IntoExpr,
) -> ExprIR:
    """Parse variadic predicates and constraints into an `ExprIR` node.

    The result is an AND-reduction of all inputs.
    """
    all_predicates = _parse_into_iter_expr_ir(
        first_predicate, *more_predicates, _list_as_series=_list_as_series
    )
    if constraints:
        chained = chain(all_predicates, _parse_constraints(constraints))
        return _combine_predicates(chained)
    return _combine_predicates(all_predicates)


def parse_sort_by_into_seq_of_expr_ir(
    by: OneOrIterable[IntoExprColumn] = (), *more_by: IntoExprColumn
) -> Seq[ExprIR]:
    """Parse `DataFrame.sort` and `Expr.sort_by` keys into a flat sequence of `ExprIR` nodes."""
    return tuple(_parse_sort_by_into_iter_expr_ir(by, more_by))


# TODO @dangotbanned: Review the rejection predicate
# It doesn't cover all length-changing expressions, only aggregations/literals
def _parse_sort_by_into_iter_expr_ir(
    by: OneOrIterable[IntoExprColumn], more_by: Iterable[IntoExprColumn]
) -> Iterator[ExprIR]:
    for e in _parse_into_iter_expr_ir(by, *more_by):
        if e.is_scalar:
            msg = f"All expressions sort keys must preserve length, but got:\n{e!r}"
            raise InvalidOperationError(msg)
        yield e


def _parse_into_iter_expr_ir(
    first_input: OneOrIterable[IntoExpr],
    *more_inputs: IntoExpr | list[Any],
    _list_as_series: PartialSeries | None = None,
    **named_inputs: IntoExpr,
) -> Iterator[ExprIR]:
    if not _is_empty_sequence(first_input):
        # NOTE: These need to be separated to introduce an intersection type
        # Otherwise, `str | bytes` always passes through typing
        if _is_iterable(first_input) and not is_iterable_reject(first_input):
            if more_inputs and (
                _list_as_series is None or not isinstance(first_input, list)
            ):
                raise invalid_into_expr_error(first_input, more_inputs, named_inputs)
            # NOTE: Ensures `first_input = [False, True, True] -> lit(Series([False, True, True]))`
            elif (
                _list_as_series is not None
                and isinstance(first_input, list)
                and not is_into_expr_column(first_input[0])
            ):
                yield parse_into_expr_ir(first_input, list_as_series=_list_as_series)
            else:
                yield from _parse_positional_inputs(first_input, _list_as_series)
        else:
            yield parse_into_expr_ir(first_input, list_as_series=_list_as_series)
    else:
        # NOTE: Passthrough case for no inputs - but gets skipped when calling next
        yield from ()
    if more_inputs:
        yield from _parse_positional_inputs(more_inputs, _list_as_series)
    if named_inputs:
        yield from _parse_named_inputs(named_inputs)


def _parse_positional_inputs(
    inputs: Iterable[IntoExpr | list[Any]], /, list_as_series: PartialSeries | None = None
) -> Iterator[ExprIR]:
    for into in inputs:
        yield parse_into_expr_ir(into, list_as_series=list_as_series)


def _parse_named_inputs(named_inputs: dict[str, IntoExpr], /) -> Iterator[ExprIR]:
    for name, input in named_inputs.items():
        yield parse_into_expr_ir(input).alias(name)


def _parse_constraints(constraints: dict[str, IntoExpr], /) -> Iterator[ExprIR]:
    from narwhals._plan import col

    for name, value in constraints.items():
        yield (col(name) == value)._ir


def _combine_predicates(predicates: Iterator[ExprIR], /) -> ExprIR:
    from narwhals._plan.expressions.boolean import AllHorizontal

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
