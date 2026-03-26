from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

# ruff: noqa: A002
from functools import lru_cache
from itertools import chain
from typing import TYPE_CHECKING

import narwhals._plan.expressions as ir
import narwhals._plan.expressions.selectors as s_ir
from narwhals._plan._guards import is_iterable_reject
from narwhals._plan.common import flatten_hash_safe
from narwhals._plan.exceptions import (
    at_least_one_error,
    invalid_into_expr_error,
    is_iterable_error,
)
from narwhals._utils import qualified_type_name
from narwhals.dependencies import get_pandas, get_polars
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, TypeVar

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

# TODO @dangotbanned: Remaining inline imports should share a cache
# - Dont cache modules
# - Try to use as few functions/types as possible


# NOTE: Unavoidable inline imports
def parse_into_expr_ir(
    input: IntoExpr | list[Any],
    *,
    str_as_lit: bool = False,
    list_as_series: PartialSeries | None = None,
) -> ExprIR:
    """Parse a single input into an `ExprIR` node.

    Arguments:
        input: The input to be parsed as an expression.
        str_as_lit: Interpret string input as a string literal. If set to `False` (default),
            strings are parsed as column names.
        list_as_series: Interpret list input as a Series literal, using the provided constructor.
            If set to `None` (default), lists will raise when passed to `lit`.
    """
    from narwhals._plan import Expr, lit

    if isinstance(input, Expr):
        return input._ir
    if not str_as_lit and isinstance(input, str):
        return ir.col(input)
    if list_as_series is not None and isinstance(input, list):
        input = list_as_series(input)
    # NOTE: Raises in `lit` when we pass a `list`
    return lit(input)._ir  # type: ignore[arg-type]


# NOTE: Unavoidable inline import
def parse_into_selector_ir(
    input: ColumnNameOrSelector | Expr, /, *, require_all: bool = True
) -> SelectorIR:
    from narwhals._plan import Expr

    if isinstance(input, Expr):
        return input._ir.to_selector_ir()
    if isinstance(input, str):
        return s_ir.ByName.from_name(input, require_all=require_all).to_selector_ir()
    msg = f"cannot turn {qualified_type_name(input)!r} into a selector"
    raise TypeError(msg)


def parse_into_combined_selector_ir(
    *inputs: OneOrIterable[ColumnNameOrSelector], require_all: bool = True
) -> SelectorIR:
    flat = tuple(flatten_hash_safe(inputs))
    selectors = deque[ir.SelectorIR]()
    if names := tuple(el for el in flat if isinstance(el, str)):
        selector_ir = s_ir.ByName(names=names, require_all=require_all).to_selector_ir()
        if len(names) == len(flat):
            return selector_ir
        selectors.append(selector_ir)
    selectors.extend(parse_into_selector_ir(el) for el in flat if not isinstance(el, str))
    if not selectors:
        return s_ir.empty()
    return selectors.popleft().or_(*selectors)


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
    predicates = _parse_into_iter_expr_ir(
        first_predicate, *more_predicates, _list_as_series=_list_as_series
    )
    if constraints:
        items = constraints.items()
        it = (ir.col(nm).eq(parse_into_expr_ir(v, str_as_lit=True)) for nm, v in items)
        predicates = chain(predicates, it)

    if (first := next(predicates, None)) is None:
        raise at_least_one_error("filter")
    if second := next(predicates, None):
        return ir.all_horizontal(first, second, *predicates)
    if first.meta.has_multiple_outputs():
        # NOTE: Safeguarding against https://github.com/pola-rs/polars/issues/25022
        return ir.all_horizontal(first)
    return first


def parse_sort_by_into_seq_of_expr_ir(
    by: OneOrIterable[IntoExprColumn] = (), *more_by: IntoExprColumn
) -> Seq[ExprIR]:
    """Parse `DataFrame.sort` and `Expr.sort_by` keys into a flat sequence of `ExprIR` nodes."""
    it = _parse_sort_by_into_iter_expr_ir(by, more_by)
    if first := next(it, None):
        return (first, *it)
    raise at_least_one_error("sort_by")


# TODO @dangotbanned: Fix the rejection predicate by adding `ExprIR.is_length_preserving`
# - This doesn't cover all length-changing expressions, only aggregations/literals
# - Adapt from `window._is_filtration` and replace that in `over`
def _parse_sort_by_into_iter_expr_ir(
    by: OneOrIterable[IntoExprColumn], more_by: Iterable[IntoExprColumn]
) -> Iterator[ExprIR]:
    for e in _parse_into_iter_expr_ir(by, *more_by):
        if e.is_scalar():
            msg = f"All expressions sort keys must preserve length, but got:\n{e!r}"
            raise InvalidOperationError(msg)
        yield e


# TODO @dangotbanned: Possibly remove? see `_parse_into_iter_selector_ir`
def parse_into_seq_of_selector_ir(
    first_input: OneOrIterable[ColumnNameOrSelector], *more_inputs: ColumnNameOrSelector
) -> Seq[SelectorIR]:
    return tuple(_parse_into_iter_selector_ir(first_input, more_inputs))


# NOTE: ^^^ is the exposed one,
# but collecting -> `expand_selector_irs_names(require_any=...)` seems a bit much
# `require_any` should be enough, right?
def _parse_into_iter_selector_ir(
    first_input: OneOrIterable[ColumnNameOrSelector],
    more_inputs: tuple[ColumnNameOrSelector, ...],
    /,
) -> Iterator[SelectorIR]:
    # TODO @dangotbanned: Is this import avoidable?
    from narwhals._plan import Expr

    if isinstance(first_input, (str, Expr)) and not more_inputs:
        yield parse_into_selector_ir(first_input)
        return

    if not _is_empty_sequence(first_input):
        if _is_iterable(first_input) and not isinstance(first_input, str):
            if more_inputs:
                raise invalid_into_expr_error(first_input, more_inputs, {})
            else:
                for into in first_input:  # type: ignore[var-annotated]
                    yield parse_into_selector_ir(into)
        else:
            yield parse_into_selector_ir(first_input)
    for into in more_inputs:
        yield parse_into_selector_ir(into)


# NOTE: Unavoidable inline imports
# TODO @dangotbanned: Too complicated!
def _parse_into_iter_expr_ir(
    first_input: OneOrIterable[IntoExpr],
    *more_inputs: IntoExpr | list[Any],
    _list_as_series: PartialSeries | None = None,
    **named_inputs: IntoExpr,
) -> Iterator[ExprIR]:
    from narwhals._plan import Expr, Series

    into_expr_ir = parse_into_expr_ir
    as_series = _list_as_series
    if not _is_empty_sequence(first_input):
        # NOTE: These need to be separated to introduce an intersection type
        # Otherwise, `str | bytes` always passes through typing
        if _is_iterable(first_input) and not is_iterable_reject(first_input):
            if more_inputs and (as_series is None or not isinstance(first_input, list)):
                raise invalid_into_expr_error(first_input, more_inputs, named_inputs)

            if (
                as_series is not None
                and isinstance(first_input, list)
                and not isinstance(first_input[0], (str, Expr, Series))
            ):
                # NOTE: Ensures `first_input = [False, True, True] -> lit(Series([False, True, True]))`
                yield into_expr_ir(first_input, list_as_series=as_series)
            else:
                for into in first_input:
                    yield into_expr_ir(into, list_as_series=as_series)
        else:
            yield into_expr_ir(first_input, list_as_series=as_series)
    for into in more_inputs:
        yield into_expr_ir(into, list_as_series=as_series)
    for name, input in named_inputs.items():
        yield into_expr_ir(input).alias(name)


def _is_iterable(obj: Iterable[T] | Any) -> TypeIs[Iterable[T]]:
    """Equivalent to `isinstance(obj, Iterable)` but raises on native types.

    Used on a very hot path, so subclass checks are cached.
    """
    tp = type(obj)
    if _type_cached_is_iterable_is_native(tp):  # type: ignore[arg-type]
        raise is_iterable_error(obj)
    return _type_cached_is_iterable(tp)  # type: ignore[arg-type]


def _is_empty_sequence(obj: Any) -> bool:
    return isinstance(obj, Sequence) and not obj


@lru_cache(maxsize=128)
def _type_cached_is_iterable_is_native(tp: type[Any], /) -> bool:
    tps: tuple[type[Any], ...] = (pd.DataFrame, pd.Series) if (pd := get_pandas()) else ()
    tps = (
        (*tps, pl.Series, pl.Expr, pl.DataFrame, pl.LazyFrame)
        if (pl := get_polars())
        else tps
    )
    return bool(tps and issubclass(tp, tps))


@lru_cache(maxsize=128)
def _type_cached_is_iterable(tp: type[Any], /) -> bool:
    return issubclass(tp, Iterable)
