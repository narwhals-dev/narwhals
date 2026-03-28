"""Parsing for narwhals-level expressions.

## Important
The **only** inline dependencies this module should have are for:
-  `Expr`
- `Series`
- `lit`

It should not be used in `_plan.expressions.*` at all.

These constraints allow top-level modules and the `functions` & `compliant` packages
to freely import from here.
"""

# ruff: noqa: A002
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from functools import cache, lru_cache
from itertools import chain
from typing import TYPE_CHECKING

import narwhals._plan.expressions as ir
import narwhals._plan.expressions.selectors as s_ir
from narwhals._plan._guards import is_iterable_reject
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
    from narwhals._plan.selectors import Selector
    from narwhals._plan.series import Series
    from narwhals._plan.typing import (
        IntoExpr,
        IntoExprColumn,
        OneOrIterable,
        PartialSeries,
        Seq,
    )

    T = TypeVar("T")

__all__ = [
    "into_expr_ir",
    "into_iter_expr_ir",
    "into_iter_selector_ir",  # stable
    "into_selector_ir",  # stable
    "into_seq_of_expr_ir",
    "predicates_constraints_into_expr_ir",  # functionality stable, name? eh
    "sort_by_into_seq_of_expr_ir",
]

# TODO @dangotbanned: Simplify the `list` special-casing, now that it is supported
Incomplete: TypeAlias = "Any"
"""Artifact from previous `lit(list)` rejection"""


def into_expr_ir(
    input: IntoExpr | list[Any],
    *,
    str_as_lit: bool = False,
    list_as_series: PartialSeries | None = None,
) -> ExprIR:
    """Parse a single input into an expression.

    Arguments:
        input: The input to be parsed as an expression.
        str_as_lit: Interpret string input as a string literal. If set to `False` (default),
            strings are parsed as column names.
        list_as_series: Interpret list input as a Series literal, using the provided constructor.
            If set to `None` (default), lists will raise when passed to `lit`.
    """
    if isinstance(input, _import_expr()):
        return input._ir
    if not str_as_lit and isinstance(input, str):
        return ir.col(input)
    if list_as_series is not None and isinstance(input, list):
        input = list_as_series(input)
    return _import_lit()(input)._ir


def into_seq_of_expr_ir(
    first_input: OneOrIterable[IntoExpr] = (),
    *more_inputs: IntoExpr | Incomplete,
    **named_inputs: IntoExpr,
) -> Seq[ExprIR]:
    """Parse variadic inputs into a flat sequence of expressions."""
    return tuple(
        into_iter_expr_ir(first_input, *more_inputs, _list_as_series=None, **named_inputs)
    )


def predicates_constraints_into_expr_ir(
    first_predicate: OneOrIterable[IntoExprColumn] | list[bool] = (),
    *more_predicates: IntoExprColumn | list[bool] | Incomplete,
    _list_as_series: PartialSeries | None = None,
    **constraints: IntoExpr,
) -> ExprIR:
    """Parse variadic predicates and constraints into an expression.

    The result is an AND-reduction of all inputs.
    """
    predicates = into_iter_expr_ir(
        first_predicate, *more_predicates, _list_as_series=_list_as_series
    )
    if constraints:
        items = constraints.items()
        it = (ir.col(nm).eq(into_expr_ir(v, str_as_lit=True)) for nm, v in items)
        predicates = chain(predicates, it)

    if (first := next(predicates, None)) is None:
        raise at_least_one_error("filter")
    if second := next(predicates, None):
        return ir.all_horizontal(first, second, *predicates)
    if first.meta.has_multiple_outputs():
        # NOTE: Safeguarding against https://github.com/pola-rs/polars/issues/25022
        return ir.all_horizontal(first)
    return first


def sort_by_into_seq_of_expr_ir(
    by: OneOrIterable[IntoExprColumn] = (), *more_by: IntoExprColumn
) -> Seq[ExprIR]:
    """Parse `DataFrame.sort` and `Expr.sort_by` keys into a flat sequence of expressions."""
    it = _sort_by_into_iter_expr_ir(by, more_by)
    if first := next(it, None):
        return (first, *it)
    raise at_least_one_error("sort_by")


# TODO @dangotbanned: Too complicated!
def into_iter_expr_ir(
    first_input: OneOrIterable[IntoExpr],
    *more_inputs: IntoExpr | list[Any],
    _list_as_series: PartialSeries | None = None,
    **named_inputs: IntoExpr,
) -> Iterator[ExprIR]:
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
                and not isinstance(
                    first_input[0], (str, _import_expr(), _import_series())
                )
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


# TODO @dangotbanned: Fix the rejection predicate by adding `ExprIR.is_length_preserving`
# - This doesn't cover all length-changing expressions, only aggregations/literals
# - Adapt from `window._is_filtration` and replace that in `over`
def _sort_by_into_iter_expr_ir(
    by: OneOrIterable[IntoExprColumn], more_by: Iterable[IntoExprColumn]
) -> Iterator[ExprIR]:
    for e in into_iter_expr_ir(by, *more_by):
        if e.is_scalar():
            msg = f"All expressions sort keys must preserve length, but got:\n{e!r}"
            raise InvalidOperationError(msg)
        yield e


# TODO @dangotbanned: Would be cool to have something like:
# `into_selector_ir(...).expand_names(schema, **kwds)`
def into_selector_ir(
    first_input: OneOrIterable[str | Selector],
    more_inputs: Seq[OneOrIterable[str | Selector]] = (),
    /,
    *,
    require_all: bool = True,
) -> SelectorIR:
    """Parse and reduce input(s) into a **single** selector.

    Tip:
        Prefer `into_iter_selector_ir` if there isn't a requirement for just one.

    Arguments:
        first_input: One or more column names or selectors.
        more_inputs: Use if `*args` were accepted *in-addition-to* `first_input` as syntax sugar.
        require_all: Whether to match *all* names (the default) or *any* of the names.

    Examples:
        The goal is for the final selector to be as simple as possible:
        >>> into_selector_ir("a")
        ncs.by_name('a', require_all=True)
        >>> into_selector_ir("a", ("b",))
        ncs.by_name('a', 'b', require_all=True)
        >>> into_selector_ir(["a"], ("b", "c", ["d", "e"]), require_all=False)
        ncs.by_name('a', 'b', 'c', 'd', 'e', require_all=False)
        >>> into_selector_ir(())
        ncs.empty()
        >>> into_selector_ir((), ((),))
        ncs.empty()

        And saving `__or__` for just the bits we can't (cheaply) reduce:
        >>> import narwhals._plan.selectors as ncs
        >>> into_selector_ir(("a", "b"), (ncs.integer(), ncs.float(), "c"))
        [([(ncs.by_name('a', 'b', 'c', ...)) | (ncs.integer())]) | (ncs.float())]
    """
    if not more_inputs and (
        isinstance(first_input, str) or not isinstance(first_input, Iterable)
    ):
        return _into_selector_ir(first_input, require_all=require_all)
    flat = _flatten_column_names_or_selectors((first_input, *more_inputs))
    if (first := next(flat, None)) is None:
        return s_ir.empty()
    if (second := next(flat, None)) is None:
        return _into_selector_ir(first, require_all=require_all)

    names = deque[str]()
    irs = deque[ir.SelectorIR]()
    for into in chain((first, second), flat):
        if isinstance(into, str):
            names.append(into)
        else:
            irs.append(_into_selector_ir(into, require_all=require_all))
    if not names:
        s = irs.popleft()
    else:
        s = s_ir.ByName(names=tuple(names), require_all=require_all).to_selector_ir()
    return s.or_(*irs)


def into_iter_selector_ir(
    first_input: OneOrIterable[str | Selector], more_inputs: Seq[str | Selector] = (), /
) -> Iterator[SelectorIR]:
    """Yield input(s) parsed into selector(s).

    Arguments:
        first_input: One or more column names or selectors.
        more_inputs: Use if `*args` were accepted *in-addition-to* `first_input` as syntax sugar.
    """
    if isinstance(first_input, str):
        yield s_ir.ByName.from_name(first_input).to_selector_ir()
    elif not isinstance(first_input, Iterable):
        yield _into_selector_ir(first_input)
    elif more_inputs:
        raise invalid_into_expr_error(first_input, more_inputs, {})
    else:
        for into in first_input:
            yield _into_selector_ir(into)
    for into in more_inputs:
        yield _into_selector_ir(into)


def _into_selector_ir(
    input: str | Selector | Expr, /, *, require_all: bool = True
) -> SelectorIR:
    if isinstance(input, _import_expr()):
        return input._ir.to_selector_ir()
    if isinstance(input, str):
        return s_ir.ByName.from_name(input, require_all=require_all).to_selector_ir()
    msg = f"cannot turn {qualified_type_name(input)!r} into a selector"
    raise TypeError(msg)


def _flatten_column_names_or_selectors(
    iterable: Iterable[OneOrIterable[str | Selector]], /
) -> Iterator[str | Selector]:
    # A more restrictive & cheaper version of `common.flatten_hash_safe`
    for element in iterable:
        if isinstance(element, str) or not isinstance(element, Iterable):
            yield element
        else:
            yield from _flatten_column_names_or_selectors(element)


# TODO @dangotbanned: make this `_is_iterable_raise_native` (or something)
# TODO @dangotbanned: Use just the `_is_iterable` part for selectors
# - The base case in `_into_selector_ir` *could* do this extra error before raising the more general thing
# - `into_expr_ir` should probably handle this differently too
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


# fmt: off
@lru_cache(maxsize=128)
def _type_cached_is_iterable(tp: type[Any], /) -> bool:
    return issubclass(tp, Iterable)
@cache
def _import_expr() -> type[Expr]:
    from narwhals._plan.expr import Expr
    return Expr
@cache
def _import_series() -> type[Series]:
    from narwhals._plan.series import Series
    return Series
@cache
def _import_lit() -> Callable[[Any], Expr]:
    from narwhals._plan.functions.literal import lit
    return lit
# fmt: on
