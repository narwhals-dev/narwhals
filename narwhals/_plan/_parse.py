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
from collections.abc import Callable, Iterable
from functools import cache, lru_cache
from itertools import chain
from typing import TYPE_CHECKING, Any, TypeVar

import narwhals._plan.expressions as ir
import narwhals._plan.expressions.selectors as s_ir
from narwhals._plan.exceptions import (
    at_least_one_error,
    invalid_into_expr_error,
    is_iterable_error,
    sort_by_key_length_changing_error,
)
from narwhals._utils import qualified_type_name
from narwhals.dependencies import get_pandas, get_polars

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import TypeAlias, TypeIs

    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions import ExprIR, SelectorIR
    from narwhals._plan.selectors import Selector
    from narwhals._plan.series import Series
    from narwhals._plan.typing import IntoExpr, OneOrIterable, PartialSeries, Seq


__all__ = [
    "into_expr_ir",
    "into_iter_expr_ir",  # stable
    "into_iter_selector_ir",  # stable
    "into_selector_ir",  # stable
    "predicates_constraints_into_expr_ir",  # functionality stable, name? eh
    "sort_by_into_iter_expr_ir",  # stable
]

T = TypeVar("T")
Incomplete: TypeAlias = "Any"
"""Marks unsound `*exprs: OneOrIterable[IntoExpr]` unpacking.

Some functions use the `expr: OneOrIterable[IntoExpr], *more: IntoExpr` pattern, but not all.
"""


# TODO @dangotbanned: `str_as_lit` maybe stays, but could do some inheritance instead?
def into_expr_ir(input: IntoExpr | dict[Any, Any], *, str_as_lit: bool = False) -> ExprIR:
    """Parse a single input into an expression.

    Arguments:
        input: The input to be parsed as an expression.
        str_as_lit: Interpret string input as a string literal. If set to `False` (default),
            strings are parsed as column names.
    """
    if isinstance(input, _import_expr()):
        return input._ir
    if not str_as_lit and isinstance(input, str):
        return ir.col(input)
    return _import_lit()(input)._ir


def predicates_constraints_into_expr_ir(
    first_predicate: OneOrIterable[IntoExpr] = (),
    *more_predicates: IntoExpr | Incomplete,
    **constraints: IntoExpr,
) -> ExprIR:
    """Parse variadic predicates and constraints into an expression.

    The result is an AND-reduction of all inputs.
    """
    predicates = into_iter_expr_ir(first_predicate, *more_predicates)
    return _predicates_constraints_into_expr_ir(predicates, constraints)


def df_filter_predicates_constraints_into_expr_ir(
    first_predicate: OneOrIterable[IntoExpr] = (),
    *more_predicates: IntoExpr | Incomplete,
    _into_series: PartialSeries,
    **constraints: IntoExpr,
) -> ExprIR:
    """Special-casing for `DataFrame.filter` boolean masks.

    **Eager-only**, since this requires creating new `Series`:

        [False, True, True] -> lit(Series([False, True, True]))
    """
    predicates = _df_filter_into_iter_expr_ir(
        first_predicate, more_predicates, into_series=_into_series
    )
    return _predicates_constraints_into_expr_ir(predicates, constraints)


def _predicates_constraints_into_expr_ir(
    predicates: Iterator[ExprIR], constraints: dict[str, IntoExpr], /
) -> ExprIR:
    if constraints:
        items = constraints.items()
        it = (ir.col(nm).eq(into_expr_ir(v, str_as_lit=True)) for nm, v in items)
        predicates = chain(predicates, it)

    if (first := next(predicates, None)) is None:
        raise at_least_one_error("filter")
    if second := next(predicates, None):
        return ir.all_horizontal(first, second, *predicates)
    return first


def sort_by_into_iter_expr_ir(
    by: OneOrIterable[Expr | str], more_by: Seq[Expr | str] = (), /
) -> Iterator[ExprIR]:
    it = into_iter_expr_ir(by, *more_by)
    if (first := next(it, None)) is None:
        raise at_least_one_error("sort_by")
    if not first.is_length_preserving():
        raise sort_by_key_length_changing_error(first)
    yield first
    for e in it:
        if not e.is_length_preserving():
            raise sort_by_key_length_changing_error(e)
        yield e


def into_iter_expr_ir(
    first_input: OneOrIterable[IntoExpr] = (),
    *more_inputs: IntoExpr | Incomplete,
    **named_inputs: IntoExpr,
) -> Iterator[ExprIR]:
    """Yield variadic inputs parsed into expressions.

    Arguments:
        first_input: Input(s) to be parsed as expressions.
        *more_inputs: Additional inputs to parse.
        **named_inputs: Keyword-arguments from one of `select`, `with_columns`, `group_by`.
            The columns will be renamed to the keyword used.

    *Most* cases are covered by these rules, see examples for special-cases:

        # IntoExpr: TypeAlias = Expr | str | Series | PythonLiteral
        Expr                   -> ExprIR
        str                    -> Column
        Series                 -> LitSeries
        PythonLiteral          -> Lit

    Examples:
        `first_input` is separate as how it is parsed *depends on* `more_inputs`.

        >>> def parse(*args, **kwds):
        ...     return list(into_iter_expr_ir(*args, **kwds))

        The general case is that we support these as all meaning, *select those two columns*:
        >>> parse("one", "two")
        [col('one'), col('two')]
        >>> parse(["one", "two"])
        [col('one'), col('two')]
        >>> parse("one", "two")
        [col('one'), col('two')]
        >>> parse(iter(("one", "two")))
        [col('one'), col('two')]

        Whereas `list | tuple` will otherwise be considered literals:
        >>> parse("one", ["two"])
        [col('one'), lit(list: ['two'])]
        >>> parse(["one"], "two")
        [lit(list: ['one']), col('two')]
        >>> parse(["one"], ("two",))
        [lit(list: ['one']), lit(list: ['two'])]

        *This behavior matches `polars`*
    """
    first = first_input
    always_expr_or_lit = (_import_series(), _import_expr(), str, bytes, dict)
    if (isinstance(first, always_expr_or_lit) or not _is_iterable(first)) or (
        more_inputs and isinstance(first, (list, tuple))
    ):
        yield into_expr_ir(first)  # type: ignore[arg-type]
    else:
        for into in first:
            yield into_expr_ir(into)
    for into in more_inputs:
        yield into_expr_ir(into)
    for name, input in named_inputs.items():
        yield into_expr_ir(input).alias(name)


def _df_filter_into_iter_expr_ir(
    first_predicate: OneOrIterable[IntoExpr],
    more_predicates: Iterable[IntoExpr | list[Any]],
    into_series: PartialSeries,
) -> Iterator[ExprIR]:
    first = first_predicate
    expr, lit = _import_expr(), _import_lit()
    non_mask_fast = (_import_series(), expr, str, bytes)
    if isinstance(first, non_mask_fast) or not _is_iterable(first):
        yield into_expr_ir(first)  # type: ignore[arg-type]
    elif isinstance(first, list) and first and not isinstance(first[0], non_mask_fast):
        more_predicates = chain([first], more_predicates)
    else:
        more_predicates = chain(first, more_predicates)
    for p in more_predicates:
        if isinstance(p, expr):
            yield p._ir
        elif isinstance(p, str):
            yield ir.col(p)
        else:
            yield lit(into_series(p) if isinstance(p, list) else p)._ir


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
        ncs.by_name('a')
        >>> into_selector_ir("a", ("b",))
        ncs.by_name('a', 'b')
        >>> into_selector_ir(["a"], ("b", "c", ["d", "e"]), require_all=False)
        ncs.by_name('a', 'b', 'c', 'd', 'e', require_all=False)
        >>> into_selector_ir(())
        ncs.empty()
        >>> into_selector_ir((), ((),))
        ncs.empty()

        And saving `__or__` for just the bits we can't (cheaply) reduce:
        >>> import narwhals._plan.selectors as ncs
        >>> into_selector_ir(("a", "b"), (ncs.integer(), ncs.float(), "c"))
        [[ncs.by_name('a', 'b', 'c') | ncs.integer()] | ncs.float()]
    """
    if not more_inputs and (
        isinstance(first_input, str) or not isinstance(first_input, Iterable)
    ):
        return _into_selector_ir(first_input, require_all=require_all)
    flat = _flatten_column_names_or_selectors((first_input, *more_inputs))
    if (first := next(flat, None)) is None:
        return s_ir.Empty()
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
        s = s_ir.ByName(names=tuple(names), require_all=require_all)
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
        yield s_ir.ByName.from_name(first_input)
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
        return s_ir.ByName.from_name(input, require_all=require_all)
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
