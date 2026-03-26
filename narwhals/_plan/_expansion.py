"""Based on [polars-plan/src/plans/conversion/expr_expansion.rs].

## Notes
- Goal is to expand every selection into a named column.
- Most will require only the column names of the schema.

## Current `narwhals`
As of [6e57eff4f059c748cf84ddcae276a74318720b85], many of the problems
this module would solve *currently* have solutions distributed throughout `narwhals`.

Their dependencies are **quite** complex, with the main ones being:
- `CompliantExpr`
  - _evaluate_output_names
    - `CompliantSelector.__(sub|or|and|invert)__`
    - `CompliantThen._evaluate_output_names`
  - _alias_output_names
  - from_column_names, from_column_indices
    - `CompliantNamespace.(all|col|exclude|nth)`
  - _eval_names_indices
  - _evaluate_aliases
    - `Compliant*Frame._evaluate_aliases`
    - `EagerDataFrame._evaluate_into_expr(s)`
- `CompliantExprNameNamespace`
  - EagerExprNameNamespace
  - LazyExprNameNamespace
- `_expression_parsing.py`
  - combine_evaluate_output_names
    - 6-7x per `CompliantNamespace`
  - combine_alias_output_names
    - 6-7x per `CompliantNamespace`
  - evaluate_output_names_and_aliases
    - Depth tracking (`Expr.over`, `GroupyBy.agg`)

[polars-plan/src/plans/conversion/expr_expansion.rs]: https://github.com/pola-rs/polars/blob/df4d21c30c2b383b651e194f8263244f2afaeda3/crates/polars-plan/src/plans/conversion/expr_expansion.rs
[6e57eff4f059c748cf84ddcae276a74318720b85]: https://github.com/narwhals-dev/narwhals/commit/6e57eff4f059c748cf84ddcae276a74318720b85
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from itertools import chain
from typing import TYPE_CHECKING, Any, Union

from narwhals._plan import expressions as ir, meta
from narwhals._plan._function import HorizontalFunction
from narwhals._plan._parse import parse_into_iter_selector_ir
from narwhals._plan.exceptions import (
    binary_expr_multi_output_error,
    column_not_found_error,
    duplicate_error,
    duplicate_names_error,
    expand_multi_output_error,
    selectors_not_found_error,
)
from narwhals._plan.expressions import (
    Alias,
    ExprIR,
    KeepName,
    NamedIR,
    RenameAlias,
    SelectorIR,
)
from narwhals._plan.schema import FrozenSchema, IntoFrozenSchema, freeze_schema
from narwhals._typing_compat import assert_never
from narwhals._utils import zip_strict
from narwhals.exceptions import ComputeError, InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

    from typing_extensions import TypeAlias

    from narwhals._plan.selectors import Selector
    from narwhals._plan.typing import Ignored, OneOrIterable, Seq

OutputNames: TypeAlias = "Seq[str]"
"""Fully expanded, validated output column names, for `NamedIR`s."""


Combination: TypeAlias = Union[
    ir.SortBy, ir.BinaryExpr, ir.TernaryExpr, ir.Filter, ir.OverOrdered, ir.Over
]


def prepare_projection(
    exprs: Sequence[ExprIR], /, ignored: Ignored = (), *, schema: IntoFrozenSchema
) -> tuple[Seq[NamedIR], FrozenSchema]:
    """Expand IRs into named column projections.

    **Primary entry-point**, for `select`, `with_columns`,
    and any other context that requires resolving expression names.

    Arguments:
        exprs: IRs that *may* contain arbitrarily nested expressions.
        ignored: Names of `group_by` columns.
        schema: Scope to expand selectors in.
    """
    expander = Expander(schema, ignored)
    return expander.prepare_projection(exprs), expander.schema


def expand_selector_irs_names(
    selectors: Sequence[SelectorIR],
    /,
    *,
    schema: IntoFrozenSchema,
    require_any: bool = True,
) -> OutputNames:
    """Expand selector-only input into the column names that match.

    Similar to `prepare_projection`, but intended for allowing a subset of `Expr` and all `Selector`s
    to be used in more places like `DataFrame.{drop,sort,partition_by}`.

    Arguments:
        selectors: IRs that **only** contain subclasses of `SelectorIR`.
        ignored: Names of `group_by` columns.
        schema: Scope to expand selectors in.
        require_any: If True (default) raise if the entire expansion selected zero columns.
    """
    expander = Expander(schema, ())
    if (names := expander.expand_selector_names(selectors)) or not require_any:
        return names
    raise selectors_not_found_error(selectors, expander.schema)


def parse_expand_selectors(
    first_input: OneOrIterable[str | Selector],
    more_inputs: tuple[str | Selector, ...] = (),
    /,
    *,
    schema: IntoFrozenSchema,
    require_any: bool = True,
) -> OutputNames:
    """Convert input(s) into selector(s), expanding them into the column names that match.

    Semantically equivalent to these independent steps:

        irs: tuple[SelectorIR, ...] = parse_into_seq_of_selector_ir(first_input, more_inputs)
        output_names: tuple[str, ...] = expand_selector_irs_names(irs, schema=..., require_any=...)

    With the possibility of performing the entire operation in a single pass.

    Arguments:
        first_input: One or more column names or selectors.
        more_inputs: Use if `*args` were accepted *in-addition-to* `first_input` as syntax sugar.
        schema: Scope to expand selectors in.
        require_any: If True (default) raise if the entire expansion selected zero columns.
            If False, we can always defer iterator collection until finishing expansion.
    """
    expander = Expander(schema, ())
    into_iter = parse_into_iter_selector_ir
    expand = expander.expand_selector_names
    first, more = first_input, more_inputs

    if not require_any:
        return expand(into_iter(first, more))
    # Balancing act to keep enough context for an error message,
    # but avoid collection if we can just repeat on fail
    if not isinstance(first, Iterator) and (names := expand(into_iter(first, more))):
        return names
    parsed = tuple(into_iter(first, more))
    if names := expand(parsed):
        return names
    raise selectors_not_found_error(parsed, expander.schema)


def is_duplicated(names: Collection[str]) -> bool:
    return len(names) != len(set(names))


def remove_alias(origin: ExprIR, /) -> ExprIR:
    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr if isinstance(child, (Alias, RenameAlias)) else child

    return origin.map_ir(fn)


# TODO @dangotbanned: Update error message to use `name.keep`, (`KeepName` is the variant name only)
def replace_keep_name(origin: ExprIR, /) -> ExprIR:
    if (name := next(meta.iter_root_names(origin), None)) is None:
        msg = f"`name.keep_name` expected at least one column name, got `{origin!r}`"
        raise InvalidOperationError(msg)

    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr.alias(name) if isinstance(child, KeepName) else child

    return origin.map_ir(fn)


class Expander:
    __slots__ = ("ignored", "schema")
    schema: FrozenSchema
    ignored: Ignored

    def __init__(self, scope: IntoFrozenSchema, ignored: Ignored = ()) -> None:
        self.schema = freeze_schema(scope)
        self.ignored = ignored

    def iter_expand_exprs(self, exprs: Iterable[ExprIR], /) -> Iterator[ExprIR]:
        # Iteratively expand all of exprs
        for expr in exprs:
            yield from self._expand(expr)

    def iter_expand_selector_names(
        self, selectors: Iterable[SelectorIR], /
    ) -> Iterator[str]:
        for s in selectors:
            yield from s.iter_expand_names(self.schema, self.ignored)

    def expand_selector_names(
        self, selectors: Iterable[SelectorIR], /, *, check_unique: bool = True
    ) -> OutputNames:
        names = tuple(self.iter_expand_selector_names(selectors))
        if check_unique and is_duplicated(names):
            raise duplicate_names_error(names)
        return names

    def prepare_projection(self, exprs: Collection[ExprIR], /) -> Seq[NamedIR]:
        named_irs, _ = self._prepare_projection(exprs)
        return named_irs

    def _prepare_projection(
        self, exprs: Collection[ExprIR], /
    ) -> tuple[Seq[NamedIR], deque[str]]:
        output_names = deque[str]()
        named_irs = deque[NamedIR]()
        root_names = deque[Iterator[str]]()
        expand = self.iter_expand_exprs
        for e in expand(exprs):
            # NOTE: "" is allowed as a name, but falsy
            if (name := e.meta.output_name(raise_if_undetermined=False)) is not None:
                target = remove_alias(e)
            else:
                replaced = replace_keep_name(e)
                name = replaced.meta.output_name()
                target = remove_alias(replaced)
            output_names.append(name)
            named_irs.append(ir.named_ir(name, target))
            root_names.append(meta.iter_root_names(e))

        # NOTE: On failure, we repeat the expansion so the happy path doesn't need to collect as much
        if is_duplicated(output_names):
            raise duplicate_error(tuple(expand(exprs)))
        if not self.schema.contains_all(root_names):
            roots = chain.from_iterable(meta.iter_root_names(e) for e in expand(exprs))
            raise column_not_found_error(roots, self.schema)
        return tuple(named_irs), output_names

    def _expand(self, expr: ExprIR, /) -> Iterator[ExprIR]:
        # For a single expr, fully expand all parts of it
        if all(not e.needs_expansion() for e in expr.iter_left()):
            yield expr
        else:
            yield from self._expand_recursive(expr)

    def _expand_recursive(self, origin: ExprIR, /) -> Iterator[ExprIR]:
        # Dispatch the kind of expansion, based on the type of expr
        # Every other method will call back here
        # Based on https://github.com/pola-rs/polars/blob/5b90db75911c70010d0c0a6941046e6144af88d4/crates/polars-plan/src/plans/conversion/dsl_to_ir/expr_expansion.rs#L253-L850
        if isinstance(origin, _EXPAND_NONE):
            yield origin
        elif isinstance(origin, ir.SelectorIR):
            names = origin.iter_expand_names(self.schema, self.ignored)
            yield from (ir.Column(name=name) for name in names)
        elif isinstance(origin, _EXPAND_SINGLE):
            for expr in self._expand_recursive(origin.expr):
                yield origin.__replace__(expr=expr)
        elif isinstance(origin, _EXPAND_COMBINATION):
            yield from self._expand_combination(origin)
        elif isinstance(origin, ir.FunctionExpr):
            yield from self._expand_function_expr(origin)
        # TODO @dangotbanned: Avoidable by replacing class-specific branching -> methods
        else:  # pragma: no cover
            msg = f"Didn't expect to see {type(origin).__name__}"
            raise ComputeError(msg)

    def _expand_inner(self, children: Seq[ExprIR], /) -> Iterator[ExprIR]:
        """Use when we want to expand non-root nodes, *without* duplicating the root.

        If we wrote:

            col("a").over(col("c", "d", "e"))

        Then the expanded version should be:

            col("a").over(col("c"), col("d"), col("e"))

        An **incorrect** output would cause an error without aliasing:

            col("a").over(col("c"))
            col("a").over(col("d"))
            col("a").over(col("e"))

        And cause an error if we needed to expand both sides:

            col("a", "b").over(col("c", "d", "e"))

        Since that would become:

            col("a").over(col("c"))
            col("b").over(col("d"))
            col(<MISSING>).over(col("e"))  # InvalidOperationError: cannot combine selectors that produce a different number of columns (3 != 2)
        """
        # used by
        # - `_expand_combination` (tuple fields)
        # - `_expand_function_expr` (horizontal)
        for child in children:
            yield from self._expand_recursive(child)

    def _expand_only(self, origin: ExprIR, child: ExprIR, /) -> ExprIR:
        # used by
        # - `_expand_combination` (ExprIR fields)
        # - `_expand_function_expr` (all others that have len(inputs)>=2, call on non-root)
        iterable = self._expand_recursive(child)
        first = next(iterable)
        if second := next(iterable, None):
            raise expand_multi_output_error(origin, child, first, second, *iterable)
        return first

    # TODO @dangotbanned: It works, but all this class-specific branching belongs in the classes themselves
    def _expand_combination(self, origin: Combination, /) -> Iterator[Combination]:
        changes: dict[str, Any] = {}
        if isinstance(origin, (ir.Over, ir.Filter, ir.SortBy)):
            if isinstance(origin, ir.Over):
                if partition_by := origin.partition_by:
                    changes["partition_by"] = tuple(self._expand_inner(partition_by))
                if isinstance(origin, ir.OverOrdered):
                    changes["order_by"] = tuple(self._expand_inner(origin.order_by))
            elif isinstance(origin, ir.SortBy):
                changes["by"] = tuple(self._expand_inner(origin.by))
            else:
                changes["by"] = self._expand_only(origin, origin.by)
            replaced = origin.__replace__(**changes)
            for root in self._expand_recursive(replaced.expr):
                yield replaced.__replace__(expr=root)
        elif isinstance(origin, ir.BinaryExpr):
            yield from self._expand_binary_expr(origin)
        elif isinstance(origin, ir.TernaryExpr):
            changes["truthy"] = self._expand_only(origin, origin.truthy)
            changes["predicate"] = self._expand_only(origin, origin.predicate)
            changes["falsy"] = self._expand_only(origin, origin.falsy)
            yield origin.__replace__(**changes)
        else:
            assert_never(origin)

    def _expand_binary_expr(self, origin: ir.BinaryExpr, /) -> Iterator[ir.BinaryExpr]:
        it_lefts = self._expand_recursive(origin.left)
        it_rights = self._expand_recursive(origin.right)
        # NOTE: Fast-path that doesn't require collection
        # - Will miss selectors that expand to 1 column
        if not origin.meta.has_multiple_outputs():
            for left, right in zip_strict(it_lefts, it_rights):
                yield origin.__replace__(left=left, right=right)
            return
        # NOTE: Covers 1:1 (where either is a selector), N:N
        lefts, rights = tuple(it_lefts), tuple(it_rights)
        len_left, len_right = len(lefts), len(rights)
        if len_left == len_right:
            for left, right in zip_strict(lefts, rights):
                yield origin.__replace__(left=left, right=right)
        # NOTE: 1:M
        elif len_left == 1:
            binary = origin.__replace__(left=lefts[0])
            yield from (binary.__replace__(right=right) for right in rights)
        # NOTE: M:1
        elif len_right == 1:
            binary = origin.__replace__(right=rights[0])
            yield from (binary.__replace__(left=left) for left in lefts)
        else:
            raise binary_expr_multi_output_error(origin, lefts, rights)

    def _expand_function_expr(
        self, origin: ir.FunctionExpr, /
    ) -> Iterator[ir.FunctionExpr]:
        if isinstance(origin.function, HorizontalFunction):
            reduced = tuple(self._expand_inner(origin.input))
            yield origin.__replace__(input=reduced)
        else:
            if non_root := origin.input[1:]:
                children = tuple(self._expand_only(origin, child) for child in non_root)
            else:
                children = ()
            for root in self._expand_recursive(origin.input[0]):
                yield origin.__replace__(input=(root, *children))


_EXPAND_NONE = (ir.Column, ir.Lit, ir.LitSeries, ir.Len)
"""we're at the root, nothing left to expand."""
_EXPAND_SINGLE = (ir.Alias, ir.Cast, ir.AggExpr, ir.Sort, ir.KeepName, ir.RenameAlias)
"""one (direct) child, always stored in `self.expr`.

An expansion will always just be cloning *everything but* `self.expr`,
we only need to be concerned with a **single** attribute.

Say we had:

    origin = Cast(expr=ByName(names=("one", "two"), require_all=True), dtype=String)

This would expand to:

    cast_one = Cast(expr=Column(name="one"), dtype=String)
    cast_two = Cast(expr=Column(name="two"), dtype=String)

"""
_EXPAND_COMBINATION = (
    ir.SortBy,
    ir.BinaryExpr,
    ir.TernaryExpr,
    ir.Filter,
    ir.OverOrdered,
    ir.Over,
)
"""more than one (direct) child and those can be nested."""
