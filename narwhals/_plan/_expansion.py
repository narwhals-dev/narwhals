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
from typing import TYPE_CHECKING, Any, Union

from narwhals._plan import common, expressions as ir, meta
from narwhals._plan.exceptions import (
    binary_expr_multi_output_error,
    column_not_found_error,
    duplicate_error,
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
from narwhals._utils import check_column_names_are_unique, zip_strict

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, Sequence

    from typing_extensions import TypeAlias

    from narwhals._plan.typing import Ignored, Seq


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
    ignored: Ignored = (),
    *,
    schema: IntoFrozenSchema,
    require_any: bool = False,
) -> OutputNames:
    """Expand selector-only input into the column names that match.

    Similar to `prepare_projection`, but intended for allowing a subset of `Expr` and all `Selector`s
    to be used in more places like `DataFrame.{drop,sort,partition_by}`.

    Arguments:
        selectors: IRs that **only** contain subclasses of `SelectorIR`.
        ignored: Names of `group_by` columns.
        schema: Scope to expand selectors in.
        require_any: Raise if the entire expansion selected zero columns.
    """
    expander = Expander(schema, ignored)
    if names := tuple(expander.iter_expand_selector_names(selectors)):
        if len(names) != len(set(names)):
            # NOTE: Can't easily reuse `duplicate_error`, falling back to main for now
            check_column_names_are_unique(names)
    elif require_any:
        raise selectors_not_found_error(selectors, expander.schema)
    return names


def remove_alias(origin: ExprIR, /) -> ExprIR:
    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr if isinstance(child, (Alias, RenameAlias)) else child

    return origin.map_ir(fn)


def replace_keep_name(origin: ExprIR, /) -> ExprIR:
    root_name = meta.root_name_first(origin)

    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr.alias(root_name) if isinstance(child, KeepName) else child

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

    def prepare_projection(self, exprs: Collection[ExprIR], /) -> Seq[NamedIR]:
        output_names = deque[str]()
        named_irs = deque[NamedIR]()
        root_names = set[str]()

        # NOTE: Collecting here isn't ideal (perf-wise), but the expanded `ExprIR`s
        # have more useful information to add in an error message
        # Another option could be keeping things lazy, but repeating the work for the error case?
        # that way, there isn't a cost paid on the happy path - and it doesn't matter when we're raising
        # if we take our time displaying the message
        expanded = tuple(self.iter_expand_exprs(exprs))
        for e in expanded:
            # NOTE: Empty string is allowed as a name, but is falsy
            if (name := e.meta.output_name(raise_if_undetermined=False)) is not None:
                target = e
            elif meta.has_expr_ir(e, KeepName):
                replaced = replace_keep_name(e)
                name = replaced.meta.output_name()
                target = replaced
            else:
                msg = f"Unable to determine output name for expression, got: `{e!r}`"
                raise NotImplementedError(msg)
            output_names.append(name)
            named_irs.append(ir.named_ir(name, remove_alias(target)))
            root_names.update(meta.iter_root_names(e))
        if len(output_names) != len(set(output_names)):
            raise duplicate_error(expanded)
        if not (set(self.schema).issuperset(root_names)):
            raise column_not_found_error(root_names, self.schema)
        return tuple(named_irs)

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
        else:
            msg = f"Didn't expect to see {type(origin).__name__}"
            raise NotImplementedError(msg)

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
            replaced = common.replace(origin, **changes)
            for root in self._expand_recursive(replaced.expr):
                yield common.replace(replaced, expr=root)
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
        if origin.options.is_input_wildcard_expansion():
            reduced = tuple(self._expand_inner(origin.input))
            yield origin.__replace__(input=reduced)
        else:
            if non_root := origin.input[1:]:
                children = tuple(self._expand_only(origin, child) for child in non_root)
            else:
                children = ()
            for root in self._expand_recursive(origin.input[0]):
                yield origin.__replace__(input=(root, *children))


_EXPAND_NONE = (ir.Column, ir.Literal, ir.Len)
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
