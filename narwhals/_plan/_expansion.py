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
from collections.abc import Container
from typing import TYPE_CHECKING, Any, Union

from narwhals._plan import common, expressions as ir, meta
from narwhals._plan.exceptions import column_not_found_error, duplicate_error
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
from narwhals._utils import check_column_names_are_unique
from narwhals.exceptions import MultiOutputExpressionError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from typing_extensions import TypeAlias

    from narwhals._plan.typing import Seq


OutputNames: TypeAlias = "Seq[str]"
"""Fully expanded, validated output column names, for `NamedIR`s."""


Ignored: TypeAlias = Container[str]
"""Ignored `Selector` column names.

Usually names resolved from `group_by(*keys)`.
"""

Combination: TypeAlias = Union[
    ir.SortBy,
    ir.BinaryExpr,
    ir.TernaryExpr,
    ir.Filter,
    ir.OrderedWindowExpr,
    ir.WindowExpr,
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
    frozen_schema = expander.schema
    rewritten = expander.expand_exprs(exprs)
    return resolve_names(rewritten, frozen_schema), frozen_schema


def expand_selector_irs_names(
    selectors: Sequence[SelectorIR], /, ignored: Ignored = (), *, schema: IntoFrozenSchema
) -> OutputNames:
    """Expand selector-only input into the column names that match.

    Similar to `prepare_projection`, but intended for allowing a subset of `Expr` and all `Selector`s
    to be used in more places like `DataFrame.{drop,sort,partition_by}`.

    Arguments:
        selectors: IRs that **only** contain subclasses of `SelectorIR`.
        ignored: Names of `group_by` columns.
        schema: Scope to expand selectors in.
    """
    expander = Expander(schema, ignored)
    names = expander.iter_expand_selector_names(selectors)
    return _ensure_valid_output_names(tuple(names), expander.schema)


# TODO @dangotbanned: Clean up
# Gets more done in a single pass
def resolve_names(exprs: Seq[ExprIR], schema: FrozenSchema) -> Seq[NamedIR]:
    names = deque[str]()
    named_irs = deque[NamedIR]()
    for e in exprs:
        # NOTE: Empty string is allowed as a name, but is falsy
        if (output_name := e.meta.output_name(raise_if_undetermined=False)) is not None:
            names.append(output_name)
            target = e
        elif meta.has_expr_ir(e, KeepName):
            replaced = replace_keep_name(e)
            output_name = replaced.meta.output_name()
            target = replaced
        else:
            msg = f"Unable to determine output name for expression, got: `{e!r}`"
            raise NotImplementedError(msg)
        named_irs.append(ir.named_ir(output_name, remove_alias(target)))
    if len(names) != len(set(names)):
        raise duplicate_error(exprs)
    root_names = meta.root_names_unique(exprs)
    if not (set(schema.names).issuperset(root_names)):
        raise column_not_found_error(root_names, schema)
    return tuple(named_irs)


def remove_alias(origin: ExprIR, /) -> ExprIR:
    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr if isinstance(child, (Alias, RenameAlias)) else child

    return origin.map_ir(fn)


def replace_keep_name(origin: ExprIR, /) -> ExprIR:
    root_name = meta.root_name_first(origin)

    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr.alias(root_name) if isinstance(child, KeepName) else child

    return origin.map_ir(fn)


def _ensure_valid_output_names(names: Seq[str], schema: FrozenSchema) -> OutputNames:
    check_column_names_are_unique(names)
    output_names = names
    if not (set(schema.names).issuperset(output_names)):
        raise column_not_found_error(output_names, schema)
    return output_names


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
            yield from self.expand(expr)

    def iter_expand_selector_names(
        self, selectors: Iterable[SelectorIR], /
    ) -> Iterator[str]:
        for s in selectors:
            yield from s.into_columns(self.schema, self.ignored)

    def expand_exprs(self, exprs: Sequence[ExprIR], /) -> Seq[ExprIR]:
        # Eagerly expand all of exprs
        return tuple(self.iter_expand_exprs(exprs))

    def expand(self, expr: ExprIR, /) -> Iterator[ExprIR]:
        # For a single expr, fully expand all parts of it
        if all(not e.needs_expansion() for e in expr.iter_left()):
            yield expr
        else:
            yield from self.expand_recursive(expr)

    def expand_recursive(self, origin: ExprIR, /) -> Iterator[ExprIR]:
        # Dispatch the kind of expansion, based on the type of expr
        # Every other method will call back here
        # Based on https://github.com/pola-rs/polars/blob/5b90db75911c70010d0c0a6941046e6144af88d4/crates/polars-plan/src/plans/conversion/dsl_to_ir/expr_expansion.rs#L253-L850
        if isinstance(origin, _EXPAND_NONE):
            yield origin
        elif isinstance(origin, ir.SelectorIR):
            names = origin.into_columns(self.schema, self.ignored)
            yield from (ir.Column(name=name) for name in names)
        elif isinstance(origin, _EXPAND_SINGLE):
            for expr in self.expand_recursive(origin.expr):
                yield origin.__replace__(expr=expr)
        elif isinstance(origin, _EXPAND_COMBINATION):
            yield from self._expand_combination(origin)
        elif isinstance(origin, ir.FunctionExpr):
            yield from self._expand_function_expr(origin)
        else:
            msg = f"Didn't expect to see {type(origin).__name__}"
            raise TypeError(msg)

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
            yield from self.expand_recursive(child)

    def _expand_only(self, child: ExprIR, /) -> ExprIR:
        # used by
        # - `_expand_combination` (ExprIR fields)
        # - `_expand_function_expr` (all others that have len(inputs)>=2, call on non-root)
        iterable = self.expand_recursive(child)
        first = next(iterable)
        if second := next(iterable, None):
            msg = f"Multi-output expressions are not supported in this context, got: `{second!r}`"
            raise MultiOutputExpressionError(msg)
        return first

    # TODO @dangotbanned: Placeholder, don't want to use the current function, too complex
    def _expand_combination(self, origin: Combination, /) -> Iterator[Combination]:
        changes: dict[str, Any] = {}
        if isinstance(origin, (ir.WindowExpr, ir.Filter, ir.SortBy)):
            if isinstance(origin, ir.WindowExpr):
                if partition_by := origin.partition_by:
                    changes["partition_by"] = tuple(self._expand_inner(partition_by))
                if isinstance(origin, ir.OrderedWindowExpr):
                    changes["order_by"] = tuple(self._expand_inner(origin.order_by))
            elif isinstance(origin, ir.SortBy):
                changes["by"] = tuple(self._expand_inner(origin.by))
            else:
                changes["by"] = self._expand_only(origin.by)
            replaced = common.replace(origin, **changes)
            for root in self.expand_recursive(replaced.expr):
                yield common.replace(replaced, expr=root)
        elif isinstance(origin, ir.BinaryExpr):
            binary = origin.__replace__(right=self._expand_only(origin.right))
            for root in self.expand_recursive(binary.left):
                yield binary.__replace__(left=root)
        elif isinstance(origin, ir.TernaryExpr):
            changes["truthy"] = self._expand_only(origin.truthy)
            changes["predicate"] = self._expand_only(origin.predicate)
            changes["falsy"] = self._expand_only(origin.falsy)
            yield origin.__replace__(**changes)
        else:
            assert_never(origin)

    def _expand_function_expr(
        self, origin: ir.FunctionExpr, /
    ) -> Iterator[ir.FunctionExpr]:
        if origin.options.is_input_wildcard_expansion():
            reduced = tuple(self._expand_inner(origin.input))
            yield origin.__replace__(input=reduced)
        else:
            if non_root := origin.input[1:]:
                children = tuple(self._expand_only(child) for child in non_root)
            else:
                children = ()
            for root in self.expand_recursive(origin.input[0]):
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
    ir.OrderedWindowExpr,
    ir.WindowExpr,
)
"""more than one (direct) child and those can be nested."""
