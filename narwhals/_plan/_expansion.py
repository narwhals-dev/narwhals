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

# ruff: noqa: A002
from __future__ import annotations

from collections import deque
from collections.abc import Container
from typing import TYPE_CHECKING, Any, Protocol, Union

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
from narwhals._plan.typing import Seq
from narwhals._typing_compat import TypeVar, assert_never
from narwhals._utils import check_column_names_are_unique
from narwhals.exceptions import MultiOutputExpressionError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from typing_extensions import Self, TypeAlias


OutputNames: TypeAlias = "Seq[str]"
"""Fully expanded, validated output column names, for `NamedIR`s."""


Ignored: TypeAlias = Container[str]
"""Ignored `Selector` column names.

Usually names resolved from `group_by(*keys)`.
"""


def prepare_projection_s(
    exprs: Sequence[ExprIR], /, ignored: Ignored = (), *, schema: IntoFrozenSchema
) -> tuple[Seq[NamedIR], FrozenSchema]:
    frozen_schema = freeze_schema(schema)
    rewritten = rewrite_projections_s(tuple(exprs), ignored, schema=frozen_schema)
    named_irs = finish_exprs(rewritten, frozen_schema)
    return named_irs, frozen_schema


def expand_selector_irs_names(
    selectors: Sequence[SelectorIR], /, ignored: Ignored = (), *, schema: IntoFrozenSchema
) -> OutputNames:
    """Expand selector-only input into the column names that match.

    Similar to `prepare_projection`, but intended for allowing a subset of `Expr` and all `Selector`s
    to be used in more places like `DataFrame.{drop,sort,partition_by}`.

    Arguments:
        selectors: IRs that **only** contain subclasses of `SelectorIR`.
        ignored: Names of `group_by` columns.
        schema: Scope to expand multi-column selectors in.
    """
    frozen_schema = freeze_schema(schema)
    names = tuple(_iter_expand_selector_names(selectors, ignored, schema=frozen_schema))
    return _ensure_valid_output_names(names, frozen_schema)


def into_named_irs_s(exprs: Seq[ExprIR], names: OutputNames) -> Seq[NamedIR]:
    if len(exprs) != len(names):
        msg = f"zip length mismatch: {len(exprs)} != {len(names)}"
        raise ValueError(msg)
    return tuple(ir.named_ir(name, remove_alias_s(e)) for e, name in zip(exprs, names))


# TODO @dangotbanned: Clean up
# Gets more done in a single pass
def finish_exprs(exprs: Seq[ExprIR], schema: FrozenSchema) -> Seq[NamedIR]:
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
        named_irs.append(ir.named_ir(output_name, remove_alias_s(target)))
    if len(names) != len(set(names)):
        raise duplicate_error(exprs)
    root_names = meta.root_names_unique(exprs)
    if not (set(schema.names).issuperset(root_names)):
        raise column_not_found_error(root_names, schema)
    return tuple(named_irs)


# TODO @dangotbanned: Remove or factor back in
def ensure_valid_exprs(exprs: Seq[ExprIR], schema: FrozenSchema) -> OutputNames:
    """Raise an appropriate error if we can't materialize."""
    output_names = _ensure_output_names_unique(exprs)
    root_names = meta.root_names_unique(exprs)
    if not (set(schema.names).issuperset(root_names)):
        raise column_not_found_error(root_names, schema)
    return output_names


def _ensure_valid_output_names(names: Seq[str], schema: FrozenSchema) -> OutputNames:
    """Selector-only variant of `ensure_valid_exprs`."""
    check_column_names_are_unique(names)
    output_names = names
    if not (set(schema.names).issuperset(output_names)):
        raise column_not_found_error(output_names, schema)
    return output_names


def _ensure_output_names_unique(exprs: Seq[ExprIR]) -> OutputNames:
    names = tuple(e.meta.output_name() for e in exprs)
    if len(names) != len(set(names)):
        raise duplicate_error(exprs)
    return names


def _iter_expand_selector_names(
    selectors: Iterable[SelectorIR], /, ignored: Ignored = (), *, schema: FrozenSchema
) -> Iterator[str]:
    for s in selectors:
        yield from s.into_columns(schema, ignored)


def rewrite_projections_s(
    input: Seq[ExprIR], /, ignored: Ignored, *, schema: FrozenSchema
) -> Seq[ExprIR]:
    result: deque[ExprIR] = deque()
    for expr in input:
        result.extend(expand_expression_s(expr, ignored, schema))
    return tuple(result)


def needs_expansion_s(expr: ExprIR) -> bool:
    return any(isinstance(e, ir.SelectorIR) for e in expr.iter_left())


def expand_expression_s(
    expr: ExprIR, ignored: Ignored, schema: FrozenSchema, /
) -> Iterator[ExprIR]:
    if all(not needs_expansion_s(e) for e in expr.iter_left()):
        yield expr
    else:
        yield from expand_expression_rec_s(expr, ignored, schema)


class CanExpandSingle(Protocol):
    @property
    def expr(self) -> Child: ...
    def __replace__(self, *, expr: Child) -> Self: ...


CanExpandSingleT = TypeVar("CanExpandSingleT", bound=CanExpandSingle)

Child: TypeAlias = ExprIR
Children: TypeAlias = Seq[ExprIR]
_Combination: TypeAlias = Union[
    ir.SortBy,
    ir.BinaryExpr,
    ir.TernaryExpr,
    ir.Filter,
    ir.OrderedWindowExpr,
    ir.WindowExpr,
]

_EXPAND_NONE = (ir.Column, ir.Literal, ir.Len)
"""we're at the root."""
_EXPAND_SINGLE = (ir.Alias, ir.Cast, ir.AggExpr, ir.Sort, ir.KeepName, ir.RenameAlias)
"""one (direct) child, always stored in `self.expr`."""
_EXPAND_COMBINATION = (
    ir.SortBy,
    ir.BinaryExpr,
    ir.TernaryExpr,
    ir.Filter,
    ir.OrderedWindowExpr,
    ir.WindowExpr,
)
"""more than one (direct) child and those can be nested."""


def _expand_inner(
    children: Children, ignored: Ignored, schema: FrozenSchema, /
) -> Iterator[ExprIR]:
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
    for child in children:
        yield from expand_expression_rec_s(child, ignored, schema)


def _expand_function_expr(
    origin: ir.FunctionExpr, ignored: Ignored, schema: FrozenSchema, /
) -> Iterator[ir.FunctionExpr]:
    # 1x output
    if origin.options.is_input_wildcard_expansion():
        reduced = tuple(_expand_inner(origin.input, ignored, schema))
        yield origin.__replace__(input=reduced)

    # (potentially) many outputs
    else:
        if non_root := origin.input[1:]:
            children = tuple(expand_only_s(child, ignored, schema) for child in non_root)
        else:
            children = ()
        for root in expand_expression_rec_s(origin.input[0], ignored, schema):
            yield origin.__replace__(input=(root, *children))


def expand_single_s(
    origin: CanExpandSingleT, ignored: Ignored, schema: FrozenSchema
) -> Iterator[CanExpandSingleT]:
    """Expand the root of `origin`, yielding each child as a replacement.

    Say we had:

        origin = Cast(expr=ByName(names=("one", "two"), require_all=True), dtype=String)

    This would expand to:

        cast_one = Cast(expr=Column(name="one"), dtype=String)
        cast_two = Cast(expr=Column(name="two"), dtype=String)
    """
    replace = origin.__replace__
    for e in expand_expression_rec_s(origin.expr, ignored, schema):
        yield replace(expr=e)


def expand_only_s(child: Child, ignored: Ignored, schema: FrozenSchema, /) -> ExprIR:
    iterable = expand_expression_rec_s(child, ignored, schema)
    first = next(iterable)
    if second := next(iterable, None):
        msg = f"Multi-output expressions are not supported in this context, got: `{second!r}`"
        raise MultiOutputExpressionError(msg)
    return first


# TODO @dangotbanned: Clean up, possibly move to be a method
def _expand_nested_nodes_s(
    origin: _Combination, ignored: Ignored, schema: FrozenSchema, /
) -> Iterator[_Combination]:
    expands = _expand_inner
    changes: dict[str, Any] = {}
    if isinstance(origin, (ir.WindowExpr, ir.Filter, ir.SortBy)):
        if isinstance(origin, ir.WindowExpr):
            if partition_by := origin.partition_by:
                changes["partition_by"] = tuple(expands(partition_by, ignored, schema))
            if isinstance(origin, ir.OrderedWindowExpr):
                changes["order_by"] = tuple(expands(origin.order_by, ignored, schema))
        elif isinstance(origin, ir.SortBy):
            changes["by"] = tuple(expands(origin.by, ignored, schema))
        else:
            changes["by"] = expand_only_s(origin.by, ignored, schema)
        replaced = common.replace(origin, **changes)
        for root in expand_expression_rec_s(replaced.expr, ignored, schema):
            yield common.replace(replaced, expr=root)
    elif isinstance(origin, ir.BinaryExpr):
        replaced = origin.__replace__(right=expand_only_s(origin.right, ignored, schema))  # type: ignore[assignment]
        for root in expand_expression_rec_s(replaced.left, ignored, schema):  # type: ignore[union-attr]
            yield replaced.__replace__(left=root)  # type: ignore[call-arg]
    elif isinstance(origin, ir.TernaryExpr):
        changes["truthy"] = expand_only_s(origin.truthy, ignored, schema)
        changes["predicate"] = expand_only_s(origin.predicate, ignored, schema)
        changes["falsy"] = expand_only_s(origin.falsy, ignored, schema)
        yield origin.__replace__(**changes)
    else:
        assert_never(origin)


def expand_expression_rec_s(
    expr: ExprIR, ignored: Ignored, schema: FrozenSchema, /
) -> Iterator[ExprIR]:
    # https://github.com/pola-rs/polars/blob/5b90db75911c70010d0c0a6941046e6144af88d4/crates/polars-plan/src/plans/conversion/dsl_to_ir/expr_expansion.rs#L253-L850

    # no expand
    if isinstance(expr, _EXPAND_NONE):
        yield expr

    # selectors, handled internally
    elif isinstance(expr, ir.SelectorIR):
        yield from (ir.Column(name=name) for name in expr.into_columns(schema, ignored))

    # `(ir.KeepName, ir.RenameAlias)` Renaming moved to *roughly* whenever `to_expr_ir` is called, leading to the resolving here
    #   https://github.com/pola-rs/polars/blob/5b90db75911c70010d0c0a6941046e6144af88d4/crates/polars-plan/src/plans/conversion/dsl_to_ir/expr_to_ir.rs#L466-L469
    #   https://github.com/pola-rs/polars/blob/5b90db75911c70010d0c0a6941046e6144af88d4/crates/polars-plan/src/plans/conversion/dsl_to_ir/expr_to_ir.rs#L481-L485
    elif isinstance(expr, _EXPAND_SINGLE):
        yield from expand_single_s(expr, ignored, schema)

    elif isinstance(expr, _EXPAND_COMBINATION):
        yield from _expand_nested_nodes_s(expr, ignored, schema)

    elif isinstance(expr, ir.FunctionExpr):
        yield from _expand_function_expr(expr, ignored, schema)

    else:
        msg = f"Didn't expect to see {type(expr).__name__}"
        raise TypeError(msg)


def remove_alias_s(origin: ExprIR, /) -> ExprIR:
    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr if isinstance(child, (Alias, RenameAlias)) else child

    return origin.map_ir(fn)


def replace_keep_name(origin: ExprIR, /) -> ExprIR:
    root_name = meta.root_name_first(origin)

    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr.alias(root_name) if isinstance(child, KeepName) else child

    return origin.map_ir(fn)
