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
from functools import lru_cache
from itertools import chain
from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan import common, expressions as ir, meta
from narwhals._plan._guards import is_horizontal_reduction, is_window_expr
from narwhals._plan._immutable import Immutable
from narwhals._plan.exceptions import (
    column_index_error,
    column_not_found_error,
    duplicate_error,
)
from narwhals._plan.expressions import (
    Alias,
    All,
    Columns,
    Exclude,
    ExprIR,
    IndexColumns,
    KeepName,
    NamedIR,
    Nth,
    RenameAlias,
    SelectorIR,
    _ColumnSelection,
    col,
    cols,
)
from narwhals._plan.schema import (
    FrozenColumns,
    FrozenSchema,
    IntoFrozenSchema,
    freeze_schema,
)
from narwhals._plan.typing import Seq
from narwhals._typing_compat import TypeVar, deprecated
from narwhals._utils import check_column_names_are_unique
from narwhals.dtypes import DType
from narwhals.exceptions import ComputeError, InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from typing_extensions import Self, TypeAlias

    from narwhals.dtypes import DType


Excluded: TypeAlias = "frozenset[str]"
"""Internally use a `set`, then freeze before returning."""

GroupByKeys: TypeAlias = "Seq[str]"
"""Represents `group_by` keys.

They need to be excluded from expansion.
"""

OutputNames: TypeAlias = "Seq[str]"
"""Fully expanded, validated output column names, for `NamedIR`s."""


Ignored: TypeAlias = Container[str]
"""Ignored `Selector` column names.

Usually names resolved from `group_by(*keys)`.
"""


class ExpansionFlags(Immutable):
    """`polars` uses a struct, but we may want to use `enum.Flag`."""

    __slots__ = (
        "has_exclude",
        "has_nth",
        "has_selector",
        "has_wildcard",
        "multiple_columns",
    )
    multiple_columns: bool
    has_nth: bool
    has_wildcard: bool
    has_selector: bool
    has_exclude: bool

    @property
    def expands(self) -> bool:
        """If we add struct stuff, that would slot in here as well."""
        return self.multiple_columns

    @staticmethod
    def from_ir(ir: ExprIR, /) -> ExpansionFlags:
        """Subset of [`find_flags`].

        [`find_flags`]: https://github.com/pola-rs/polars/blob/df4d21c30c2b383b651e194f8263244f2afaeda3/crates/polars-plan/src/plans/conversion/expr_expansion.rs#L607-L660
        """
        multiple_columns: bool = False
        has_nth: bool = False
        has_wildcard: bool = False
        has_selector: bool = False
        has_exclude: bool = False
        for e in ir.iter_left():
            if isinstance(e, (_ColumnSelection, SelectorIR)):
                if isinstance(e, (Columns, IndexColumns)):
                    multiple_columns = True
                elif isinstance(e, Nth):
                    has_nth = True
                elif isinstance(e, All):
                    has_wildcard = True
                elif isinstance(e, SelectorIR):
                    has_selector = True
                elif isinstance(e, Exclude):
                    has_exclude = True
        return ExpansionFlags(
            multiple_columns=multiple_columns,
            has_nth=has_nth,
            has_wildcard=has_wildcard,
            has_selector=has_selector,
            has_exclude=has_exclude,
        )

    def with_multiple_columns(self) -> ExpansionFlags:
        return common.replace(self, multiple_columns=True)


def prepare_projection(
    exprs: Sequence[ExprIR], /, keys: GroupByKeys = (), *, schema: IntoFrozenSchema
) -> tuple[Seq[NamedIR], FrozenSchema]:
    """Expand IRs into named column projections.

    **Primary entry-point**, for `select`, `with_columns`,
    and any other context that requires resolving expression names.

    Arguments:
        exprs: IRs that *may* contain things like `Columns`, `SelectorIR`, `Exclude`, etc.
        keys: Names of `group_by` columns.
        schema: Scope to expand multi-column selectors in.
    """
    frozen_schema = freeze_schema(schema)
    rewritten = rewrite_projections(tuple(exprs), keys=keys, schema=frozen_schema)
    output_names = ensure_valid_exprs(rewritten, frozen_schema)
    named_irs = into_named_irs(rewritten, output_names)
    return named_irs, frozen_schema


def prepare_projection_s(
    exprs: Sequence[ExprIR], /, ignored: Ignored = (), *, schema: IntoFrozenSchema
) -> tuple[Seq[NamedIR], FrozenSchema]:
    frozen_schema = freeze_schema(schema)
    rewritten = rewrite_projections_s(tuple(exprs), ignored, schema=frozen_schema)

    # TODO @dangotbanned: Plan when to rewrite aliases
    # This will raise because renaming happens at a different level than before
    output_names = ensure_valid_exprs(rewritten, frozen_schema)
    named_irs = into_named_irs(rewritten, output_names)
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


def into_named_irs(exprs: Seq[ExprIR], names: OutputNames) -> Seq[NamedIR]:
    if len(exprs) != len(names):
        msg = f"zip length mismatch: {len(exprs)} != {len(names)}"
        raise ValueError(msg)
    return tuple(ir.named_ir(name, remove_alias(e)) for e, name in zip(exprs, names))


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


# NOTE: Recursive for all `input` expressions which themselves contain `Seq[ExprIR]`
def rewrite_projections(
    input: Seq[ExprIR], /, keys: GroupByKeys = (), *, schema: FrozenSchema
) -> Seq[ExprIR]:
    result: deque[ExprIR] = deque()
    for expr in input:
        expanded = _expand_nested_nodes(expr, schema=schema)
        flags = ExpansionFlags.from_ir(expanded)
        if flags.has_selector:
            expanded = replace_selector(expanded, schema=schema)
            flags = flags.with_multiple_columns()
        result.extend(iter_replace(expanded, keys, col_names=schema.names, flags=flags))
    return tuple(result)


def rewrite_projections_s(
    input: Seq[ExprIR], /, ignored: Ignored, *, schema: FrozenSchema
) -> Seq[ExprIR]:
    result: deque[ExprIR] = deque()
    for expr in input:
        result.extend(expand_expression_s(expr, ignored, schema))
    return tuple(result)


# TODO @dangotbanned: Remove the `ir._ColumnSelection` branch when they're gone
# Pretty much a temp reminder for me to **not** use `col("a", "b")` on this path *yet*
def needs_expansion_s(expr: ExprIR) -> bool:
    if any(isinstance(e, ir._ColumnSelection) for e in expr.iter_left()):
        msg = f"Cannot use non-`Selector` column selections here, got: `{expr!r}`.\n\nHint: instead try `{expr!r}.meta.as_selector()`."
        raise TypeError(msg)
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


class CanExpandFunction(Protocol):
    @property
    def input(self) -> Children: ...
    def __replace__(self, *, input: Children) -> Self: ...


CanExpandSingleT = TypeVar("CanExpandSingleT", bound=CanExpandSingle)
CanExpandFunctionT = TypeVar("CanExpandFunctionT", bound=CanExpandFunction)
Origin = TypeVar("Origin", bound=ExprIR)
Child: TypeAlias = ExprIR
Children: TypeAlias = Seq[ExprIR]


def _replace_single(origin: CanExpandSingleT, /) -> Callable[[Child], CanExpandSingleT]:
    """Defines a (single, positional-only parameter) constructor for expanding children.

    Say we had:

        origin = Cast(expr=ByName(names=("one", "two"), require_all=True), dtype=String)

    This would expand to:

        cast_one = Cast(expr=Column(name="one"), dtype=String)
        cast_two = Cast(expr=Column(name="two"), dtype=String)

    The function returned here (`fn`) will allow us to pass in these replacements
    to generate the expansions:

        fn = _replace_single(origin)
        cast_one = fn(Column(name="one"))
        cast_two = fn(Column(name="two"))
    """
    replace = origin.__replace__

    def fn(expr: Child, /) -> CanExpandSingleT:
        return replace(expr=expr)

    return fn


def _replace_function_input(
    origin: CanExpandFunctionT, /
) -> Callable[[Children], CanExpandFunctionT]:
    """Like `_replace_single`, but for all non-horizontal `FunctionExpr`s."""
    replace = origin.__replace__

    def fn(children: Children, /) -> CanExpandFunctionT:
        return replace(input=children)

    return fn


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
        yield from expand_single_s(expr.expr, ignored, schema, _replace_single(expr))

    elif isinstance(expr, _EXPAND_COMBINATION):
        # The bit here is very similar to single:
        # - instead of `expand_single(expr.expr)` it is `expand_expr...([expr.child1, expr.child2, ...])
        # - instead of `_replace_single`, it defines a function that maps `*exprs` -> to the positions per-variant
        exprs, expand = expr._into_expand()
        yield from expand_expression_by_combination_s(exprs, ignored, schema, expand)

    elif isinstance(expr, ir.FunctionExpr):
        if expr.options.is_input_wildcard_expansion():
            it = chain.from_iterable(
                expand_expression_rec_s(e, ignored, schema) for e in expr.input
            )
            yield common.replace(expr, input=tuple(it))
            return
        expand = _replace_function_input(expr)
        yield from expand_expression_by_combination_s(expr.input, ignored, schema, expand)

    else:
        msg = f"Didn't expect to see {type(expr).__name__}"
        raise TypeError(msg)


def expand_single_s(
    child: ExprIR,
    ignored: Ignored,
    schema: FrozenSchema,
    replace_in_origin: Callable[[ExprIR], Origin],
) -> Iterator[Origin]:
    # Before: `expand_expression_rec`
    # Current: `expand_single`
    # Next: `try_expand_single`
    # Recurse: `expand_expression_rec`

    # NOTE: Like maybe 60% sure this is correct
    it_expanding_child = expand_expression_rec_s(child, ignored, schema)
    for e in it_expanding_child:
        yield replace_in_origin(e)


# TODO @dangotbanned: Translating the last branch
# https://github.com/pola-rs/polars/blob/5b90db75911c70010d0c0a6941046e6144af88d4/crates/polars-plan/src/plans/conversion/dsl_to_ir/expr_expansion.rs#L124-L193
def expand_expression_by_combination_s(
    children: Children,
    ignored: Ignored,
    schema: FrozenSchema,
    replace_in_origin: Callable[[Children], Origin],
) -> Iterator[Origin]:
    expanded = deque[ExprIR]()
    # Expand expressions until we find one that expands to more than 1 expression.
    expansion_size: int = 0
    continue_at: int = 0
    for i, child in enumerate(children):
        it_expanding_child = expand_expression_rec_s(child, ignored, schema)
        grandchildren = tuple(it_expanding_child)
        n_grandchildren = len(grandchildren)
        if n_grandchildren != 1:
            expansion_size = n_grandchildren
            expanded.extend(grandchildren)
            continue_at = i + 1
            break
        else:
            expanded.append(grandchildren[0])

    # Check if all expressions expanded to 1 expression.
    # This case already works correctly
    if expansion_size == 0:
        yield replace_in_origin(tuple(expanded))
        return

    # Now do the remaining expression, and check if they match the size of the original expansion
    # (or 1)
    for child in children[continue_at:]:
        it_expanding_child = expand_expression_rec_s(child, ignored, schema)
        grandchildren = tuple(it_expanding_child)
        n_grandchildren = len(grandchildren)
        if n_grandchildren not in {1, expansion_size}:
            # NOTE: Unclear on the intended goal
            # Raised https://github.com/pola-rs/polars/issues/25022
            msg = f"cannot combine selectors that produce a different number of columns ({n_grandchildren} != {expansion_size})"
            raise InvalidOperationError(msg)
        expanded.extend(grandchildren)

    # Create actual output expressions.
    # TODO @dangotbanned: Fix this, it is silently dropping anything that expanded beyond the length
    # of the `origin` constructor
    msg = "TODO: Mixed expansion `expand_expression_by_combination`."
    raise NotImplementedError(msg)
    # The size/len/indices that polars is tracking are into `out`, to map back onto each expansion
    yield replace_in_origin(tuple(expanded))


def _expand_nested_nodes(origin: ExprIR, /, *, schema: FrozenSchema) -> ExprIR:
    """Adapted from [`expand_function_inputs`].

    Added additional cases for nodes that *also* need to be expanded in the same way.

    [`expand_function_inputs`]: https://github.com/pola-rs/polars/blob/df4d21c30c2b383b651e194f8263244f2afaeda3/crates/polars-plan/src/plans/conversion/expr_expansion.rs#L557-L581
    """
    rewrite = rewrite_projections

    def fn(child: ExprIR, /) -> ExprIR:
        if not isinstance(child, (ir.FunctionExpr, ir.WindowExpr, ir.SortBy)):
            return child
        expanded: dict[str, Any] = {}
        if is_horizontal_reduction(child):
            expanded["input"] = rewrite(child.input, schema=schema)
        elif is_window_expr(child):
            if partition_by := child.partition_by:
                expanded["partition_by"] = rewrite(partition_by, schema=schema)
            if isinstance(child, ir.OrderedWindowExpr):
                expanded["order_by"] = rewrite(child.order_by, schema=schema)
        elif isinstance(child, ir.SortBy):
            expanded["by"] = rewrite(child.by, schema=schema)
        if not expanded:
            return child
        return common.replace(child, **expanded)

    return origin.map_ir(fn)


def replace_nth(origin: ExprIR, /, col_names: FrozenColumns) -> ExprIR:
    n_fields = len(col_names)

    def fn(child: ExprIR, /) -> ExprIR:
        if isinstance(child, Nth):
            if not is_index_in_range(child.index, n_fields):
                raise column_index_error(child.index, col_names)
            return col(col_names[child.index])
        return child

    return origin.map_ir(fn)


def is_index_in_range(index: int, n_fields: int) -> bool:
    idx = index + n_fields if index < 0 else index
    return not (idx < 0 or idx >= n_fields)


def remove_alias(origin: ExprIR, /) -> ExprIR:
    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr if isinstance(child, Alias) else child

    return origin.map_ir(fn)


def replace_with_column(
    origin: ExprIR, tp: type[_ColumnSelection], /, name: str
) -> ExprIR:
    """Expand a single column within a multi-selection using `name`."""

    def fn(child: ExprIR, /) -> ExprIR:
        if isinstance(child, tp):
            return col(name)
        return child.expr if isinstance(child, Exclude) else child

    return origin.map_ir(fn)


# NOTE: The other calls have `lru_cache` swallowing the warning
# (`expand_selector`, `selector_matches_column`)
@deprecated("Use `matches` or `into_columns` instead")
def replace_selector(ir: ExprIR, /, *, schema: FrozenSchema) -> ExprIR:
    def fn(child: ExprIR, /) -> ExprIR:
        return expand_selector(child, schema) if isinstance(child, SelectorIR) else child

    return ir.map_ir(fn)


@lru_cache(maxsize=100)
def selector_matches_column(selector: SelectorIR, name: str, dtype: DType, /) -> bool:
    """Cached version of `SelectorIR.matches.column`.

    Allows results of evaluations can be shared across:
    - Instances of `SelectorIR`
    - Multiple schemas
    """
    return selector.matches_column(name, dtype)


@lru_cache(maxsize=100)
def expand_selector(selector: SelectorIR, schema: FrozenSchema) -> Columns:
    """Expand `selector` into `Columns`, within the context of `schema`."""
    matches = selector_matches_column
    return cols(*(k for k, v in schema.items() if matches(selector, k, v)))


def iter_replace(
    origin: ExprIR,
    /,
    keys: GroupByKeys,
    *,
    col_names: FrozenColumns,
    flags: ExpansionFlags,
) -> Iterator[ExprIR]:
    if flags.has_nth:
        origin = replace_nth(origin, col_names)
    if flags.expands:
        it = (e for e in origin.iter_left() if isinstance(e, (Columns, IndexColumns)))
        if e := next(it, None):
            if isinstance(e, Columns):
                if not _all_columns_match(origin, e):
                    msg = "expanding more than one `col` is not allowed"
                    raise ComputeError(msg)
                names: Iterable[str] = e.names
            else:
                names = _iter_index_names(e, col_names)
            exclude = prepare_excluded(origin, keys, flags)
            yield from expand_column_selection(origin, type(e), names, exclude)
    elif flags.has_wildcard:
        exclude = prepare_excluded(origin, keys, flags)
        yield from expand_column_selection(origin, All, col_names, exclude)
    else:
        yield rewrite_special_aliases(origin)


def prepare_excluded(
    origin: ExprIR, keys: GroupByKeys, flags: ExpansionFlags, /
) -> Excluded:
    """Huge simplification of https://github.com/pola-rs/polars/blob/0fa7141ce718c6f0a4d6ae46865c867b177a59ed/crates/polars-plan/src/plans/conversion/expr_expansion.rs#L484-L555."""
    gb_keys = frozenset(keys)
    if not flags.has_exclude:
        return gb_keys
    return gb_keys.union(*(e.names for e in origin.iter_left() if isinstance(e, Exclude)))


def _all_columns_match(origin: ExprIR, /, columns: Columns) -> bool:
    it = (e == columns if isinstance(e, Columns) else True for e in origin.iter_left())
    return all(it)


def _iter_index_names(indices: IndexColumns, names: FrozenColumns, /) -> Iterator[str]:
    n_fields = len(names)
    for index in indices.indices:
        if not is_index_in_range(index, n_fields):
            raise column_index_error(index, names)
        yield names[index]


def expand_column_selection(
    origin: ExprIR, tp: type[_ColumnSelection], /, names: Iterable[str], exclude: Excluded
) -> Iterator[ExprIR]:
    for name in names:
        if name not in exclude:
            yield rewrite_special_aliases(replace_with_column(origin, tp, name))


def rewrite_special_aliases(origin: ExprIR, /) -> ExprIR:
    """Expand `KeepName` and `RenameAlias` into `Alias`.

    Warning:
        Only valid **after**
        - Expanding all selections into `Column`
        - Dealing with `FunctionExpr.input`
    """
    if meta.has_expr_ir(origin, KeepName, RenameAlias):
        if isinstance(origin, KeepName):
            parent = origin.expr
            return parent.alias(next(iter(parent.meta.root_names())))
        if isinstance(origin, RenameAlias):
            parent = origin.expr
            leaf_name_or_err = meta.get_single_leaf_name(parent)
            if not isinstance(leaf_name_or_err, str):
                raise leaf_name_or_err
            return parent.alias(origin.function(leaf_name_or_err))
        msg = "`keep`, `suffix`, `prefix` should be last expression"
        raise InvalidOperationError(msg)
    return origin
