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
from functools import lru_cache
from itertools import chain
from typing import TYPE_CHECKING

from narwhals._plan import common
from narwhals._plan.common import (
    ExprIR,
    Immutable,
    NamedIR,
    SelectorIR,
    is_horizontal_reduction,
)
from narwhals._plan.exceptions import (
    column_index_error,
    column_not_found_error,
    duplicate_error,
)
from narwhals._plan.expr import (
    Alias,
    All,
    Columns,
    Exclude,
    IndexColumns,
    KeepName,
    Nth,
    RenameAlias,
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
from narwhals.dtypes import DType
from narwhals.exceptions import ComputeError, InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from typing_extensions import TypeAlias

    from narwhals._plan.dummy import Expr
    from narwhals._plan.typing import Seq
    from narwhals.dtypes import DType


Excluded: TypeAlias = "frozenset[str]"
"""Internally use a `set`, then freeze before returning."""

GroupByKeys: TypeAlias = "Seq[ExprIR]"
"""Represents group_by keys.

- Originates from `polars_plan::plans::conversion::dsl_to_ir::resolve_group_by`
- Not fully utilized in `narwhals` version yet
"""

OutputNames: TypeAlias = "Seq[str]"
"""Fully expanded, validated output column names, for `NamedIR`s."""


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

    @classmethod
    def from_expr(cls, expr: Expr, /) -> ExpansionFlags:
        return cls.from_ir(expr._ir)

    def with_multiple_columns(self) -> ExpansionFlags:
        return common.replace(self, multiple_columns=True)


def prepare_projection(
    exprs: Sequence[ExprIR], schema: IntoFrozenSchema
) -> tuple[Seq[ExprIR], FrozenSchema, OutputNames]:
    """Expand IRs into named column selections.

    **Primary entry-point**, will be used by `select`, `with_columns`,
    and any other context that requires resolving expression names.

    Arguments:
        exprs: IRs that *may* contain things like `Columns`, `SelectorIR`, `Exclude`, etc.
        schema: Scope to expand multi-column selectors in.

    Returns:
        `exprs`, rewritten using `Column(name)` only.
    """
    frozen_schema = freeze_schema(schema)
    rewritten = rewrite_projections(tuple(exprs), keys=(), schema=frozen_schema)
    output_names = ensure_valid_exprs(rewritten, frozen_schema)
    return rewritten, frozen_schema, output_names


def into_named_irs(exprs: Seq[ExprIR], names: OutputNames) -> Seq[NamedIR]:
    if len(exprs) != len(names):
        msg = f"zip length mismatch: {len(exprs)} != {len(names)}"
        raise ValueError(msg)
    return tuple(
        NamedIR(expr=remove_alias(ir), name=name) for ir, name in zip(exprs, names)
    )


def ensure_valid_exprs(exprs: Seq[ExprIR], schema: FrozenSchema) -> OutputNames:
    """Raise an appropriate error if we can't materialize."""
    output_names = _ensure_output_names_unique(exprs)
    root_names = _root_names_unique(exprs)
    if not (set(schema.names).issuperset(root_names)):
        raise column_not_found_error(root_names, schema)
    return output_names


def _ensure_output_names_unique(exprs: Seq[ExprIR]) -> OutputNames:
    names = tuple(e.meta.output_name() for e in exprs)
    if len(names) != len(set(names)):
        raise duplicate_error(exprs)
    return names


def _root_names_unique(exprs: Seq[ExprIR]) -> set[str]:
    from narwhals._plan.meta import _expr_to_leaf_column_names_iter

    it = chain.from_iterable(_expr_to_leaf_column_names_iter(expr) for expr in exprs)
    return set(it)


def expand_function_inputs(origin: ExprIR, /, *, schema: FrozenSchema) -> ExprIR:
    def fn(child: ExprIR, /) -> ExprIR:
        if is_horizontal_reduction(child):
            return child.with_input(
                rewrite_projections(child.input, keys=(), schema=schema)
            )
        return child

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
        if isinstance(child, Alias):
            return child.expr
        return child

    return origin.map_ir(fn)


def remove_exclude(origin: ExprIR, /) -> ExprIR:
    def fn(child: ExprIR, /) -> ExprIR:
        if isinstance(child, Exclude):
            return child.expr
        return child

    return origin.map_ir(fn)


def replace_with_column(
    origin: ExprIR, tp: type[_ColumnSelection], /, name: str
) -> ExprIR:
    """Expand a single column within a multi-selection using `name`."""

    def fn(child: ExprIR, /) -> ExprIR:
        if isinstance(child, tp):
            return col(name)
        if isinstance(child, Exclude):
            return child.expr
        return child

    return origin.map_ir(fn)


def replace_selector(ir: ExprIR, /, *, schema: FrozenSchema) -> ExprIR:
    """Fully diverging from `polars`, we'll see how that goes."""

    def fn(child: ExprIR, /) -> ExprIR:
        if isinstance(child, SelectorIR):
            return expand_selector(child, schema=schema)
        return child

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
def expand_selector(selector: SelectorIR, *, schema: FrozenSchema) -> Columns:
    """Expand `selector` into `Columns`, within the context of `schema`."""
    matches = selector_matches_column
    return cols(*(k for k, v in schema.items() if matches(selector, k, v)))


def rewrite_projections(
    input: Seq[ExprIR],  # `FunctionExpr.input`
    /,
    keys: GroupByKeys,
    *,
    schema: FrozenSchema,
) -> Seq[ExprIR]:
    result: deque[ExprIR] = deque()
    for expr in input:
        expanded = expand_function_inputs(expr, schema=schema)
        flags = ExpansionFlags.from_ir(expanded)
        if flags.has_selector:
            expanded = replace_selector(expanded, schema=schema)
            flags = flags.with_multiple_columns()
        result.extend(
            replace_and_add_to_results(
                expanded, keys=keys, col_names=schema.names, flags=flags
            )
        )
    return tuple(result)


def replace_and_add_to_results(
    origin: ExprIR,
    /,
    keys: GroupByKeys,
    *,
    col_names: FrozenColumns,
    flags: ExpansionFlags,
) -> Seq[ExprIR]:
    result: deque[ExprIR] = deque()
    if flags.has_nth:
        origin = replace_nth(origin, col_names)
    if flags.expands:
        it = (e for e in origin.iter_left() if isinstance(e, (Columns, IndexColumns)))
        if e := next(it, None):
            if isinstance(e, Columns):
                exclude = prepare_excluded(
                    origin, keys=keys, has_exclude=flags.has_exclude
                )
                result.extend(expand_columns(origin, e, exclude=exclude))
            else:
                exclude = prepare_excluded(
                    origin, keys=keys, has_exclude=flags.has_exclude
                )
                result.extend(
                    expand_indices(origin, e, col_names=col_names, exclude=exclude)
                )
    elif flags.has_wildcard:
        exclude = prepare_excluded(origin, keys=keys, has_exclude=flags.has_exclude)
        result.extend(replace_wildcard(origin, col_names=col_names, exclude=exclude))
    else:
        exclude = prepare_excluded(origin, keys=keys, has_exclude=flags.has_exclude)
        expanded = rewrite_special_aliases(origin)
        result.append(expanded)
    return tuple(result)


def _iter_exclude_names(origin: ExprIR, /) -> Iterator[str]:
    """Yield all excluded names in `origin`."""
    for e in origin.iter_left():
        if isinstance(e, Exclude):
            yield from e.names


def prepare_excluded(
    origin: ExprIR, /, keys: GroupByKeys, *, has_exclude: bool
) -> Excluded:
    """Huge simplification of https://github.com/pola-rs/polars/blob/0fa7141ce718c6f0a4d6ae46865c867b177a59ed/crates/polars-plan/src/plans/conversion/expr_expansion.rs#L484-L555."""
    exclude: set[str] = set()
    if has_exclude:
        exclude.update(_iter_exclude_names(origin))
    for group_by_key in keys:
        if name := group_by_key.meta.output_name(raise_if_undetermined=False):
            exclude.add(name)
    return frozenset(exclude)


def _all_columns_match(origin: ExprIR, /, columns: Columns) -> bool:
    it = (e == columns if isinstance(e, Columns) else True for e in origin.iter_left())
    return all(it)


def expand_columns(
    origin: ExprIR, /, columns: Columns, *, exclude: Excluded
) -> Seq[ExprIR]:
    if not _all_columns_match(origin, columns):
        msg = "expanding more than one `col` is not allowed"
        raise ComputeError(msg)
    result: deque[ExprIR] = deque()
    for name in columns.names:
        if name not in exclude:
            expanded = replace_with_column(origin, Columns, name)
            expanded = rewrite_special_aliases(expanded)
            result.append(expanded)
    return tuple(result)


def expand_indices(
    origin: ExprIR,
    /,
    indices: IndexColumns,
    *,
    col_names: FrozenColumns,
    exclude: Excluded,
) -> Seq[ExprIR]:
    result: deque[ExprIR] = deque()
    n_fields = len(col_names)
    for index in indices.indices:
        if not is_index_in_range(index, n_fields):
            raise column_index_error(index, col_names)
        name = col_names[index]
        if name not in exclude:
            expanded = replace_with_column(origin, IndexColumns, name)
            expanded = rewrite_special_aliases(expanded)
            result.append(expanded)
    return tuple(result)


def replace_wildcard(
    origin: ExprIR, /, *, col_names: FrozenColumns, exclude: Excluded
) -> Seq[ExprIR]:
    result: deque[ExprIR] = deque()
    for name in col_names:
        if name not in exclude:
            expanded = replace_with_column(origin, All, name)
            expanded = rewrite_special_aliases(expanded)
            result.append(expanded)
    return tuple(result)


def rewrite_special_aliases(origin: ExprIR, /) -> ExprIR:
    """Expand `KeepName` and `RenameAlias` into `Alias`.

    Warning:
        Only valid **after**
        - Expanding all selections into `Column`
        - Dealing with `FunctionExpr.input`
    """
    from narwhals._plan import meta

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
