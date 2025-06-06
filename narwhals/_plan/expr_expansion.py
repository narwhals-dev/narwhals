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
from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from narwhals._plan.common import Immutable
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan import expr, selectors
    from narwhals._plan.common import ExprIR, Seq
    from narwhals._plan.dummy import DummyExpr
    from narwhals.dtypes import DType


FrozenSchema: TypeAlias = "MappingProxyType[str, DType]"
FrozenColumns: TypeAlias = "Seq[str]"
Excluded: TypeAlias = "frozenset[str]"
"""Internally use a `set`, then freeze before returning."""

Inplace: TypeAlias = Any
"""Functions where `polars` does in-place mutations on `Expr`.

Very likely that **we won't** do this in `narwhals`, instead return a new object.
"""

ResultIRs: TypeAlias = "deque[ExprIR]"


# NOTE: Both `_freeze` functions will probably want to be cached
# In the traversal/expand/replacement functions, their returns will be hashable -> safe to cache those as well
def _freeze_schema(**schema: DType) -> FrozenSchema:
    copied = deepcopy(schema)
    return MappingProxyType(copied)


def _freeze_columns(schema: FrozenSchema, /) -> FrozenColumns:
    return tuple(schema)


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
        from narwhals._plan import expr

        multiple_columns: bool = False
        has_nth: bool = False
        has_wildcard: bool = False
        has_selector: bool = False
        has_exclude: bool = False
        for e in ir.iter_left():
            if isinstance(e, (expr.Columns, expr.IndexColumns)):
                multiple_columns = True
            elif isinstance(e, expr.Nth):
                has_nth = True
            elif isinstance(e, expr.All):
                has_wildcard = True
            elif isinstance(e, expr.SelectorIR):
                has_selector = True
            elif isinstance(e, expr.Exclude):
                has_exclude = True
        return ExpansionFlags(
            multiple_columns=multiple_columns,
            has_nth=has_nth,
            has_wildcard=has_wildcard,
            has_selector=has_selector,
            has_exclude=has_exclude,
        )

    @classmethod
    def from_expr(cls, expr: DummyExpr, /) -> ExpansionFlags:
        return cls.from_ir(expr._ir)

    def with_multiple_columns(self) -> ExpansionFlags:
        return ExpansionFlags(
            multiple_columns=True,
            has_nth=self.has_nth,
            has_wildcard=self.has_wildcard,
            has_selector=self.has_selector,
            has_exclude=self.has_exclude,
        )


def prepare_projection(
    exprs: Sequence[ExprIR], schema: Mapping[str, DType]
) -> tuple[Seq[ExprIR], FrozenSchema]:
    frozen_schema = _freeze_schema(**schema)
    rewritten = rewrite_projections(tuple(exprs), keys=(), schema=frozen_schema)
    # NOTE: There's an `expressions_to_schema` step that I'm skipping for now
    # seems too big of a rabbit hole to go down
    return rewritten, frozen_schema


# NOTE: Parameters have been re-ordered, renamed, changed types
# - `origin` is the `Expr` that's being iterated over
# - `result` *haven't got to yet*
#    - Couldn't this just be the return type?
#    - Certainly less complicated in python
# - `<third_param>` is the current child of `origin`
# - `col_names: FrozenColumns` is used when we don't need the dtypes
# - `exclude` is the return of `prepare_excluded`


# NOTE: The inner function is ready
def expand_function_inputs(origin: ExprIR, /, *, schema: FrozenSchema) -> ExprIR:
    from narwhals._plan import expr

    def fn(child: ExprIR, /) -> ExprIR:
        if not (
            isinstance(child, expr.FunctionExpr)
            and child.options.is_input_wildcard_expansion()
        ):
            return child
        return child.with_input(rewrite_projections(child.input, keys=(), schema=schema))

    return origin.map_ir(fn)


def rewrite_projections(
    input: Seq[ExprIR],  # `FunctionExpr.input`
    /,
    keys: Seq[
        ExprIR
    ],  # NOTE: Mutable (empty) array initialized on call (except in `polars_plan::plans::conversion::dsl_to_ir::resolve_group_by`)
    *,  # NOTE: Represents group_by keys
    schema: FrozenSchema,
) -> Seq[ExprIR]:
    # NOTE: This is where the mutable `result` is initialized
    result_length = len(input) + len(schema)
    result: deque[ExprIR] = deque(maxlen=result_length)
    for expr in input:
        expanded = expand_function_inputs(expr, schema=schema)
        flags = ExpansionFlags.from_ir(expanded)
        if flags.has_selector:
            expanded = replace_selector(expanded, keys, schema=schema)
            flags = flags.with_multiple_columns()
        # NOTE: `result` is what I'd want as a return, rather than inplace
        replace_and_add_to_results(
            expanded, result, keys=keys, schema=schema, flags=flags
        )
    return tuple(result)


def replace_selector(
    ir: ExprIR,  # an element of `FunctionExpr.input`
    /,
    keys: Seq[ExprIR],
    *,
    schema: FrozenSchema,
) -> ExprIR:
    raise NotImplementedError


def expand_selector(
    s: expr.SelectorIR, /, keys: Seq[ExprIR], *, schema: FrozenSchema
) -> Seq[str]:
    """Converts into input of `Columns(...)`."""
    raise NotImplementedError


def replace_selector_inner(
    s: expr.SelectorIR,
    /,
    keys: Seq[ExprIR],
    members: Any,  # mutable, insertion order preserving set `PlIndexSet<Expr>`
    scratch: Seq[ExprIR],  # passed as `result` into `replace_and_add_to_results`
    *,
    schema: FrozenSchema,
) -> Inplace:
    raise NotImplementedError


def replace_and_add_to_results(
    origin: ExprIR,
    /,
    result: ResultIRs,
    keys: Seq[ExprIR],
    *,
    schema: FrozenSchema,
    flags: ExpansionFlags,
) -> Inplace:
    raise NotImplementedError


# NOTE: See how far we can get with just the direct node replacements
# - `polars` is using `map_expr`, but I haven't implemented that (yet?)
def replace_nth(nth: expr.Nth, /, col_names: FrozenColumns) -> expr.Column:
    from narwhals._plan import expr

    return expr.Column(name=col_names[nth.index])


def prepare_excluded(
    origin: ExprIR, /, keys: Seq[ExprIR], *, schema: FrozenSchema, has_exclude: bool
) -> Excluded:
    raise NotImplementedError


def expand_columns(
    origin: ExprIR,
    /,
    result: ResultIRs,
    columns: expr.Columns,  # `polars` uses columns.names
    *,
    col_names: FrozenColumns,
    exclude: Excluded,
) -> Inplace:
    raise NotImplementedError


def expand_dtypes(
    origin: ExprIR,
    /,
    result: ResultIRs,
    dtypes: selectors.ByDType,  # we haven't got `DtypeColumn`
    *,
    schema: FrozenSchema,
    exclude: Excluded,
) -> Inplace:
    raise NotImplementedError


def expand_indices(
    origin: ExprIR,
    /,
    result: ResultIRs,
    indices: expr.IndexColumns,
    *,
    schema: FrozenSchema,
    exclude: Excluded,
) -> Inplace:
    raise NotImplementedError


def replace_wildcard(
    origin: ExprIR, /, result: ResultIRs, *, col_names: FrozenColumns, exclude: Excluded
) -> Inplace:
    raise NotImplementedError


def replace_wildcard_with_column(origin: ExprIR, /, column_name: str) -> ExprIR:
    """`expr.All` and `Exclude`."""
    raise NotImplementedError


def rewrite_special_aliases(origin: ExprIR, /) -> ExprIR:
    """Expand `KeepName` and `RenameAlias` into `Alias`.

    Warning:
        Only valid **after**
        - Expanding all selections into `Column`
        - Dealing with `FunctionExpr.input`
    """
    from narwhals._plan import expr, meta

    if meta.has_expr_ir(origin, expr.KeepName, expr.RenameAlias):
        if isinstance(origin, expr.KeepName):
            parent = origin.expr
            roots = parent.meta.root_names()
            alias = next(iter(roots))
            return expr.Alias(expr=parent, name=alias)
        elif isinstance(origin, expr.RenameAlias):
            parent = origin.expr
            leaf_name_or_err = meta.get_single_leaf_name(parent)
            if not isinstance(leaf_name_or_err, str):
                raise leaf_name_or_err
            alias = origin.function(leaf_name_or_err)
            return expr.Alias(expr=parent, name=alias)
        else:
            msg = "`keep`, `suffix`, `prefix` should be last expression"
            raise InvalidOperationError(msg)
    return origin


def replace_dtype_or_index_with_column(
    origin: ExprIR, /, column_name: str, *, replace_dtype: bool
) -> ExprIR:
    raise NotImplementedError


def dtypes_match(left: DType, right: DType | type[DType]) -> bool:
    return left == right


def replace_regex(
    origin: ExprIR,
    /,
    result: ResultIRs,
    pattern: selectors.Matches,
    *,
    col_names: FrozenColumns,
    exclude: Excluded,
) -> Inplace:
    raise NotImplementedError


def expand_regex(
    origin: ExprIR, /, result: ResultIRs, *, col_names: FrozenColumns, exclude: Excluded
) -> Inplace:
    raise NotImplementedError
