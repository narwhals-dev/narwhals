"""Based on [polars-plan/src/plans/conversion/expr_expansion.rs].

- Goal is to expand every selection into a named column.
- Most will require only the column names of the schema.

[polars-plan/src/plans/conversion/expr_expansion.rs]: https://github.com/pola-rs/polars/blob/df4d21c30c2b383b651e194f8263244f2afaeda3/crates/polars-plan/src/plans/conversion/expr_expansion.rs
"""

# ruff: noqa: A002
from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from narwhals._plan.common import Immutable

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


def expand_function_inputs(origin: ExprIR, /, *, schema: FrozenSchema) -> ExprIR:
    raise NotImplementedError


def rewrite_projections(
    input: Seq[ExprIR],  # `FunctionExpr.input`
    /,
    keys: Seq[ExprIR],
    *,
    schema: FrozenSchema,
) -> Seq[ExprIR]:
    raise NotImplementedError


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
    result: Seq[ExprIR],
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
    result: Seq[ExprIR],
    columns: expr.Columns,  # `polars` uses columns.names
    *,
    col_names: FrozenColumns,
    exclude: Excluded,
) -> Inplace:
    raise NotImplementedError


def expand_dtypes(
    origin: ExprIR,
    /,
    result: Seq[ExprIR],
    dtypes: selectors.ByDType,  # we haven't got `DtypeColumn`
    *,
    schema: FrozenSchema,
    exclude: Excluded,
) -> Inplace:
    raise NotImplementedError


def expand_indices(
    origin: ExprIR,
    /,
    result: Seq[ExprIR],
    indices: expr.IndexColumns,
    *,
    schema: FrozenSchema,
    exclude: Excluded,
) -> Inplace:
    raise NotImplementedError


def replace_wildcard(
    origin: ExprIR, /, result: Seq[ExprIR], *, col_names: FrozenColumns, exclude: Excluded
) -> Inplace:
    raise NotImplementedError


def replace_wildcard_with_column(origin: ExprIR, /, column_name: str) -> ExprIR:
    """`expr.All` and `Exclude`."""
    raise NotImplementedError


def rewrite_special_aliases(origin: ExprIR, /) -> ExprIR:
    """`KeepName` and `RenameAlias`.

    Reuses some of the `meta` functions to traverse the names.
    """
    raise NotImplementedError


def replace_dtype_or_index_with_column(
    origin: ExprIR, /, column_name: str, *, replace_dtype: bool
) -> ExprIR:
    raise NotImplementedError


def dtypes_match(left: DType, right: DType | type[DType]) -> bool:
    return left == right


def replace_regex(
    origin: ExprIR,
    /,
    result: Seq[ExprIR],
    pattern: selectors.Matches,
    *,
    col_names: FrozenColumns,
    exclude: Excluded,
) -> Inplace:
    raise NotImplementedError


def expand_regex(
    origin: ExprIR, /, result: Seq[ExprIR], *, col_names: FrozenColumns, exclude: Excluded
) -> Inplace:
    raise NotImplementedError
