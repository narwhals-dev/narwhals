"""Top-level `Expr` nodes."""

# mypy: disable-error-code="misc"
# NOTE: Sadly no way to disable *just* the variance inference part
from __future__ import annotations

from typing import TYPE_CHECKING, Generic

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._expr_ir import ExprIR, SelectorIR
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._nodes import node, nodes
from narwhals._plan.exceptions import over_order_by_names_error
from narwhals._plan.expressions.selectors import ByName
from narwhals._plan.typing import (
    FunctionT_co,
    HorizontalT_co,
    LeftT_co,
    OperatorT,
    RangeT_co,
    RightT_co,
    RollingT_co,
    Seq,
    StructT_co,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from narwhals._plan._expansion import Expander
    from narwhals._plan.compliant.typing import Ctx, FrameT_contra, R_co
    from narwhals._plan.expressions.functions import MapBatches  # noqa: F401
    from narwhals._plan.options import SortMultipleOptions, SortOptions
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType


__all__ = [
    "Alias",
    "AnonymousExpr",
    "BinaryExpr",
    "Cast",
    "Column",
    "Filter",
    "FunctionExpr",
    "Len",
    "Over",
    "RollingExpr",
    "Sort",
    "SortBy",
    "StructExpr",
    "TernaryExpr",
    "col",
    "ternary_expr",
]

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
get_dtype = ResolveDType.get_dtype
same_dtype = ResolveDType.expr_ir.same_dtype
namespaced = DispatcherOptions.namespaced
renamed = DispatcherOptions.renamed


def col(name: str, /) -> Column:
    return Column(name=name)


class Alias(ExprIR, dispatch="no_dispatch"):
    """Rename an expression.

    Arguments:
        expr: An expression with a root of exactly one `Column`.
        name: The new name.

    Important:
        All expressions that change the output name are
        resolved and removed following expression expansion.
        This means that you can do arbitrarily complex renames,
        **at the narwhals-level** but there is intentionally no support
        for them at the compliant-level.
    """

    __slots__ = ("expr", "name")
    expr: ExprIR = node()
    name: str

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self

    def __repr__(self) -> str:
        return f"{self.expr!r}.alias({self.name!r})"


class Column(ExprIR, dispatch=namespaced("col")):
    """An expression that selects exactly one column.

    Arguments:
        name: A single column name.

    Examples:
        >>> import narwhals._plan as nw
        >>> expr = nw.col("one")
        >>> expr._ir
        col('one')
        >>> print(expr._ir)
        Column(name='one')

        A `Column` is not a selector, but can be converted into one:
        >>> expr.meta.as_selector()._ir
        ncs.by_name('one')
    """

    __slots__ = ("name",)
    name: str

    def __repr__(self) -> str:
        return f"col({self.name!r})"

    def to_selector_ir(self) -> SelectorIR:
        return ByName.from_name(self.name)

    def resolve_dtype(self, schema: FrozenSchema) -> DType:
        return schema[self.name]

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self


class BinaryExpr(ExprIR, Generic[LeftT_co, OperatorT, RightT_co]):
    """A binary operation applied to two expressions."""

    __slots__ = ("left", "op", "right")
    left: LeftT_co = node()
    op: OperatorT
    right: RightT_co = node()

    def __repr__(self) -> str:
        return f"[({self.left!r}) {self.op!r} ({self.right!r})]"

    def resolve_dtype(self, schema: FrozenSchema) -> DType:  # pragma: no cover
        """NOTE: Supported on `Logical` and `TrueDivide` operators only.

        Requires `get_supertype`:
        - `Add`
        - `Sub`
        - `Multiply`
        - `FloorDivide`
        - `Modulus`
        """
        return self.op.resolve_dtype(self, schema)

    def iter_expand(self, ctx: Expander, /) -> Iterator[ExprIR]:
        yield from self.__expr_ir_nodes__.iter_expand_by_combination(self, ctx)


class Cast(ExprIR, dtype=get_dtype()):
    __slots__ = ("dtype", "expr")
    expr: ExprIR = node()
    dtype: DType

    def __repr__(self) -> str:
        return f"{self.expr!r}.cast({self.dtype!r})"


class Sort(ExprIR, dtype=same_dtype()):
    __slots__ = ("expr", "options")
    expr: ExprIR = node()
    options: SortOptions

    def __repr__(self) -> str:
        direction = "desc" if self.options.descending else "asc"
        return f"{self.expr!r}.sort({direction})"


class SortBy(ExprIR, dtype=same_dtype()):
    __slots__ = ("expr", "by", "options")  # noqa: RUF023
    expr: ExprIR = node()
    by: Seq[ExprIR] = nodes()
    options: SortMultipleOptions

    def __repr__(self) -> str:
        opts = ""
        if any(self.descending):
            opts += f", descending={list(self.descending)}"
        if any(self.nulls_last):
            opts += f", nulls_last={list(self.nulls_last)}"
        return f"{self.expr!r}.sort_by({list(self.by)!r}{opts})"

    @property
    def descending(self) -> Seq[bool]:
        return self.options.descending

    @property
    def nulls_last(self) -> Seq[bool]:
        return self.options.nulls_last


# TODO @dangotbanned: Docs should complement `Function`
# - The two are very tightly coupled
class FunctionExpr(ExprIR, Generic[FunctionT_co]):
    """**Representing `Expr::Function`**.

    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L114-L120
    - https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L123
    """

    __slots__ = ("function", "input")
    input: Seq[ExprIR] = nodes()
    function: FunctionT_co
    """Operation applied to each element of `input`."""

    @property
    def flags(self) -> FunctionFlags:
        return self.function.__function_flags__

    def is_scalar(self) -> bool:
        return FunctionFlags.AGGREGATION in self.flags

    def __repr__(self) -> str:
        if self.input:
            first = self.input[0]
            if len(self.input) >= 2:
                return f"{first!r}.{self.function!r}({list(self.input[1:])!r})"
            return f"{first!r}.{self.function!r}()"
        return f"{self.function!r}()"

    def dispatch(
        self: Self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str
    ) -> R_co:
        return self.function.__expr_ir_dispatch__(self, ctx, frame, name)

    def resolve_dtype(self, schema: FrozenSchema) -> DType:
        """NOTE: Supported on many functions, but there are important gaps.

        Requires `get_supertype`:
        - `{max,min,sum}_horizontal`
        - `coalesce`
        - `replace_strict(..., dtype=None)`

        Partially requires `get_supertype`:
        - `mean_horizontal`
        - `fill_null(value)`

        Unlikely to ever be supported:
        - `map_batches(..., dtype=None)`
        """
        return self.function.resolve_dtype(self, schema)

    # TODO @dangotbanned: Integrate `FunctionExpr` (similar to `Filter`)
    def iter_expand(self, ctx: Expander, /) -> Iterator[ExprIR]:
        input_root, *non_root = self.input
        children = tuple(ctx.only(self, child) for child in non_root) if non_root else ()
        for root in input_root.iter_expand(ctx):
            yield self.__replace__(input=(root, *children))


class RollingExpr(FunctionExpr[RollingT_co]):
    def dispatch(
        self: Self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str
    ) -> R_co:
        return self.__expr_ir_dispatch__(self, ctx, frame, name)


class AnonymousExpr(FunctionExpr["MapBatches"], dispatch=renamed("map_batches")):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L158-L166."""

    @property
    def flags(self) -> FunctionFlags:
        return self.function.flags

    def dispatch(
        self: Self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str
    ) -> R_co:
        return self.__expr_ir_dispatch__(self, ctx, frame, name)

    def resolve_dtype(self, schema: FrozenSchema) -> DType:  # pragma: no cover
        if dtype := self.function.return_dtype:
            return dtype
        return super().resolve_dtype(schema)


class HorizontalExpr(FunctionExpr[HorizontalT_co]):
    iter_expand = ExprIR.iter_expand


class RangeExpr(FunctionExpr[RangeT_co]):
    """E.g. `int_range(...)`."""

    def __repr__(self) -> str:
        return f"{self.function!r}({list(self.input)!r})"


class StructExpr(FunctionExpr[StructT_co]):
    """E.g. `col("a").struct.field(...)`.

    Requires special handling during expression expansion.
    """

    def needs_expansion(self) -> bool:
        return self.function.needs_expansion or super().needs_expansion()

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self


class Filter(ExprIR, dtype=same_dtype()):
    __slots__ = ("expr", "by")  # noqa: RUF023
    expr: ExprIR = node(observe_scalar=False)
    by: ExprIR = node(observe_scalar=False)

    def __repr__(self) -> str:
        return f"{self.expr!r}.filter({self.by!r})"


class Over(ExprIR, dtype=same_dtype()):
    """A fully specified `.over()`, that occurred after another expression.

    Related:
    - https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L129-L136
    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L835-L838
    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L840-L876
    """

    __slots__ = ("expr", "partition_by")
    expr: ExprIR = node(observe_scalar=False)
    """For lazy backends, this should be the only place we allow `rolling_*`, `cum_*`."""
    partition_by: Seq[ExprIR] = nodes()

    def __repr__(self) -> str:
        return f"{self.expr!r}.over({list(self.partition_by)!r})"


class OverOrdered(Over):
    __slots__ = ("order_by", "sort_options")
    order_by: Seq[ExprIR] = nodes()
    sort_options: SortOptions

    def __repr__(self) -> str:
        order = self.order_by
        if not self.partition_by:
            args = f"order_by={list(order)!r}"
        else:
            args = f"partition_by={list(self.partition_by)!r}, order_by={list(order)!r}"
        return f"{self.expr!r}.over({args})"

    def order_by_names(self) -> Iterator[str]:
        """Yield the names resolved from expanding `order_by`.

        Raises:
            InvalidOperationError: If used *before* expansion, or
                `order_by` contains expressions that do more than select.
        """
        for by in self.order_by:
            if isinstance(by, Column):
                yield by.name
            else:
                raise over_order_by_names_error(self, by)


class Len(ExprIR, dispatch=namespaced(), dtype=dtm.IDX_DTYPE):
    def is_scalar(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "len"

    def __repr__(self) -> str:
        return "len()"

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self


# TODO @dangotbanned: `get_supertype`, `nw.Null`
class TernaryExpr(ExprIR):
    """When-Then-Otherwise."""

    __slots__ = ("truthy", "falsy", "predicate")  # noqa: RUF023
    # `truthy` is defined first because the root is from `when(...).then(<here>)`
    truthy: ExprIR = node()
    predicate: ExprIR = node()
    falsy: ExprIR = node()

    def __repr__(self) -> str:
        return (
            f".when({self.predicate!r}).then({self.truthy!r}).otherwise({self.falsy!r})"
        )

    def resolve_dtype(self, schema: FrozenSchema) -> DType:  # pragma: no cover
        msg = f"Unable to resolve dtype for {(type(self).__name__)!r}:\n{self!r}\n\n"
        "Requires `get_supertype` and `nw.Null`:\n"
        " - https://github.com/narwhals-dev/narwhals/issues/2835\n"
        " - https://github.com/narwhals-dev/narwhals/pull/3396\n\n"
        "See Also: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/schema.rs#L257-L273"
        raise NotImplementedError(msg)

    def iter_expand(self, ctx: Expander, /) -> Iterator[ExprIR]:
        yield from self.__expr_ir_nodes__.iter_expand_by_combination(self, ctx)


def ternary_expr(predicate: ExprIR, truthy: ExprIR, falsy: ExprIR, /) -> TernaryExpr:
    return TernaryExpr(predicate=predicate, truthy=truthy, falsy=falsy)
