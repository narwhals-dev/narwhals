"""Top-level `Expr` nodes."""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING, Generic, final

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._expr_ir import ExprIR, SelectorIR
from narwhals._plan._nodes import node, nodes
from narwhals._plan.exceptions import (
    function_expr_invalid_operation_error,
    over_order_by_names_error,
    range_expr_non_scalar_error,
)
from narwhals._plan.expressions.selectors import ByName
from narwhals._plan.typing import (
    FunctionT_co,
    Ignored,
    LeftSelectorT,
    LeftT,
    OperatorT,
    RangeT_co,
    RightSelectorT,
    RightT,
    RollingT_co,
    SelectorOperatorT,
    SelectorT,
    Seq,
    StructT_co,
)
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from typing_extensions import Self

    from narwhals._plan.compliant.typing import Ctx, FrameT_contra, R_co
    from narwhals._plan.expressions import selectors as cs
    from narwhals._plan.expressions.functions import MapBatches  # noqa: F401
    from narwhals._plan.options import FunctionOptions, SortMultipleOptions, SortOptions
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType

__all__ = [
    "Alias",
    "AnonymousExpr",
    "BinaryExpr",
    "BinarySelector",
    "Cast",
    "Column",
    "Filter",
    "FunctionExpr",
    "Len",
    "Over",
    "RollingExpr",
    "RootSelector",
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

    Important:
        All expressions that change the output name are
        resolved and removed following expression expansion.
        This means that you can do arbitrarily complex renames,
        **at the narwhals-level** but there is intentionally no support
        for them at the compliant-level.
    """

    __slots__ = ("expr", "name")
    expr: ExprIR = node()
    """The expression to rename."""
    name: str
    """The new name."""

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self

    def __repr__(self) -> str:
        return f"{self.expr!r}.alias({self.name!r})"


class Column(ExprIR, dispatch=namespaced("col")):
    __slots__ = ("name",)
    name: str

    def __repr__(self) -> str:
        return f"col({self.name!r})"

    def to_selector_ir(self) -> RootSelector:
        return ByName.from_name(self.name).to_selector_ir()

    def resolve_dtype(self, schema: FrozenSchema) -> DType:
        return schema[self.name]

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self


class BinaryExpr(ExprIR, Generic[LeftT, OperatorT, RightT]):
    """Application of two exprs via an `Operator`."""

    __slots__ = ("left", "op", "right")
    left: LeftT = node()
    op: OperatorT
    right: RightT = node()

    def __repr__(self) -> str:
        return f"[({self.left!r}) {self.op!r} ({self.right!r})]"

    def resolve_dtype(self, schema: FrozenSchema) -> DType:
        """NOTE: Supported on `Logical` and `TrueDivide` operators only.

        Requires `get_supertype`:
        - `Add`
        - `Sub`
        - `Multiply`
        - `FloorDivide`
        - `Modulus`
        """
        return self.op.resolve_dtype(self, schema)


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
# mypy: disable-error-code="misc"
class FunctionExpr(ExprIR, Generic[FunctionT_co]):
    """**Representing `Expr::Function`**.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L114-L120
    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L123
    """

    __slots__ = ("function", "input", "options")
    input: Seq[ExprIR] = nodes()
    # NOTE: mypy being mypy - the top error can't be silenced 🤦‍♂️
    # narwhals/_plan/expr.py: error: Cannot use a covariant type variable as a parameter  [misc]
    # narwhals/_plan/expr.py:272:15: error: Cannot use a covariant type variable as a parameter  [misc]
    #         function: FunctionT_co  # noqa: ERA001
    #                   ^
    # Found 2 errors in 1 file (checked 476 source files)
    function: FunctionT_co
    """Operation applied to each element of `input`."""

    options: FunctionOptions
    """Combined flags from chained operations."""

    def is_scalar(self) -> bool:
        return self.function.is_scalar

    def __repr__(self) -> str:
        if self.input:
            first = self.input[0]
            if len(self.input) >= 2:
                return f"{first!r}.{self.function!r}({list(self.input[1:])!r})"
            return f"{first!r}.{self.function!r}()"
        return f"{self.function!r}()"

    # NOTE: Interacting badly with `pyright` synthesizing the `__replace__` signature
    if not TYPE_CHECKING:

        def __init__(
            self,
            *,
            input: Seq[ExprIR],  # noqa: A002
            function: FunctionT_co,
            options: FunctionOptions,
            **kwds: t.Any,
        ) -> None:
            parent = input[0]
            if parent.is_scalar() and not options.is_elementwise():
                raise function_expr_invalid_operation_error(function, parent)
            kwargs = dict(input=input, function=function, options=options, **kwds)
            super().__init__(**kwargs)
    else:  # pragma: no cover
        ...

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


class RollingExpr(FunctionExpr[RollingT_co]):
    def dispatch(
        self: Self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str
    ) -> R_co:
        return self.__expr_ir_dispatch__(self, ctx, frame, name)


class AnonymousExpr(FunctionExpr["MapBatches"], dispatch=renamed("map_batches")):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L158-L166."""

    def dispatch(
        self: Self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str
    ) -> R_co:
        return self.__expr_ir_dispatch__(self, ctx, frame, name)

    def resolve_dtype(self, schema: FrozenSchema) -> DType:
        if dtype := self.function.return_dtype:
            return dtype
        return super().resolve_dtype(schema)


class RangeExpr(FunctionExpr[RangeT_co]):
    """E.g. `int_range(...)`.

    Special-cased as it is only allowed scalar inputs, and is row_separable.
    """

    def __init__(
        self,
        *,
        input: Seq[ExprIR],  # noqa: A002
        function: RangeT_co,
        options: FunctionOptions,
        **kwds: t.Any,
    ) -> None:
        # NOTE: `IntRange` has 2x scalar inputs, so always triggered error in parent
        if len(input) < 2:
            msg = f"Expected at least 2 inputs for `{function!r}()`, but got `{len(input)}`.\n`{input}`"
            raise InvalidOperationError(msg)
        if not all(e.is_scalar() for e in input):
            raise range_expr_non_scalar_error(input, function)
        super(ExprIR, self).__init__(
            **dict(input=input, function=function, options=options, **kwds)
        )

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

    def resolve_dtype(self, schema: FrozenSchema) -> DType:
        msg = f"Unable to resolve dtype for {(type(self).__name__)!r}:\n{self!r}\n\n"
        "Requires `get_supertype` and `nw.Null`:\n"
        " - https://github.com/narwhals-dev/narwhals/issues/2835\n"
        " - https://github.com/narwhals-dev/narwhals/pull/3396\n\n"
        "See Also: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/schema.rs#L257-L273"
        raise NotImplementedError(msg)


@final
class RootSelector(SelectorIR):
    """A single selector expression."""

    __slots__ = ("selector",)
    selector: cs.Selector

    def __repr__(self) -> str:
        return f"{self.selector!r}"

    def iter_expand_names(
        self, schema: FrozenSchema, ignored_columns: Ignored
    ) -> Iterator[str]:
        yield from self.selector.iter_expand(schema, ignored_columns)

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self

    def matches(self, dtype: IntoDType) -> bool:
        return self.selector.to_dtype_selector().matches(dtype)

    def to_dtype_selector(self) -> Self:
        return self.__replace__(selector=self.selector.to_dtype_selector())


@final
class BinarySelector(
    SelectorIR, Generic[LeftSelectorT, SelectorOperatorT, RightSelectorT]
):
    """Application of two selector exprs via a set operator."""

    __slots__ = ("left", "op", "right")
    left: LeftSelectorT
    op: SelectorOperatorT
    right: RightSelectorT

    def __repr__(self) -> str:
        return f"[({self.left!r}) {self.op!r} ({self.right!r})]"

    def iter_expand_names(
        self, schema: FrozenSchema, ignored_columns: Ignored
    ) -> Iterator[str]:
        # by_name, by_index (upstream) lose their ability to reorder when used as a binary op
        # (As designed) https://github.com/pola-rs/polars/issues/19384
        names = schema.names
        left = frozenset(self.left.iter_expand_names(schema, ignored_columns))
        right = frozenset(self.right.iter_expand_names(schema, ignored_columns))
        remaining: frozenset[str] = self.op(left, right)
        target: Iterable[str]
        if remaining:
            target = (
                names
                if len(remaining) == len(names)
                else (nm for nm in names if nm in remaining)
            )
        else:
            target = ()
        yield from target

    def matches(self, dtype: IntoDType) -> bool:
        left = self.left.matches(dtype)
        right = self.right.matches(dtype)
        return bool(self.op(left, right))

    def to_dtype_selector(self) -> Self:
        return self.__replace__(
            left=self.left.to_dtype_selector(), right=self.right.to_dtype_selector()
        )


@final
class InvertSelector(SelectorIR, Generic[SelectorT]):
    __slots__ = ("selector",)
    selector: SelectorT

    def __repr__(self) -> str:
        return f"~{self.selector!r}"

    def iter_expand_names(
        self, schema: FrozenSchema, ignored_columns: Ignored
    ) -> Iterator[str]:
        # by_name, by_index (upstream) lose their ability to reorder when used as a binary op
        # that includes invert, which is implemented as Difference(All, Selector)
        # (As designed) https://github.com/pola-rs/polars/issues/19384
        names = schema.names
        ignore = frozenset(self.selector.iter_expand_names(schema, ignored_columns))
        target: Iterable[str]
        if ignore:
            target = (
                ()
                if len(ignore) == len(names)
                else (nm for nm in names if nm not in ignore)
            )
        else:
            target = names
        yield from target

    def matches(self, dtype: IntoDType) -> bool:
        return not self.selector.to_dtype_selector().matches(dtype)

    def to_dtype_selector(self) -> Self:
        return self.__replace__(selector=self.selector.to_dtype_selector())


def ternary_expr(predicate: ExprIR, truthy: ExprIR, falsy: ExprIR, /) -> TernaryExpr:
    return TernaryExpr(predicate=predicate, truthy=truthy, falsy=falsy)
