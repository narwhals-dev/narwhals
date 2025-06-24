"""Top-level `Expr` nodes."""

from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Literal
import typing as t

from narwhals._plan.aggregation import Agg, OrderableAgg
from narwhals._plan.common import (
    ExprIR,
    SelectorIR,
    collect,
    is_non_nested_literal,
    is_regex_projection,
)
from narwhals._plan.exceptions import function_expr_invalid_operation_error
from narwhals._plan.name import KeepName, RenameAlias
from narwhals._plan.typing import (
    ExprT,
    FunctionT,
    LeftSelectorT,
    LeftT,
    LeftT2,
    LiteralT,
    MapIR,
    Ns,
    OperatorT,
    RightSelectorT,
    RightT,
    RightT2,
    RollingT,
    SelectorOperatorT,
    SelectorT,
    Seq,
)
from narwhals._utils import flatten

if t.TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.functions import MapBatches  # noqa: F401
    from narwhals._plan.literal import LiteralValue
    from narwhals._plan.options import FunctionOptions, SortMultipleOptions, SortOptions
    from narwhals._plan.selectors import Selector
    from narwhals._plan.window import Window
    from narwhals.dtypes import DType

__all__ = [
    "Agg",
    "Alias",
    "All",
    "AnonymousExpr",
    "BinaryExpr",
    "BinarySelector",
    "Cast",
    "Column",
    "Columns",
    "Exclude",
    "Filter",
    "FunctionExpr",
    "IndexColumns",
    "KeepName",
    "Len",
    "Literal",
    "Nth",
    "OrderableAgg",
    "RenameAlias",
    "RollingExpr",
    "RootSelector",
    "SelectorIR",
    "Sort",
    "SortBy",
    "Ternary",
    "WindowExpr",
    "col",
]


def col(name: str, /) -> Column:
    """Sugar for a **single** column selection node."""
    return Column(name=name)


class Alias(ExprIR):
    __slots__ = ("expr", "name")

    expr: ExprIR
    name: str

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.alias({self.name!r})"

    def iter_left(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        yield from self.expr.iter_right()

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self.with_expr(self.expr.map_ir(function)))

    def with_expr(self, expr: ExprIR, /) -> Self:
        return self if expr == self.expr else type(self)(expr=expr, name=self.name)


class Column(ExprIR):
    __slots__ = ("name",)

    name: str

    def __repr__(self) -> str:
        return f"col({self.name!r})"

    def to_compliant(self, plx: Ns[ExprT], /) -> ExprT:
        return plx.col(self.name)

    def with_name(self, name: str, /) -> Column:
        return self if name == self.name else col(name)

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self)


class _ColumnSelection(ExprIR):
    """Nodes which can resolve to `Column`(s) with a `Schema`."""

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self)


class Columns(_ColumnSelection):
    __slots__ = ("names",)

    names: Seq[str]

    def __repr__(self) -> str:
        return f"cols({list(self.names)!r})"

    def to_compliant(self, plx: Ns[ExprT], /) -> ExprT:
        return plx.col(*self.names)


class Nth(_ColumnSelection):
    __slots__ = ("index",)

    index: int

    def __repr__(self) -> str:
        return f"nth({self.index})"


class IndexColumns(_ColumnSelection):
    """Renamed from `IndexColumn`.

    `Nth` provides the singular variant.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L80
    """

    __slots__ = ("indices",)

    indices: Seq[int]

    def __repr__(self) -> str:
        return f"index_columns({self.indices!r})"


class All(_ColumnSelection):
    """Aka Wildcard (`pl.all()` or `pl.col("*")`).

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L137
    """

    def __repr__(self) -> str:
        return "all()"


class Exclude(_ColumnSelection):
    __slots__ = ("expr", "names")

    expr: ExprIR
    """Default is `all()`."""
    names: Seq[str]
    """Excluded names.

    - We're using a `frozenset` in main.
    - Might want to switch to that later.
    """

    @staticmethod
    def from_names(expr: ExprIR, *names: str | t.Iterable[str]) -> Exclude:
        flat = flatten(names)
        if any(is_regex_projection(nm) for nm in flat):
            msg = f"Using regex in `exclude(...)` is not yet supported.\nnames={flat!r}"
            raise NotImplementedError(msg)
        return Exclude(expr=expr, names=tuple(flat))

    def __repr__(self) -> str:
        return f"{self.expr!r}.exclude({list(self.names)!r})"

    def iter_left(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        yield from self.expr.iter_right()

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self.with_expr(self.expr.map_ir(function)))

    def with_expr(self, expr: ExprIR, /) -> Self:
        return self if expr == self.expr else type(self)(expr=expr, names=self.names)


class Literal(ExprIR, t.Generic[LiteralT]):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L81."""

    __slots__ = ("value",)

    value: LiteralValue[LiteralT]

    @property
    def is_scalar(self) -> bool:
        return self.value.is_scalar

    @property
    def dtype(self) -> DType:
        return self.value.dtype

    @property
    def name(self) -> str:
        return self.value.name

    def __repr__(self) -> str:
        return f"lit({self.value!r})"

    def to_compliant(self, plx: Ns[ExprT], /) -> ExprT:
        value = self.unwrap()
        if is_non_nested_literal(value):
            return plx.lit(value, self.dtype)
        raise NotImplementedError(type(self.value))

    def unwrap(self) -> LiteralT:
        return self.value.unwrap()

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self)


class _BinaryOp(ExprIR, t.Generic[LeftT, OperatorT, RightT]):
    __slots__ = ("left", "op", "right")

    left: LeftT
    op: OperatorT
    right: RightT

    @property
    def is_scalar(self) -> bool:
        return self.left.is_scalar and self.right.is_scalar

    def __repr__(self) -> str:
        return f"[({self.left!r}) {self.op!r} ({self.right!r})]"


class BinaryExpr(
    _BinaryOp[LeftT, OperatorT, RightT], t.Generic[LeftT, OperatorT, RightT]
):
    """Application of two exprs via an `Operator`."""

    def iter_left(self) -> t.Iterator[ExprIR]:
        yield from self.left.iter_left()
        yield from self.right.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        yield from self.right.iter_right()
        yield from self.left.iter_right()

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.left.iter_output_name()

    def with_left(self, left: LeftT2, /) -> BinaryExpr[LeftT2, OperatorT, RightT]:
        if left == self.left:
            return t.cast("BinaryExpr[LeftT2, OperatorT, RightT]", self)
        return BinaryExpr(left=left, op=self.op, right=self.right)

    def with_right(self, right: RightT2, /) -> BinaryExpr[LeftT, OperatorT, RightT2]:
        if right == self.right:
            return t.cast("BinaryExpr[LeftT, OperatorT, RightT2]", self)
        return BinaryExpr(left=self.left, op=self.op, right=right)

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(
            self.with_left(self.left.map_ir(function)).with_right(
                self.right.map_ir(function)
            )
        )


class Cast(ExprIR):
    __slots__ = ("expr", "dtype")  # noqa: RUF023

    expr: ExprIR
    dtype: DType

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.cast({self.dtype!r})"

    def iter_left(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        yield from self.expr.iter_right()

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_output_name()

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self.with_expr(self.expr.map_ir(function)))

    def with_expr(self, expr: ExprIR, /) -> Self:
        return self if expr == self.expr else type(self)(expr=expr, dtype=self.dtype)


class Sort(ExprIR):
    __slots__ = ("expr", "options")

    expr: ExprIR
    options: SortOptions

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        direction = "desc" if self.options.descending else "asc"
        return f"{self.expr!r}.sort({direction})"

    def iter_left(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        yield from self.expr.iter_right()

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_output_name()

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self.with_expr(self.expr.map_ir(function)))

    def with_expr(self, expr: ExprIR, /) -> Self:
        return self if expr == self.expr else type(self)(expr=expr, options=self.options)


class SortBy(ExprIR):
    """https://github.com/narwhals-dev/narwhals/issues/2534."""

    __slots__ = ("expr", "by", "options")  # noqa: RUF023

    expr: ExprIR
    by: Seq[ExprIR]
    options: SortMultipleOptions

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.sort_by(by={self.by!r}, options={self.options!r})"

    def iter_left(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_left()
        for e in self.by:
            yield from e.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        for e in reversed(self.by):
            yield from e.iter_right()
        yield from self.expr.iter_right()

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_output_name()

    def map_ir(self, function: MapIR, /) -> ExprIR:
        by = (ir.map_ir(function) for ir in self.by)
        return function(self.with_expr(self.expr.map_ir(function)).with_by(by))

    def with_expr(self, expr: ExprIR, /) -> Self:
        if expr == self.expr:
            return self
        return type(self)(expr=expr, by=self.by, options=self.options)

    def with_by(self, by: t.Iterable[ExprIR], /) -> Self:
        by = collect(by)
        if by == self.by:
            return self
        return type(self)(expr=self.expr, by=by, options=self.options)


class FunctionExpr(ExprIR, t.Generic[FunctionT]):
    """**Representing `Expr::Function`**.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L114-L120
    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L123
    """

    __slots__ = ("function", "input", "options")

    input: Seq[ExprIR]
    function: FunctionT
    """Operation applied to each element of `input`.

    Notes:
       [Upstream enum type] is named `FunctionExpr` in `rust`.
       Mirroring *exactly* doesn't make much sense in OOP.

    [Upstream enum type]: https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L123
    """

    options: FunctionOptions
    """Combined flags from chained operations."""

    @property
    def is_scalar(self) -> bool:
        return self.function.is_scalar

    def with_options(self, options: FunctionOptions, /) -> Self:
        options = self.options.with_flags(options.flags)
        return type(self)(input=self.input, function=self.function, options=options)

    def with_input(self, input: t.Iterable[ExprIR], /) -> Self:  # noqa: A002
        input = collect(input)
        if input == self.input:
            return self
        return type(self)(input=input, function=self.function, options=self.options)

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self.with_input(ir.map_ir(function) for ir in self.input))

    def __repr__(self) -> str:
        if self.input:
            first = self.input[0]
            if len(self.input) >= 2:
                return f"{first!r}.{self.function!r}({list(self.input[1:])!r})"
            return f"{first!r}.{self.function!r}()"
        else:
            return f"{self.function!r}()"

    def iter_left(self) -> t.Iterator[ExprIR]:
        for e in self.input:
            yield from e.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        for e in reversed(self.input):
            yield from e.iter_right()

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        """When we have multiple inputs, we want the name of the left-most expression.

        For expr:

            col("c").alias("x").fill_null(50)

        We are interested in the name which comes from the root:

            FunctionExpr(..., [Alias(..., name='...'), Literal(...), ...])
            #                  ^^^^^            ^^^
        """
        for e in self.input[:1]:
            yield from e.iter_output_name()

    def __init__(
        self,
        *,
        input: Seq[ExprIR],  # noqa: A002
        function: FunctionT,
        options: FunctionOptions,
        **kwds: t.Any,
    ) -> None:
        parent = input[0]
        if parent.is_scalar and not options.is_elementwise():
            raise function_expr_invalid_operation_error(function, parent)
        super().__init__(**dict(input=input, function=function, options=options, **kwds))


class RollingExpr(FunctionExpr[RollingT]): ...


class AnonymousExpr(FunctionExpr["MapBatches"]):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L158-L166."""


class Filter(ExprIR):
    __slots__ = ("expr", "by")  # noqa: RUF023

    expr: ExprIR
    by: ExprIR

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar and self.by.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.filter({self.by!r})"

    def iter_left(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_left()
        yield from self.by.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        yield from self.by.iter_right()
        yield from self.expr.iter_right()

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_output_name()

    def map_ir(self, function: MapIR, /) -> ExprIR:
        expr = self.expr.map_ir(function)
        by = self.by.map_ir(function)
        expr = self.expr if self.expr == expr else expr
        by = self.by if self.by == by else by
        return function(Filter(expr=expr, by=by))


# TODO @dangotbanned: Clean up docs/notes
class WindowExpr(ExprIR):
    """A fully specified `.over()`, that occurred after another expression.

    I think we want variants for partitioned, ordered, both.

    Related:
    - https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L129-L136
    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L835-L838
    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L840-L876
    """

    __slots__ = ("expr", "partition_by", "options")  # noqa: RUF023

    expr: ExprIR
    """Renamed from `function`.

    For lazy backends, this should be the only place we allow `rolling_*`, `cum_*`.
    """

    partition_by: Seq[ExprIR]

    options: Window
    """Currently **always** represents over.

    Expr::Window { options: WindowType::Over(WindowMapping) }
    Expr::Window { options: WindowType::Rolling(RollingGroupOptions) }
    """

    def __repr__(self) -> str:
        return f"{self.expr!r}.over({list(self.partition_by)!r})"

    def iter_left(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_left()
        for e in self.partition_by:
            yield from e.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        for e in reversed(self.partition_by):
            yield from e.iter_right()
        yield from self.expr.iter_right()

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_output_name()

    def map_ir(self, function: MapIR, /) -> ExprIR:
        over = self.with_expr(self.expr.map_ir(function)).with_partition_by(
            ir.map_ir(function) for ir in self.partition_by
        )
        return function(over)

    def with_expr(self, expr: ExprIR, /) -> Self:
        if expr == self.expr:
            return self
        return type(self)(expr=expr, partition_by=self.partition_by, options=self.options)

    def with_partition_by(self, partition_by: t.Iterable[ExprIR], /) -> Self:
        by = collect(partition_by)
        if by == self.partition_by:
            return self
        return type(self)(expr=self.expr, partition_by=by, options=self.options)


# TODO @dangotbanned: Reduce repetition from `WindowExpr`
class OrderedWindowExpr(WindowExpr):
    __slots__ = ("expr", "partition_by", "order_by", "sort_options", "options")  # noqa: RUF023

    expr: ExprIR
    partition_by: Seq[ExprIR]
    order_by: Seq[ExprIR]
    """Deviates from the `polars` version.

    - `order_by` starts the same as here, but `polars` reduces into a struct - becoming a single (nested) node.
    """
    sort_options: SortOptions
    options: Window

    def __repr__(self) -> str:
        order = self.order_by
        if not self.partition_by:
            args = f"order_by={list(order)!r}"
        else:
            args = f"partition_by={list(self.partition_by)!r}, order_by={list(order)!r}"
        return f"{self.expr!r}.over({args})"

    def iter_left(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_left()
        for e in self.partition_by:
            yield from e.iter_left()
        for e in self.order_by:
            yield from e.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        for e in reversed(self.order_by):
            yield from e.iter_right()
        for e in reversed(self.partition_by):
            yield from e.iter_right()
        yield from self.expr.iter_right()

    def iter_root_names(self) -> t.Iterator[ExprIR]:
        # NOTE: `order_by` is never considered in `polars`
        # To match that behavior for `root_names` - but still expand in all other cases
        # - this little escape hatch exists
        # https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/plans/iterator.rs#L76-L86
        yield from super().iter_left()

    def map_ir(self, function: MapIR, /) -> ExprIR:
        over = self.with_expr(self.expr.map_ir(function)).with_partition_by(
            ir.map_ir(function) for ir in self.partition_by
        )
        over = over.with_order_by(ir.map_ir(function) for ir in self.order_by)
        return function(over)

    def with_order_by(self, order_by: t.Iterable[ExprIR], /) -> Self:
        by = collect(order_by)
        if by == self.order_by:
            return self
        return type(self)(
            expr=self.expr,
            partition_by=self.partition_by,
            order_by=by,
            sort_options=self.sort_options,
            options=self.options,
        )

    def with_expr(self, expr: ExprIR, /) -> Self:
        if expr == self.expr:
            return self
        return type(self)(
            expr=expr,
            partition_by=self.partition_by,
            order_by=self.order_by,
            sort_options=self.sort_options,
            options=self.options,
        )

    def with_partition_by(self, partition_by: t.Iterable[ExprIR], /) -> Self:
        by = collect(partition_by)
        if by == self.partition_by:
            return self
        return type(self)(
            expr=self.expr,
            partition_by=by,
            order_by=self.order_by,
            sort_options=self.sort_options,
            options=self.options,
        )


class Len(ExprIR):
    @property
    def is_scalar(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "len"

    def __repr__(self) -> str:
        return "len()"

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self)


class RootSelector(SelectorIR):
    """A single selector expression."""

    __slots__ = ("selector",)

    selector: Selector
    """by_dtype, matches, numeric, boolean, string, categorical, datetime, all."""

    def __repr__(self) -> str:
        return f"{self.selector!r}"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return self.selector.matches_column(name, dtype)

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self)


class BinarySelector(
    _BinaryOp[LeftSelectorT, SelectorOperatorT, RightSelectorT],
    SelectorIR,
    t.Generic[LeftSelectorT, SelectorOperatorT, RightSelectorT],
):
    """Application of two selector exprs via a set operator.

    Note:
        `left` and `right` may also nest other `BinarySelector`s.
    """

    def matches_column(self, name: str, dtype: DType) -> bool:
        left = self.left.matches_column(name, dtype)
        right = self.right.matches_column(name, dtype)
        return bool(self.op(left, right))

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self)


class InvertSelector(SelectorIR, t.Generic[SelectorT]):
    __slots__ = ("selector",)

    selector: SelectorT
    """`(Root|Binary)Selector`."""

    def __repr__(self) -> str:
        return f"~{self.selector!r}"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return not self.selector.matches_column(name, dtype)

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self)


class Ternary(ExprIR):
    """When-Then-Otherwise."""

    __slots__ = ("predicate", "truthy", "falsy")  # noqa: RUF023

    predicate: ExprIR
    truthy: ExprIR
    falsy: ExprIR

    @property
    def is_scalar(self) -> bool:
        return self.predicate.is_scalar and self.truthy.is_scalar and self.falsy.is_scalar

    def __repr__(self) -> str:
        return (
            f".when({self.predicate!r}).then({self.truthy!r}).otherwise({self.falsy!r})"
        )

    def iter_left(self) -> t.Iterator[ExprIR]:
        yield from self.truthy.iter_left()
        yield from self.falsy.iter_left()
        yield from self.predicate.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        yield from self.predicate.iter_right()
        yield from self.falsy.iter_right()
        yield from self.truthy.iter_right()

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.truthy.iter_output_name()

    def map_ir(self, function: MapIR, /) -> ExprIR:
        predicate = self.predicate.map_ir(function)
        truthy = self.truthy.map_ir(function)
        falsy = self.falsy.map_ir(function)
        predicate = self.predicate if self.predicate == predicate else predicate
        truthy = self.truthy if self.truthy == truthy else truthy
        falsy = self.falsy if self.falsy == falsy else falsy
        return function(Ternary(predicate=predicate, truthy=truthy, falsy=falsy))
