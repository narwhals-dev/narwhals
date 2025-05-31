"""Top-level `Expr` nodes."""

from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Literal
import typing as t

from narwhals._plan.aggregation import Agg, OrderableAgg
from narwhals._plan.common import ExprIR, SelectorIR, _field_str, is_non_nested_literal
from narwhals._plan.name import KeepName, RenameAlias
from narwhals._plan.typing import (
    ExprT,
    FunctionT,
    LeftSelectorT,
    LeftT,
    LiteralT,
    Ns,
    OperatorT,
    RightSelectorT,
    RightT,
    RollingT,
    SelectorOperatorT,
)
from narwhals.exceptions import InvalidOperationError
from narwhals.utils import flatten

if t.TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.common import Seq
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
]


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


class Column(ExprIR):
    __slots__ = ("name",)

    name: str

    def __repr__(self) -> str:
        return f"col({self.name!r})"

    def to_compliant(self, plx: Ns[ExprT], /) -> ExprT:
        return plx.col(self.name)


class Columns(ExprIR):
    __slots__ = ("names",)

    names: Seq[str]

    def __repr__(self) -> str:
        return f"cols({list(self.names)!r})"

    def to_compliant(self, plx: Ns[ExprT], /) -> ExprT:
        return plx.col(*self.names)


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


class Cast(ExprIR):
    __slots__ = ("dtype", "expr")

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


class SortBy(ExprIR):
    """https://github.com/narwhals-dev/narwhals/issues/2534."""

    __slots__ = ("by", "expr", "options")

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


class FunctionExpr(ExprIR, t.Generic[FunctionT]):
    """**Representing `Expr::Function`**.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L114-L120

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L123
    """

    __slots__ = ("function", "input", "options")

    input: Seq[ExprIR]
    function: FunctionT
    """Enum type is named `FunctionExpr` in `polars`.

    Mirroring *exactly* doesn't make much sense in OOP.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L123
    """

    options: FunctionOptions
    """Assuming this is **either**:

    1. `function.function_options`
    2. The union of (1) and any `FunctionOptions` in `inputs`
    """

    @property
    def is_scalar(self) -> bool:
        return self.function.is_scalar

    def with_options(self, options: FunctionOptions, /) -> Self:
        options = self.options.with_flags(options.flags)
        return type(self)(input=self.input, function=self.function, options=options)

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
            msg = f"Cannot use `{function!r}()` on aggregated expression `{parent!r}`."
            raise InvalidOperationError(msg)
        super().__init__(**dict(input=input, function=function, options=options, **kwds))


class RollingExpr(FunctionExpr[RollingT]): ...


class AnonymousExpr(FunctionExpr["MapBatches"]):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L158-L166."""


class Filter(ExprIR):
    __slots__ = ("by", "expr")

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


class WindowExpr(ExprIR):
    """A fully specified `.over()`, that occurred after another expression.

    I think we want variants for partitioned, ordered, both.

    Related:
    - https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L129-L136
    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L835-L838
    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L840-L876
    """

    __slots__ = ("expr", "options", "order_by", "partition_by")

    expr: ExprIR
    """Renamed from `function`.

    For lazy backends, this should be the only place we allow `rolling_*`, `cum_*`.
    """

    partition_by: Seq[ExprIR]
    order_by: tuple[Seq[ExprIR], SortOptions] | None
    """Deviates from the `polars` version.

    - `order_by` starts the same as here, but `polars` reduces into a struct - becoming a single (nested) node.
    """

    options: Window
    """Little confused on the nesting.

    - We don't allow choosing `WindowMapping` kinds
    - Haven't ventured into rolling much yet
      - Turns out this is for `Expr.rolling` (not `Expr.rolling_<agg>`)
      - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L879-L888

    Expr::Window { options: WindowType::Over(WindowMapping) }
    Expr::Window { options: WindowType::Rolling(RollingGroupOptions) }
    """

    def __repr__(self) -> str:
        if self.order_by is None:
            return f"{self.expr!r}.over({list(self.partition_by)!r})"
        order, _ = self.order_by
        if not self.partition_by:
            args = f"order_by={list(order)!r}"
        else:
            args = f"partition_by={list(self.partition_by)!r}, order_by={list(order)!r}"
        return f"{self.expr!r}.over({args})"

    def __str__(self) -> str:
        if self.order_by is None:
            order_by = "None"
        else:
            order, opts = self.order_by
            order_by = f"({order}, {opts})"
        args = f"expr={self.expr}, partition_by={self.partition_by}, order_by={order_by}, options={self.options}"
        return f"{type(self).__name__}({args})"

    def iter_left(self) -> t.Iterator[ExprIR]:
        # NOTE: `order_by` is never considered in `polars`
        # https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/plans/iterator.rs#L76-L86
        yield from self.expr.iter_left()
        for e in self.partition_by:
            yield from e.iter_left()
        yield self

    def iter_right(self) -> t.Iterator[ExprIR]:
        yield self
        for e in reversed(self.partition_by):
            yield from e.iter_right()
        yield from self.expr.iter_right()


class Len(ExprIR):
    @property
    def is_scalar(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "len"

    def __repr__(self) -> str:
        return "len()"


class Exclude(ExprIR):
    __slots__ = ("expr", "names")

    expr: ExprIR
    """Default is `all()`."""
    names: Seq[str]
    """We're using a `frozenset` in main.

    Might want to switch to that later.
    """

    @staticmethod
    def from_names(expr: ExprIR, *names: str | t.Iterable[str]) -> Exclude:
        return Exclude(expr=expr, names=tuple(flatten(names)))

    def __repr__(self) -> str:
        return f"{self.expr!r}.exclude({list(self.names)!r})"


class Nth(ExprIR):
    __slots__ = ("index",)

    index: int

    def __repr__(self) -> str:
        return f"nth({self.index})"


class IndexColumns(ExprIR):
    """Renamed from `IndexColumn`.

    `Nth` provides the singular variant.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L80
    """

    __slots__ = ("indices",)

    indices: Seq[int]

    def __repr__(self) -> str:
        return f"index_columns({self.indices!r})"


class All(ExprIR):
    """Aka Wildcard (`pl.all()` or `pl.col("*")`).

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L137
    """

    def __repr__(self) -> str:
        return "all()"


class RootSelector(SelectorIR):
    """A single selector expression."""

    __slots__ = ("selector",)

    selector: Selector
    """by_dtype, matches, numeric, boolean, string, categorical, datetime, all."""

    def __repr__(self) -> str:
        return f"{self.selector!r}"


class BinarySelector(
    _BinaryOp[LeftSelectorT, SelectorOperatorT, RightSelectorT],
    SelectorIR,
    t.Generic[LeftSelectorT, SelectorOperatorT, RightSelectorT],
):
    """Application of two selector exprs via a set operator.

    Note:
        `left` and `right` may also nest other `BinarySelector`s.
    """


class Ternary(ExprIR):
    """When-Then-Otherwise."""

    __slots__ = ("falsy", "predicate", "truthy")

    predicate: ExprIR
    truthy: ExprIR
    falsy: ExprIR

    def __str__(self) -> str:
        # NOTE: Default slot ordering made it difficult to read
        fields = (
            _field_str("predicate", self.predicate),
            _field_str("truthy", self.truthy),
            _field_str("falsy", self.falsy),
        )
        return f"{type(self).__name__}({', '.join(fields)})"

    def __repr__(self) -> str:
        return (
            f".when({self.predicate!r}).then({self.truthy!r}).otherwise({self.falsy!r})"
        )
