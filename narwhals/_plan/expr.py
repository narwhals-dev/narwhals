from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Literal
import typing as t

from narwhals._plan.common import ExprIR

if t.TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.common import Function
    from narwhals._plan.common import Seq
    from narwhals._plan.functions import MapBatches
    from narwhals._plan.functions import RollingWindow
    from narwhals._plan.literal import LiteralValue
    from narwhals._plan.operators import Operator
    from narwhals._plan.options import FunctionOptions
    from narwhals._plan.options import SortMultipleOptions
    from narwhals._plan.options import SortOptions
    from narwhals._plan.window import Window
    from narwhals.dtypes import DType

_FunctionT = t.TypeVar("_FunctionT", bound="Function")
_RollingT = t.TypeVar("_RollingT", bound="RollingWindow")


class Alias(ExprIR):
    __slots__ = ("expr", "name")

    expr: ExprIR
    name: str

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.alias({self.name!r})"


class Column(ExprIR):
    __slots__ = ("name",)

    name: str

    def __repr__(self) -> str:
        return f"col({self.name!r})"


class Columns(ExprIR):
    __slots__ = ("names",)

    names: Seq[str]

    def __repr__(self) -> str:
        return f"cols({list(self.names)!r})"


class Literal(ExprIR):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L81."""

    __slots__ = ("value",)

    value: LiteralValue

    @property
    def is_scalar(self) -> bool:
        return self.value.is_scalar

    @property
    def dtype(self) -> DType:
        return self.value.dtype

    def __repr__(self) -> str:
        return f"lit({self.value!r})"


class BinaryExpr(ExprIR):
    """Application of two exprs via an `Operator`.

    This ✅
    - https://github.com/pola-rs/polars/blob/6df23a09a81c640c21788607611e09d9f43b1abc/crates/polars-plan/src/plans/aexpr/mod.rs#L152-L155

    Not this ❌
    - https://github.com/pola-rs/polars/blob/da27decd9a1adabe0498b786585287eb730d1d91/crates/polars-plan/src/dsl/function_expr/mod.rs#L127
    """

    __slots__ = ("left", "op", "right")

    left: ExprIR
    op: Operator
    right: ExprIR

    @property
    def is_scalar(self) -> bool:
        return self.left.is_scalar and self.right.is_scalar

    def __repr__(self) -> str:
        return f"[({self.left!r}) {self.op!r} ({self.right!r})]"


class Cast(ExprIR):
    __slots__ = ("dtype", "expr")

    expr: ExprIR
    dtype: DType

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.cast({self.dtype!r})"


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


class FunctionExpr(ExprIR, t.Generic[_FunctionT]):
    """**Representing `Expr::Function`**.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L114-L120

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L123
    """

    __slots__ = ("function", "input", "options")

    input: Seq[ExprIR]
    function: _FunctionT
    """Enum type is named `FunctionExpr` in `polars`.

    Mirroring *exactly* doesn't make much sense in OOP.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L123
    """

    options: FunctionOptions
    """Assuming this is **either**:

    1. `function.function_options`
    2. The union of (1) and any `FunctionOptions` in `inputs`
    """

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


class RollingExpr(FunctionExpr[_RollingT]): ...


class AnonymousFunctionExpr(ExprIR):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L158-L166."""

    __slots__ = ("function", "input", "options")

    input: Seq[ExprIR]
    function: MapBatches
    options: FunctionOptions

    @property
    def is_scalar(self) -> bool:
        return self.function.function_options.returns_scalar()


class Filter(ExprIR):
    __slots__ = ("by", "expr")

    expr: ExprIR
    by: ExprIR

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar and self.by.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.filter({self.by!r})"


class WindowExpr(ExprIR):
    """A fully specified `.over()`, that occured after another expression.

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
    order_by: tuple[ExprIR, SortOptions] | None
    options: Window
    """Little confused on the nesting.

    - We don't allow choosing `WindowMapping` kinds
    - Haven't ventured into rolling much yet
      - Turns out this is for `Expr.rolling` (not `Expr.rolling_<agg>`)
      - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L879-L888

    Expr::Window { options: WindowType::Over(WindowMapping) }
    Expr::Window { options: WindowType::Rolling(RollingGroupOptions) }
    """


class Len(ExprIR):
    @property
    def is_scalar(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "len()"


class Exclude(ExprIR):
    __slots__ = ("names",)

    names: Seq[str]


class Nth(ExprIR):
    __slots__ = ("index",)

    index: int

    def __repr__(self) -> str:
        return f"nth({self.index})"


class IndexColumns(ExprIR):
    """Renamed from `IndexColumn`.

    `Nth` provides the singlular variant.

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
        return "*"


# NOTE: by_dtype, matches, numeric, boolean, string, categorical, datetime, all
class Selector(ExprIR): ...
