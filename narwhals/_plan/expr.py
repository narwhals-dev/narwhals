from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Literal
import typing as t

from narwhals._plan.common import ExprIR

if t.TYPE_CHECKING:
    from narwhals._plan.common import Function
    from narwhals._plan.literal import LiteralValue
    from narwhals._plan.operators import Operator
    from narwhals._plan.options import FunctionOptions
    from narwhals._plan.options import SortMultipleOptions
    from narwhals._plan.options import SortOptions
    from narwhals._plan.window import Window
    from narwhals.dtypes import DType


class Alias(ExprIR):
    __slots__ = ("expr", "name")

    expr: ExprIR
    name: str

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar


class Column(ExprIR):
    __slots__ = ("name",)

    name: str


class Columns(ExprIR):
    __slots__ = ("names",)

    names: t.Sequence[str]


class Literal(ExprIR):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L81."""

    __slots__ = ("value",)

    value: LiteralValue

    @property
    def is_scalar(self) -> bool:
        return self.value.is_scalar


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


class Cast(ExprIR):
    __slots__ = ("dtype", "expr")

    expr: ExprIR
    dtype: DType

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar


class Sort(ExprIR):
    __slots__ = ("expr", "options")

    expr: ExprIR
    options: SortOptions

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar


class SortBy(ExprIR):
    """https://github.com/narwhals-dev/narwhals/issues/2534."""

    __slots__ = ("by", "expr", "options")

    expr: ExprIR
    by: t.Sequence[ExprIR]
    options: SortMultipleOptions

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar


class FunctionExpr(ExprIR):
    """Polars uses *seemingly* for namespacing, but maybe I'll use for traversal?

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L123
    """

    __slots__ = ("function", "input", "options")

    input: t.Sequence[ExprIR]
    function: Function
    options: FunctionOptions
    """Assuming this is **either**:

    1. `function.function_options`
    2. The union of (1) and any `FunctionOptions` in `inputs`
    """


class Filter(ExprIR):
    __slots__ = ("by", "expr")

    expr: ExprIR
    by: ExprIR

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar and self.by.is_scalar


class WindowExpr(ExprIR):
    """https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L129-L136."""

    __slots__ = ("expr", "options", "order_by", "partition_by")

    expr: ExprIR
    """Renamed from `function`."""

    partition_by: t.Sequence[ExprIR]
    order_by: tuple[ExprIR, SortOptions] | None
    options: Window
    """Little confused on the nesting.

    - We don't allow choosing `WindowMapping` kinds
    - Haven't ventured into rolling much yet

    Expr::Window { options: WindowType::Over(WindowMapping) }
    Expr::Window { options: WindowType::Rolling(RollingGroupOptions) }
    """


class Len(ExprIR):
    @property
    def is_scalar(self) -> bool:
        return True


class Exclude(ExprIR):
    __slots__ = ("names",)

    names: t.Sequence[str]


class Nth(ExprIR):
    __slots__ = ("index",)

    index: int


class IndexColumns(ExprIR):
    """Renamed from `IndexColumn`.

    `Nth` provides the single variant.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L80
    """

    __slots__ = ("indices",)

    indices: t.Sequence[int]


class All(ExprIR): ...


# NOTE: by_dtype, matches, numeric, boolean, string, categorical, datetime, all
class Selector(ExprIR): ...
