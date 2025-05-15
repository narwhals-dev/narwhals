from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Literal
import typing as t

from narwhals._plan.common import ExprIR

if t.TYPE_CHECKING:
    from narwhals._plan.common import Function
    from narwhals._plan.operators import Operator
    from narwhals._plan.options import FunctionOptions
    from narwhals._plan.options import SortMultipleOptions
    from narwhals._plan.options import SortOptions
    from narwhals._plan.window import Window
    from narwhals.dtypes import DType
    from narwhals.typing import PythonLiteral


class Alias(ExprIR):
    expr: ExprIR
    name: str


class Column(ExprIR):
    name: str


class Columns(ExprIR):
    names: t.Sequence[str]


class Literal(ExprIR):
    value: PythonLiteral


class BinaryExpr(ExprIR):
    """Application of two exprs via an `Operator`.

    This ✅
    - https://github.com/pola-rs/polars/blob/6df23a09a81c640c21788607611e09d9f43b1abc/crates/polars-plan/src/plans/aexpr/mod.rs#L152-L155

    Not this ❌
    - https://github.com/pola-rs/polars/blob/da27decd9a1adabe0498b786585287eb730d1d91/crates/polars-plan/src/dsl/function_expr/mod.rs#L127
    """

    left: ExprIR
    op: Operator
    right: ExprIR


class Cast(ExprIR):
    expr: ExprIR
    dtype: DType


class Sort(ExprIR):
    expr: ExprIR
    options: SortOptions


class SortBy(ExprIR):
    """https://github.com/narwhals-dev/narwhals/issues/2534."""

    expr: ExprIR
    by: t.Sequence[ExprIR]
    options: SortMultipleOptions


class FunctionExpr(ExprIR):
    """Polars uses seemingly for namespacing, but maybe I'll use for traversal?

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L123
    """

    input: t.Sequence[ExprIR]
    function: Function
    options: FunctionOptions


class Filter(ExprIR):
    expr: ExprIR
    by: ExprIR


class WindowExpr(ExprIR):
    """https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L129-L136."""

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


class Len(ExprIR): ...


class Exclude(ExprIR):
    names: t.Sequence[str]


class Nth(ExprIR):
    index: int


class IndexColumns(ExprIR):
    """Renamed from `IndexColumn`.

    `Nth` provides the single variant.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L80
    """

    indices: t.Sequence[int]


class All(ExprIR): ...


# NOTE: by_dtype, matches, numeric, boolean, string, categorical, datetime, all
class Selector(ExprIR): ...
