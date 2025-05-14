from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Literal
import typing as t

from narwhals._plan.common import ExprIR

if t.TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan.operators import Operator
    from narwhals.dtypes import DType
    from narwhals.typing import PythonLiteral

    SortOptions: TypeAlias = t.Any
    SortMultipleOptions: TypeAlias = t.Any


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


class Filter(ExprIR):
    expr: ExprIR
    by: ExprIR


class Len(ExprIR): ...


class Exclude(ExprIR):
    names: t.Sequence[str]


class Nth(ExprIR):
    index: int


class IndexColumns(ExprIR):
    indices: t.Sequence[int]


class All(ExprIR): ...


# NOTE: by_dtype, matches, numeric, boolean, string, categorical, datetime, all
class Selector(ExprIR): ...
