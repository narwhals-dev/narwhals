from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Literal
import typing as t  # noqa: F401

from narwhals._plan.common import ExprIR


class Alias(ExprIR): ...


class Column(ExprIR): ...


class Literal(ExprIR): ...


class BinaryExpr(ExprIR):
    """Seems like the application of two exprs via an `Operator`.

    This ✅
    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-python/src/lazyframe/visitor/expr_nodes.rs#L271-L279
    - https://github.com/pola-rs/polars/blob/6df23a09a81c640c21788607611e09d9f43b1abc/crates/polars-plan/src/plans/aexpr/mod.rs#L152-L155
    - https://github.com/pola-rs/polars/blob/da27decd9a1adabe0498b786585287eb730d1d91/crates/polars-expr/src/expressions/binary.rs

    Not this ❌
    - https://github.com/pola-rs/polars/blob/da27decd9a1adabe0498b786585287eb730d1d91/crates/polars-plan/src/dsl/function_expr/mod.rs#L127
    """


class Cast(ExprIR): ...


class Sort(ExprIR): ...


class SortBy(ExprIR):
    """https://github.com/narwhals-dev/narwhals/issues/2534."""


class Filter(ExprIR): ...


class Len(ExprIR): ...


class Exclude(ExprIR): ...


class Nth(ExprIR): ...


class All(ExprIR): ...


# NOTE: by_dtype, matches, numeric, boolean, string, categorical, datetime, all
class Selector(ExprIR): ...
