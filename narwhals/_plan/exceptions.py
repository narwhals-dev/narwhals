"""Exceptions and tools to format them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals.exceptions import (
    ComputeError,
    InvalidIntoExprError,
    InvalidOperationError,
    LengthChangingExprError,
    MultiOutputExpressionError,
    ShapeError,
)

if TYPE_CHECKING:
    from typing import Any, Iterable

    import pandas as pd
    import polars as pl

    from narwhals._plan.aggregation import Agg
    from narwhals._plan.common import ExprIR, Function, IntoExpr, Seq
    from narwhals._plan.expr import FunctionExpr, WindowExpr
    from narwhals._plan.operators import Operator
    from narwhals._plan.options import SortOptions


# NOTE: Using verbose names to start with
# TODO @dangotbanned: Think about something better/more consistent once the new messages are finalized


# TODO @dangotbanned: Use arguments in error message
def agg_scalar_error(agg: Agg, scalar: ExprIR, /) -> InvalidOperationError:  # noqa: ARG001
    msg = "Can't apply aggregations to scalar-like expressions."
    return InvalidOperationError(msg)


def function_expr_invalid_operation_error(
    function: Function, parent: ExprIR
) -> InvalidOperationError:
    msg = f"Cannot use `{function!r}()` on aggregated expression `{parent!r}`."
    return InvalidOperationError(msg)


# TODO @dangotbanned: Use arguments in error message
def hist_bins_monotonic_error(bins: Seq[float]) -> ComputeError:  # noqa: ARG001
    msg = "bins must increase monotonically"
    return ComputeError(msg)


# NOTE: Always underlining `right`, since the message refers to both types of exprs
# Assuming the most recent as the issue
def binary_expr_shape_error(left: ExprIR, op: Operator, right: ExprIR) -> ShapeError:
    lhs_op = f"{left!r} {op!r} "
    rhs = repr(right)
    indent = len(lhs_op) * " "
    underline = len(rhs) * "^"
    msg = (
        f"Cannot combine length-changing expressions with length-preserving ones.\n"
        f"{lhs_op}{rhs}\n{indent}{underline}"
    )
    return ShapeError(msg)


# TODO @dangotbanned: Share the right underline code w/ `binary_expr_shape_error`
def binary_expr_multi_output_error(
    left: ExprIR, op: Operator, right: ExprIR
) -> MultiOutputExpressionError:
    lhs_op = f"{left!r} {op!r} "
    rhs = repr(right)
    indent = len(lhs_op) * " "
    underline = len(rhs) * "^"
    msg = (
        "Multi-output expressions are only supported on the "
        f"left-hand side of a binary operation.\n"
        f"{lhs_op}{rhs}\n{indent}{underline}"
    )
    return MultiOutputExpressionError(msg)


def binary_expr_length_changing_error(
    left: ExprIR, op: Operator, right: ExprIR
) -> LengthChangingExprError:
    lhs, rhs = repr(left), repr(right)
    op_s = f" {op!r} "
    underline_left = len(lhs) * "^"
    underline_right = len(rhs) * "^"
    pad_middle = len(op_s) * " "
    msg = (
        "Length-changing expressions can only be used in isolation, "
        "or followed by an aggregation.\n"
        f"{lhs}{op_s}{rhs}\n{underline_left}{pad_middle}{underline_right}"
    )
    return LengthChangingExprError(msg)


# TODO @dangotbanned: Use arguments in error message
def over_nested_error(
    expr: WindowExpr,  # noqa: ARG001
    partition_by: Seq[ExprIR],  # noqa: ARG001
    order_by: tuple[Seq[ExprIR], SortOptions] | None,  # noqa: ARG001
) -> InvalidOperationError:
    msg = "Cannot nest `over` statements."
    return InvalidOperationError(msg)


# TODO @dangotbanned: Use arguments in error message
def over_elementwise_error(
    expr: FunctionExpr[Function],
    partition_by: Seq[ExprIR],  # noqa: ARG001
    order_by: tuple[Seq[ExprIR], SortOptions] | None,  # noqa: ARG001
) -> InvalidOperationError:
    msg = f"Cannot use `over` on expressions which are elementwise.\n{expr!r}"
    return InvalidOperationError(msg)


# TODO @dangotbanned: Use arguments in error message
def over_row_separable_error(
    expr: FunctionExpr[Function],
    partition_by: Seq[ExprIR],  # noqa: ARG001
    order_by: tuple[Seq[ExprIR], SortOptions] | None,  # noqa: ARG001
) -> InvalidOperationError:
    msg = f"Cannot use `over` on expressions which change length.\n{expr!r}"
    return InvalidOperationError(msg)


def invalid_into_expr_error(
    first_input: Iterable[IntoExpr],
    more_inputs: tuple[IntoExpr, ...],
    named_inputs: dict[str, IntoExpr],
    /,
) -> InvalidIntoExprError:
    msg = (
        f"Passing both iterable and positional inputs is not supported.\n"
        f"Hint:\nInstead try collecting all arguments into a {type(first_input).__name__!r}\n"
        f"{first_input!r}\n{more_inputs!r}\n{named_inputs!r}"
    )
    return InvalidIntoExprError(msg)


def is_iterable_pandas_error(obj: pd.DataFrame | pd.Series[Any], /) -> TypeError:
    msg = (
        f"Expected Narwhals class or scalar, got: {type(obj)}. "
        "Perhaps you forgot a `nw.from_native` somewhere?"
    )
    return TypeError(msg)


def is_iterable_polars_error(
    obj: pl.Series | pl.Expr | pl.DataFrame | pl.LazyFrame, /
) -> TypeError:
    msg = (
        f"Expected Narwhals class or scalar, got: {type(obj)}.\n\n"
        "Hint: Perhaps you\n"
        "- forgot a `nw.from_native` somewhere?\n"
        "- used `pl.col` instead of `nw.col`?"
    )
    return TypeError(msg)
