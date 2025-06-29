"""Exceptions and tools to format them."""

from __future__ import annotations

from collections import Counter
from itertools import groupby
from typing import TYPE_CHECKING

from narwhals.exceptions import (
    ColumnNotFoundError,
    ComputeError,
    DuplicateError,
    InvalidIntoExprError,
    InvalidOperationError,
    LengthChangingExprError,
    MultiOutputExpressionError,
    ShapeError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    import pandas as pd
    import polars as pl

    from narwhals._plan.aggregation import Agg
    from narwhals._plan.common import ExprIR, Function
    from narwhals._plan.expr import FunctionExpr, WindowExpr
    from narwhals._plan.operators import Operator
    from narwhals._plan.options import SortOptions
    from narwhals._plan.typing import IntoExpr, Seq


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
    order_by: Seq[ExprIR] = (),  # noqa: ARG001
    sort_options: SortOptions | None = None,  # noqa: ARG001
) -> InvalidOperationError:
    msg = "Cannot nest `over` statements."
    return InvalidOperationError(msg)


# TODO @dangotbanned: Use arguments in error message
def over_elementwise_error(
    expr: FunctionExpr[Function],
    partition_by: Seq[ExprIR],  # noqa: ARG001
    order_by: Seq[ExprIR] = (),  # noqa: ARG001
    sort_options: SortOptions | None = None,  # noqa: ARG001
) -> InvalidOperationError:
    msg = f"Cannot use `over` on expressions which are elementwise.\n{expr!r}"
    return InvalidOperationError(msg)


# TODO @dangotbanned: Use arguments in error message
def over_row_separable_error(
    expr: FunctionExpr[Function],
    partition_by: Seq[ExprIR],  # noqa: ARG001
    order_by: Seq[ExprIR] = (),  # noqa: ARG001
    sort_options: SortOptions | None = None,  # noqa: ARG001
) -> InvalidOperationError:
    msg = f"Cannot use `over` on expressions which change length.\n{expr!r}"
    return InvalidOperationError(msg)


def invalid_into_expr_error(
    first_input: Iterable[IntoExpr],
    more_inputs: tuple[IntoExpr, ...],
    named_inputs: dict[str, IntoExpr],
    /,
) -> InvalidIntoExprError:
    named = f"\n{named_inputs!r}" if named_inputs else ""
    msg = (
        f"Passing both iterable and positional inputs is not supported.\n"
        f"Hint:\nInstead try collecting all arguments into a {type(first_input).__name__!r}\n"
        f"{first_input!r}\n{more_inputs!r}{named}"
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


def duplicate_error(exprs: Seq[ExprIR]) -> DuplicateError:
    INDENT = "\n  "  # noqa: N806
    names = [_output_name(expr) for expr in exprs]
    duplicates = {k for k, v in Counter(names).items() if v > 1}
    group_by_name = groupby(exprs, _output_name)
    name_exprs = {
        k: INDENT.join(f"{el!r}" for el in it)
        for k, it in group_by_name
        if k in duplicates
    }
    msg = "\n".join(f"[{name!r}]{INDENT}{e}" for name, e in name_exprs.items())
    msg = f"Expected unique column names, but found duplicates:\n\n{msg}"
    return DuplicateError(msg)


def _output_name(expr: ExprIR) -> str:
    return expr.meta.output_name()


def column_not_found_error(
    subset: Iterable[str], /, available: Iterable[str]
) -> ColumnNotFoundError:
    """Similar to `utils.check_columns_exist`, but when we already know there are missing.

    Signature differs to allow passing in a schema to `available`.
    That form is what we're working with here.
    """
    available = tuple(available)
    missing = set(subset).difference(available)
    return ColumnNotFoundError.from_missing_and_available_column_names(missing, available)


def column_index_error(
    index: int, schema_or_column_names: Iterable[str], /
) -> ComputeError:
    # NOTE: If the original expression used a negative index, we should use that as well
    n_names = len(tuple(schema_or_column_names))
    max_nth = f"`nth({n_names - 1})`" if index >= 0 else f"`nth(-{n_names})`"
    msg = f"Invalid column index {index!r}\nHint: The schema's last column is {max_nth}"
    return ComputeError(msg)
