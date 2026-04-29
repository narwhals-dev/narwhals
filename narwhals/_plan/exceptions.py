"""Exceptions and tools to format them."""

from __future__ import annotations

from collections import Counter
from itertools import groupby
from typing import TYPE_CHECKING, Final, Literal

from narwhals._utils import Implementation, qualified_type_name
from narwhals.exceptions import (
    ColumnNotFoundError,
    ComputeError,
    DuplicateError,
    InvalidIntoExprError,
    InvalidOperationError,
    MultiOutputExpressionError,
    ShapeError,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence
    from typing import Any

    from typing_extensions import TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan._function import Function
    from narwhals._plan.expressions.operators import Operator
    from narwhals._plan.options import SortOptions
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import IntoExpr, Seq
    from narwhals.dtypes import DType
    from narwhals.typing import Backend, IntoBackend, IntoDType, IntoSchema

ExprFunction: TypeAlias = Literal["filter", "when", "sort_by"]
SelectorValue: TypeAlias = Literal["index", "name"]

Unsupported: TypeAlias = Literal[
    "concat",
    "date_range",
    "int_range",
    "linear_space",
    "read_csv",
    "read_csv_schema",
    "read_parquet",
    "read_parquet_schema",
    "scan_csv",
    "scan_parquet",
    "DataFrame",
    "DataFrame.from_dict",
    "LazyFrame.collect",
    "Series.from_iterable",
    "v1",
    "v2",
]
"""A method/function/feature that is not supported for a backend."""

OMIT_PARENS: Final = frozenset[Unsupported](("v1", "v2"))

# NOTE: Using verbose names to start with
# TODO @dangotbanned: Think about something better/more consistent once the new messages are finalized


# TODO @dangotbanned: Use arguments in error message
def agg_scalar_error(agg: ir.AggExpr, scalar: ir.ExprIR, /) -> InvalidOperationError:  # noqa: ARG001
    msg = "Can't apply aggregations to scalar-like expressions."
    return InvalidOperationError(msg)


def function_expr_invalid_operation_error(
    function: Function, parent: ir.ExprIR
) -> InvalidOperationError:
    msg = f"Cannot use `{function!r}()` on aggregated expression `{parent!r}`."
    return InvalidOperationError(msg)


def function_arg_non_scalar_error(function: Function, value: Any) -> ShapeError:
    msg = f"`{function!r}()` does not support non-scalar expressions, got: `{value!r}`."
    return ShapeError(msg)


def function_arity_error(
    function: Function, arity: Literal[1, 2, 3, "*"], inputs: Collection[ir.ExprIR]
) -> TypeError:
    exprs = "" if not inputs else f":\n{format_expressions(*inputs)}"
    msg = f"Expected {arity} {'input' if arity == 1 else 'inputs'} for `{function!r}()`, got {len(inputs)}{exprs}"
    return TypeError(msg)


def literal_type_error(value: Any) -> TypeError:
    msg = f"{qualified_type_name(value)!r} is not supported in `nw.lit`"
    if not isinstance(value, type):
        msg = f"{msg}, got: {value!r}."
    return TypeError(msg)


# TODO @dangotbanned: Use arguments in error message
def hist_bins_monotonic_error(bins: Seq[float]) -> ComputeError:  # noqa: ARG001
    msg = "bins must increase monotonically"
    return ComputeError(msg)


def shape_error(expected_length: int, actual_length: int) -> ShapeError:
    msg = f"Expected object of length {expected_length}, got {actual_length}."
    return ShapeError(msg)


def _binary_underline(
    left: ir.ExprIR,
    operator: Operator,
    right: ir.ExprIR,
    /,
    *,
    underline_left: bool = False,
    underline_right: bool = True,
) -> str:
    lhs, op, rhs = repr(left), repr(operator), repr(right)
    if underline_right and not underline_left:
        indent = (len(lhs) + len(op) + 2) * " "
        underline = len(rhs) * "^"
    elif underline_right:
        indent = ""
        pad_middle = (2 + len(op)) * " "
        underline = (len(lhs) * "^") + pad_middle + (len(rhs) * "^")
    else:
        indent = ""
        underline = len(lhs) * "^"
    return f"{lhs} {op} {rhs}\n{indent}{underline}"


# TODO @dangobanned: Make fancier error for `when`
# - *maybe* underline things?
def combination_mixed_multi_output_error(
    origin: ir.BinaryExpr | ir.TernaryExpr | ir.ExprIR, lengths: Sequence[int]
) -> MultiOutputExpressionError:
    from narwhals._plan import expressions as ir

    if isinstance(origin, ir.TernaryExpr):
        # Flip the order from "root first" -> "as written"
        truthy, predicate, falsy = lengths
        lengths = (predicate, truthy, falsy)
        extra = repr(origin)
    elif isinstance(origin, ir.BinaryExpr):
        lhs, op, rhs = origin.left, origin.op, origin.right
        extra = _binary_underline(lhs, op, rhs, underline_right=lengths[0] < lengths[1])
    else:  # pragma: no cover
        extra = repr(origin)
    why = f"({' != '.join(map(repr, lengths))})"
    what = "Cannot combine selectors that produce a different number of columns"
    return MultiOutputExpressionError(f"{what} {why}.\n{extra}")


def binary_expr_length_changing_error(
    left: ir.ExprIR, op: Operator, right: ir.ExprIR, kind: Literal["mixed", "multi"]
) -> InvalidOperationError:
    expr = _binary_underline(left, op, right, underline_left=(kind == "multi"))
    if kind == "mixed":
        msg = "Cannot combine length-changing expressions with length-preserving ones."
    else:
        msg = "Length-changing expressions can only be used in isolation, or followed by an aggregation."
    msg = f"{msg}\n{expr}"
    return InvalidOperationError(msg)


# TODO @dangotbanned: (low-priority) Underline which part is not length-preserving (outer/inner)
# `col('a').first()` (first)
# `nwp.int_range(2).sort()` (int_range)
def sort_by_key_length_changing_error(expr: ir.ExprIR) -> InvalidOperationError:
    msg = f"All `sort_by` expression keys must be length-preserving, got:\n`{expr!r}`"
    return InvalidOperationError(msg)


# TODO @dangotbanned: Use arguments in error message
def over_nested_error(
    expr: ir.Over,  # noqa: ARG001
    partition_by: Seq[ir.ExprIR],  # noqa: ARG001
    order_by: Seq[ir.ExprIR] = (),  # noqa: ARG001
    sort_options: SortOptions | None = None,  # noqa: ARG001
) -> InvalidOperationError:
    msg = "Cannot nest `over` statements."
    return InvalidOperationError(msg)


# TODO @dangotbanned: Use arguments in error message
def over_elementwise_error(
    expr: ir.FunctionExpr,
    partition_by: Seq[ir.ExprIR],  # noqa: ARG001
    order_by: Seq[ir.ExprIR] = (),  # noqa: ARG001
    sort_options: SortOptions | None = None,  # noqa: ARG001
) -> InvalidOperationError:
    msg = f"Cannot use `over` on expressions which are elementwise.\n{expr!r}"
    return InvalidOperationError(msg)


# TODO @dangotbanned: Use arguments in error message
def over_row_separable_error(
    expr: ir.FunctionExpr,
    partition_by: Seq[ir.ExprIR],  # noqa: ARG001
    order_by: Seq[ir.ExprIR] = (),  # noqa: ARG001
    sort_options: SortOptions | None = None,  # noqa: ARG001
) -> InvalidOperationError:
    msg = f"Cannot use `over` on expressions which change length.\n{expr!r}"
    return InvalidOperationError(msg)


def over_order_by_names_error(
    expr: ir.OverOrdered, by: ir.ExprIR
) -> InvalidOperationError:
    if by.meta.is_column_selection(allow_aliasing=True):
        # narwhals dev error
        msg = (
            f"Cannot use `{type(expr).__name__}.order_by_names()` before expression expansion.\n"
            f"Found unresolved selection {by!r}, in:\n{expr!r}"
        )
    else:
        # user error
        msg = (
            f"Only column selection expressions are supported in `over(order_by=...)`.\n"
            f"Found {by!r}, in:\n{expr!r}"
        )
    return InvalidOperationError(msg)


def invalid_into_expr_error(
    first_input: Iterable[IntoExpr],
    more_inputs: tuple[Any, ...],
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


def at_least_one_error(method: ExprFunction, /) -> TypeError:
    predicate = "predicate or constraint"
    kind = {"filter": predicate, "when": predicate, "sort_by": "sort key"}[method]
    msg = f"at least one {kind} must be provided"
    return TypeError(msg)


def is_iterable_error(obj: object, /) -> TypeError:
    msg = (
        f"Expected Narwhals class or scalar, got: {qualified_type_name(obj)!r}.\n\n"
        "Hint: Perhaps you\n"
        "- forgot a `nw.from_native` somewhere?\n"
        "- used `pl.col` instead of `nw.col`?"
    )
    return TypeError(msg)


def duplicate_error(exprs: Collection[ir.ExprIR]) -> DuplicateError:
    INDENT = "\n  "  # noqa: N806
    names = [_output_name(expr) for expr in exprs]
    exprs = sorted(exprs, key=_output_name)
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


def duplicate_names_error(names: Collection[str]) -> DuplicateError:
    msg = "\n".join(f"- {k!r} {v} times" for k, v in Counter(names).items() if v > 1)
    msg = f"Expected unique column names, but found duplicates:\n\n{msg}"
    return DuplicateError(msg)


def _output_name(expr: ir.ExprIR) -> str:
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
) -> ColumnNotFoundError:
    # NOTE: If the original expression used a negative index, we should use that as well
    n_names = len(tuple(schema_or_column_names))
    max_nth = f"`nth({n_names - 1})`" if index >= 0 else f"`nth(-{n_names})`"
    msg = f"Invalid column index {index!r}\nHint: The schema's last column is {max_nth}"
    return ColumnNotFoundError(msg)


def one_or_iterable_type_error(
    kind: SelectorValue, inner: object, outer: Iterable[object] | None = None, /
) -> TypeError:
    msg = f"invalid {kind}: {inner!r}"
    msg = msg if outer is None else f"{msg} in {outer!r}"
    return TypeError(msg)


# TODO @dangotbanned: Remove or get coverage for failing:
# - `GroupByResolver.key_names`
# - `DataFrameGroupBy.key_names`
def group_by_no_keys_error() -> ComputeError:  # pragma: no cover
    msg = "at least one key is required in a group_by operation"
    return ComputeError(msg)


def format_expressions(*exprs: ir.ExprIR, indent: int = 2) -> str:
    """Return an indented list of `exprs` reprs."""
    indent_str = " " * indent
    return "\n".join(f"{indent_str}{e!r}" for e in exprs)


def selectors_not_found_error(
    selectors: Iterable[ir.SelectorIR], schema: IntoSchema | FrozenSchema
) -> ColumnNotFoundError:
    selectors = tuple(selectors)
    msg = "Found no columns when expanding:"
    if len(selectors) == 1:
        msg = f"{msg} {next(iter(selectors))!r}"
    else:
        msg = f"{msg}\n{format_expressions(*selectors)}"
    items = dict(schema)
    msg = f"{msg}\n\nHint: Did you mean one of these columns: {items!r}?"
    return ColumnNotFoundError(msg)


def expand_multi_output_error(
    origin: ir.ExprIR, child: ir.ExprIR, *expanded: ir.ExprIR
) -> MultiOutputExpressionError:
    msg = (
        "Multi-output expressions are not supported in this context.\n"
        f"Got `{origin!r}`, but `{child!r}` expanded into {len(expanded)} outputs:\n"
        f"{format_expressions(*expanded)}"
    )
    return MultiOutputExpressionError(msg)


def unsupported_error(
    backend: IntoBackend[Backend] | Any, name: Unsupported, /
) -> NotImplementedError:  # pragma: no cover
    """Return an error for an unsupported operation in `backend`."""
    if not isinstance(backend, str):
        backend = Implementation.from_backend(backend).value
    what = name if name in OMIT_PARENS else f"{name}()"
    msg = f"`{what}` is not yet supported for {backend!r}"
    return NotImplementedError(msg)


def invalid_dtype_operation_error(
    dtype: IntoDType, method_name: str, *expected: DType | type[DType]
) -> InvalidOperationError:  # pragma: no cover
    msg = f"`{method_name}` operation is not supported for dtype `{dtype}`"
    if expected:
        if len(expected) == 1:
            allow = f"{expected[0]}"
        else:
            allow = " or ".join(str(tp) for tp in expected)
        msg = f"{msg}, expected {allow} type"
    return InvalidOperationError(msg)
