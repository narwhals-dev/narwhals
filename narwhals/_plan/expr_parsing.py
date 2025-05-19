from __future__ import annotations

# ruff: noqa: A002
from typing import TYPE_CHECKING
from typing import Iterable
from typing import Sequence
from typing import TypeVar

from narwhals._plan.common import is_expr
from narwhals._plan.common import is_iterable_reject
from narwhals.dependencies import get_polars
from narwhals.dependencies import is_pandas_dataframe
from narwhals.dependencies import is_pandas_series
from narwhals.exceptions import InvalidIntoExprError

if TYPE_CHECKING:
    from typing import Any
    from typing import Iterator

    from typing_extensions import TypeAlias
    from typing_extensions import TypeIs

    from narwhals._plan.common import ExprIR
    from narwhals._plan.common import IntoExpr
    from narwhals._plan.common import Seq
    from narwhals.dtypes import DType

T = TypeVar("T")

_RaisesInvalidIntoExprError: TypeAlias = "Any"
"""
Placeholder for multiple `Iterable[IntoExpr]`.

We only support cases `a`, `b`, but the typing for most contexts is more permissive:

>>> import polars as pl
>>> df = pl.DataFrame({"one": ["A", "B", "A"], "two": [1, 2, 3], "three": [4, 5, 6]})
>>> a = ("one", "two")
>>> b = (["one", "two"],)
>>>
>>> c = ("one", ["two"])
>>> d = (["one"], "two")
>>> [df.select(*into) for into in (a, b, c, d)]
[shape: (3, 2)
 ┌─────┬─────┐
 │ one ┆ two │
 │ --- ┆ --- │
 │ str ┆ i64 │
 ╞═════╪═════╡
 │ A   ┆ 1   │
 │ B   ┆ 2   │
 │ A   ┆ 3   │
 └─────┴─────┘,
 shape: (3, 2)
 ┌─────┬─────┐
 │ one ┆ two │
 │ --- ┆ --- │
 │ str ┆ i64 │
 ╞═════╪═════╡
 │ A   ┆ 1   │
 │ B   ┆ 2   │
 │ A   ┆ 3   │
 └─────┴─────┘,
 shape: (3, 2)
 ┌─────┬───────────┐
 │ one ┆ literal   │
 │ --- ┆ ---       │
 │ str ┆ list[str] │
 ╞═════╪═══════════╡
 │ A   ┆ ["two"]   │
 │ B   ┆ ["two"]   │
 │ A   ┆ ["two"]   │
 └─────┴───────────┘,
 shape: (3, 2)
 ┌───────────┬─────┐
 │ literal   ┆ two │
 │ ---       ┆ --- │
 │ list[str] ┆ i64 │
 ╞═══════════╪═════╡
 │ ["one"]   ┆ 1   │
 │ ["one"]   ┆ 2   │
 │ ["one"]   ┆ 3   │
 └───────────┴─────┘]
"""


def parse_into_expr_ir(
    input: IntoExpr, *, str_as_lit: bool = False, dtype: DType | None = None
) -> ExprIR:
    """Parse a single input into an `ExprIR` node."""
    from narwhals._plan import demo as nwd

    if is_expr(input):
        expr = input
    elif isinstance(input, str) and not str_as_lit:
        expr = nwd.col(input)
    else:
        expr = nwd.lit(input, dtype=dtype)
    return expr._ir


def parse_into_seq_of_expr_ir(
    first_input: IntoExpr | Iterable[IntoExpr] = (),
    *more_inputs: IntoExpr | _RaisesInvalidIntoExprError,
    **named_inputs: IntoExpr,
) -> Seq[ExprIR]:
    """Parse variadic inputs into a flat sequence of `ExprIR` nodes."""
    return tuple(_parse_into_iter_expr_ir(first_input, *more_inputs, **named_inputs))


def _parse_into_iter_expr_ir(
    first_input: IntoExpr | Iterable[IntoExpr],
    *more_inputs: IntoExpr,
    **named_inputs: IntoExpr,
) -> Iterator[ExprIR]:
    if not _is_empty_sequence(first_input):
        # NOTE: These need to be separated to introduce an intersection type
        # Otherwise, `str | bytes` always passes through typing
        if _is_iterable(first_input) and not is_iterable_reject(first_input):
            if more_inputs:
                raise _invalid_into_expr_error(first_input, more_inputs, named_inputs)
            else:
                yield from _parse_positional_inputs(first_input)
        else:
            yield parse_into_expr_ir(first_input)
    else:
        # NOTE: Passthrough case for no inputs - but gets skipped when calling next
        yield from ()
    if more_inputs:
        yield from _parse_positional_inputs(more_inputs)
    if named_inputs:
        yield from _parse_named_inputs(named_inputs)


def _parse_positional_inputs(inputs: Iterable[IntoExpr], /) -> Iterator[ExprIR]:
    for into in inputs:
        yield parse_into_expr_ir(into)


def _parse_named_inputs(named_inputs: dict[str, IntoExpr], /) -> Iterator[ExprIR]:
    from narwhals._plan.expr import Alias

    for name, input in named_inputs.items():
        yield Alias(expr=parse_into_expr_ir(input), name=name)


def _is_iterable(obj: Iterable[T] | Any) -> TypeIs[Iterable[T]]:
    if is_pandas_dataframe(obj) or is_pandas_series(obj):
        msg = f"Expected Narwhals class or scalar, got: {type(obj)}. Perhaps you forgot a `nw.from_native` somewhere?"
        raise TypeError(msg)
    if _is_polars(obj):
        msg = (
            f"Expected Narwhals class or scalar, got: {type(obj)}.\n\n"
            "Hint: Perhaps you\n"
            "- forgot a `nw.from_native` somewhere?\n"
            "- used `pl.col` instead of `nw.col`?"
        )
        raise TypeError(msg)
    return isinstance(obj, Iterable)


def _is_empty_sequence(obj: Any) -> bool:
    return isinstance(obj, Sequence) and not obj


def _is_polars(obj: Any) -> bool:
    return (pl := get_polars()) is not None and isinstance(
        obj, (pl.Series, pl.Expr, pl.DataFrame, pl.LazyFrame)
    )


def _invalid_into_expr_error(
    first_input: Any, more_inputs: Any, named_inputs: Any
) -> InvalidIntoExprError:
    msg = (
        f"Passing both iterable and positional inputs is not supported.\n"
        f"Hint:\nInstead try collecting all arguments into a {type(first_input).__name__!r}\n"
        f"{first_input!r}\n{more_inputs!r}\n{named_inputs!r}"
    )
    return InvalidIntoExprError(msg)
