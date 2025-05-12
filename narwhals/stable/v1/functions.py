from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence
from typing import TypeVar
from typing import cast

import narwhals as nw
from narwhals.stable.v1.utils import _stableify

if TYPE_CHECKING:
    from narwhals.dtypes import DType
    from narwhals.stable.v1.dataframe import DataFrame
    from narwhals.stable.v1.dataframe import LazyFrame
    from narwhals.stable.v1.expr import Expr
    from narwhals.typing import ConcatMethod
    from narwhals.typing import IntoExpr
    from narwhals.typing import NonNestedLiteral

    FrameT = TypeVar("FrameT", "DataFrame[Any]", "LazyFrame[Any]")


def all() -> Expr:
    """Instantiate an expression representing all columns.

    Returns:
        A new expression.
    """
    return _stableify(nw.all())


def col(*names: str | Iterable[str]) -> Expr:
    """Creates an expression that references one or more columns by their name(s).

    Arguments:
        names: Name(s) of the columns to use.

    Returns:
        A new expression.
    """
    return _stableify(nw.col(*names))


def exclude(*names: str | Iterable[str]) -> Expr:
    """Creates an expression that excludes columns by their name(s).

    Arguments:
        names: Name(s) of the columns to exclude.

    Returns:
        A new expression.
    """
    return _stableify(nw.exclude(*names))


def nth(*indices: int | Sequence[int]) -> Expr:
    """Creates an expression that references one or more columns by their index(es).

    Notes:
        `nth` is not supported for Polars version<1.0.0. Please use
        [`narwhals.col`][] instead.

    Arguments:
        indices: One or more indices representing the columns to retrieve.

    Returns:
        A new expression.
    """
    return _stableify(nw.nth(*indices))


def len() -> Expr:
    """Return the number of rows.

    Returns:
        A new expression.
    """
    return _stableify(nw.len())


def lit(value: NonNestedLiteral, dtype: DType | type[DType] | None = None) -> Expr:
    """Return an expression representing a literal value.

    Arguments:
        value: The value to use as literal.
        dtype: The data type of the literal value. If not provided, the data type will
            be inferred by the native library.

    Returns:
        A new expression.
    """
    return _stableify(nw.lit(value, dtype))


def min(*columns: str) -> Expr:
    """Return the minimum value.

    Note:
       Syntactic sugar for ``nw.col(columns).min()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Returns:
        A new expression.
    """
    return _stableify(nw.min(*columns))


def max(*columns: str) -> Expr:
    """Return the maximum value.

    Note:
       Syntactic sugar for ``nw.col(columns).max()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Returns:
        A new expression.
    """
    return _stableify(nw.max(*columns))


def mean(*columns: str) -> Expr:
    """Get the mean value.

    Note:
        Syntactic sugar for ``nw.col(columns).mean()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Returns:
        A new expression.
    """
    return _stableify(nw.mean(*columns))


def median(*columns: str) -> Expr:
    """Get the median value.

    Notes:
        - Syntactic sugar for ``nw.col(columns).median()``
        - Results might slightly differ across backends due to differences in the
            underlying algorithms used to compute the median.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Returns:
        A new expression.
    """
    return _stableify(nw.median(*columns))


def sum(*columns: str) -> Expr:
    """Sum all values.

    Note:
        Syntactic sugar for ``nw.col(columns).sum()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Returns:
        A new expression.
    """
    return _stableify(nw.sum(*columns))


def sum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Sum all values horizontally across columns.

    Warning:
        Unlike Polars, we support horizontal sum over numeric columns only.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.sum_horizontal(*exprs))


def all_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    r"""Compute the bitwise AND horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.all_horizontal(*exprs))


def any_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    r"""Compute the bitwise OR horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.any_horizontal(*exprs))


def mean_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Compute the mean of all values horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.mean_horizontal(*exprs))


def min_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Get the minimum value horizontally across columns.

    Notes:
        We support `min_horizontal` over numeric columns only.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.min_horizontal(*exprs))


def max_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Get the maximum value horizontally across columns.

    Notes:
        We support `max_horizontal` over numeric columns only.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.max_horizontal(*exprs))


def concat(items: Iterable[FrameT], *, how: ConcatMethod = "vertical") -> FrameT:
    """Concatenate multiple DataFrames, LazyFrames into a single entity.

    Arguments:
        items: DataFrames, LazyFrames to concatenate.
        how: concatenating strategy

            - vertical: Concatenate vertically. Column names must match.
            - horizontal: Concatenate horizontally. If lengths don't match, then
                missing rows are filled with null values. This is only supported
                when all inputs are (eager) DataFrames.
            - diagonal: Finds a union between the column schemas and fills missing column
                values with null.

    Returns:
        A new DataFrame or LazyFrame resulting from the concatenation.

    Raises:
        TypeError: The items to concatenate should either all be eager, or all lazy
    """
    return cast("FrameT", _stableify(nw.concat(items, how=how)))


def concat_str(
    exprs: IntoExpr | Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    separator: str = "",
    ignore_nulls: bool = False,
) -> Expr:
    r"""Horizontally concatenate columns into a single string column.

    Arguments:
        exprs: Columns to concatenate into a single string column. Accepts expression
            input. Strings are parsed as column names, other non-expression inputs are
            parsed as literals. Non-`String` columns are cast to `String`.
        *more_exprs: Additional columns to concatenate into a single string column,
            specified as positional arguments.
        separator: String that will be used to separate the values of each column.
        ignore_nulls: Ignore null values (default is `False`).
            If set to `False`, null values will be propagated and if the row contains any
            null values, the output is null.

    Returns:
        A new expression.
    """
    return _stableify(
        nw.concat_str(exprs, *more_exprs, separator=separator, ignore_nulls=ignore_nulls)
    )
