from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from narwhals._plan._version import into_version
from narwhals._utils import Version, unstable

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa

    from narwhals._plan import DataFrame, LazyFrame
    from narwhals._plan.typing import IntoExpr, OneOrIterable
    from narwhals._typing import Arrow, Polars
    from narwhals.typing import EagerAllowed, IntoBackend, LazyAllowed

__all__ = ("select",)


# NOTE: (Per known backend)
# Both overloads share the same [fully static] return type
# - 1x Simply match the single keyword (should look like `.from_*` constructors)
# - 1x Repeat, explicitly fill in the inverse default, add `**named_exprs`
#   - This prevents overlapping between all *possible* keywords
# [fully static]: https://typing.python.org/en/latest/spec/glossary.html#term-fully-static-type


# `Arrow` -> `DataFrame`
@overload
def select(
    *exprs: OneOrIterable[IntoExpr], eager: Arrow
) -> DataFrame[pa.Table, pa.ChunkedArray[Any]]: ...
@overload
def select(
    *exprs: OneOrIterable[IntoExpr],
    eager: Arrow,
    lazy: None = ...,
    **named_exprs: IntoExpr,
) -> DataFrame[pa.Table, pa.ChunkedArray[Any]]: ...


# `Polars` -> `DataFrame`
@overload
def select(
    *exprs: OneOrIterable[IntoExpr], eager: Polars
) -> DataFrame[pl.DataFrame, pl.Series]: ...
@overload
def select(
    *exprs: OneOrIterable[IntoExpr],
    eager: Polars,
    lazy: None = ...,
    **named_exprs: IntoExpr,
) -> DataFrame[pl.DataFrame, pl.Series]: ...


# `Polars` -> `LazyFrame`
@overload
def select(*exprs: OneOrIterable[IntoExpr], lazy: Polars) -> LazyFrame[pl.LazyFrame]: ...
@overload
def select(
    *exprs: OneOrIterable[IntoExpr],
    eager: None = ...,
    lazy: Polars,
    **named_exprs: IntoExpr,
) -> LazyFrame[pl.LazyFrame]: ...


# NOTE: (Per keyword)
# Both overloads share the same [gradual] return type
# Same ideas as (Per known backend), but using the widest possible parameter type
# [gradual]: https://typing.python.org/en/latest/spec/glossary.html#term-gradual-form


# `*` -> `DataFrame`
@overload
def select(
    *exprs: OneOrIterable[IntoExpr], eager: IntoBackend[EagerAllowed]
) -> DataFrame[Any, Any]: ...
@overload
def select(
    *exprs: OneOrIterable[IntoExpr],
    eager: IntoBackend[EagerAllowed],
    lazy: None = ...,
    **named_exprs: IntoExpr,
) -> DataFrame[Any, Any]: ...


# `*` -> `LazyFrame`
@overload
def select(
    *exprs: OneOrIterable[IntoExpr], lazy: IntoBackend[LazyAllowed]
) -> LazyFrame[Any]: ...
@overload
def select(
    *exprs: OneOrIterable[IntoExpr],
    eager: None = ...,
    lazy: IntoBackend[LazyAllowed],
    **named_exprs: IntoExpr,
) -> LazyFrame[Any]: ...


@unstable
def select(
    *exprs: OneOrIterable[IntoExpr],
    eager: IntoBackend[EagerAllowed] | None = None,
    lazy: IntoBackend[LazyAllowed] | None = None,
    **named_exprs: IntoExpr,
) -> DataFrame[Any, Any] | LazyFrame[Any]:
    """Run narwhals expressions without a context.

    This is syntactic sugar for running `frame.select` on an empty DataFrame or LazyFrame.

    Important:
        Exactly one of `eager` or `lazy` must be provided.

    Arguments:
        *exprs: Column(s) to select, specified as positional arguments.
            Accepts expression input. Strings are parsed as column names,
            other non-expression inputs are parsed as literals.
        **named_exprs: Additional columns to select, specified as keyword arguments.
            The columns will be renamed to the keyword used.
        eager: The eager backend to use.
        lazy: The lazy backend to use.

    Tip:
        If you need to rename a column to either `"eager"` or `"lazy"`, specify it as
        `col(...).alias("eager")` to avoid colliding with the keyword arguments.
    """
    if eager is None:
        if lazy is None:
            msg = f"Either `eager` or `lazy` may be None, got: {eager=}, {lazy=}"
            raise TypeError(msg)
        from narwhals._plan.plans import LogicalPlan

        return (
            LogicalPlan.scan_empty()
            .to_narwhals(lazy, Version.MAIN)
            .select(*exprs, **named_exprs)
        )

    if lazy is not None:
        msg = f"Either `eager` or `lazy` may be provided, got: {eager=}, {lazy=}"
        raise TypeError(msg)
    return (
        into_version(Version.MAIN)
        .dataframe.from_dict({}, backend=eager)
        .select(*exprs, **named_exprs)
    )
