from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._plan._version import into_version
from narwhals._utils import Version

if TYPE_CHECKING:
    from narwhals._plan import DataFrame, LazyFrame
    from narwhals._plan.typing import IntoExpr, OneOrIterable
    from narwhals.typing import EagerAllowed, IntoBackend, LazyAllowed


__all__ = ("select",)


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
