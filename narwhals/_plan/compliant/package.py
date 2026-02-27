"""Defines the expected shape of an `__init__.py` for a backend.

Rather than using a `Namespace` and accessing classes via properties with inline imports:
why not just use the import system?

TypeVar(s) would have an outer scope to avoid cycles
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.compliant.typing import Native as NativeLazy
from narwhals._utils import unstable

# ruff: noqa: N802

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa
    from typing_extensions import TypeAlias

    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.compliant.expr import CompliantExpr
    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
    from narwhals._plan.compliant.namespace import CompliantNamespace
    from narwhals._plan.compliant.scalar import CompliantScalar
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.plans.visitors import LogicalToResolved, ResolvedToCompliant
    from narwhals._plan.typing import (
        NativeDataFrameT as NativeEager,
        NativeSeriesT as NativeSeries,
    )

Incomplete: TypeAlias = Any


@unstable
class CompliantPackage(Protocol[NativeLazy]):
    """Could be something we resolve *once*, and then treat everything like a plugin?

    We'd take these exports and transform them into something that describes capabilities.
    """

    # NOTE: Annotating the owner of the native type only
    @property
    def DataFrame(
        self,
    ) -> type[CompliantDataFrame[Incomplete, NativeEager, Incomplete]] | None:
        """Optional for lazy-only."""

    @property
    def Series(self) -> type[CompliantSeries[NativeSeries]] | None:
        """Optional for lazy-only."""

    @property
    def Expr(self) -> type[CompliantExpr[Incomplete, Incomplete]]:
        """Required, but not implemented for `polars` yet."""
        ...

    @property
    def Scalar(self) -> type[CompliantScalar[Incomplete, Incomplete]] | None:
        """Optional for *at-least* `polars`."""

    @property
    def Namespace(self) -> type[CompliantNamespace[Incomplete, Incomplete, Incomplete]]:
        """Required, but the protocol should not include types that are exported here."""
        ...

    @property
    def LazyFrame(self) -> type[CompliantLazyFrame[NativeLazy]]:
        """Required, but has a much smaller footprint than on main."""
        ...

    @property
    def PlanResolver(self) -> type[LogicalToResolved] | None:
        """Optional, can default to `_plan.plans.conversion.Resolver`."""

    @property
    def PlanEvaluator(self) -> type[ResolvedToCompliant[NativeLazy]] | Incomplete:
        """Required, but not implemented for `pyarrow` yet."""


# NOTE: `pyright` is fine with this
def try_package(package: CompliantPackage[NativeLazy]) -> CompliantPackage[NativeLazy]:
    return package


# NOTE: Argument 1 to "try_package" has incompatible type Module; expected "CompliantPackage[Table]"  [arg-type]
# mypy: disable-error-code="arg-type"
def try_arrow() -> CompliantPackage[pa.Table]:
    import narwhals._plan.arrow

    return try_package(narwhals._plan.arrow)


# NOTE: Argument 1 to "try_package" has incompatible type Module; expected "CompliantPackage[LazyFrame]" [arg-type]
def try_polars() -> CompliantPackage[pl.LazyFrame]:
    import narwhals._plan.polars

    return try_package(narwhals._plan.polars)
