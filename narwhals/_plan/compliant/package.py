"""Defines the expected shape of an `__init__.py` for a backend.

Rather than using a `Namespace` and accessing classes via properties with inline imports:
why not just use the import system?

TypeVar(s) would have an outer scope to avoid cycles


Could be something we resolve *once*, and then treat everything like a plugin?

We'd take these exports and transform them into something that describes capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.compliant.typing import Native as NativeLazy
from narwhals._plan.typing import (
    NativeDataFrameT as NativeEager,
    NativeSeriesT as NativeSeries,
    NativeSeriesT_co as NativeSeries_co,
)
from narwhals._typing_compat import TypeVar
from narwhals._utils import unstable

# ruff: noqa: N802, N806

if TYPE_CHECKING:
    from types import ModuleType

    import polars as pl
    import pyarrow as pa
    from typing_extensions import TypeAlias

    from narwhals._plan.arrow.dataframe import ArrowDataFrame
    from narwhals._plan.arrow.expr import ArrowExpr
    from narwhals._plan.arrow.lazyframe import ArrowLazyFrame
    from narwhals._plan.arrow.series import ArrowSeries
    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.compliant.expr import CompliantExpr
    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
    from narwhals._plan.compliant.namespace import CompliantNamespace
    from narwhals._plan.compliant.scalar import CompliantScalar
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.plans.visitors import LogicalToResolved, ResolvedToCompliant
    from narwhals._plan.polars.dataframe import PolarsDataFrame
    from narwhals._plan.polars.expr import PolarsExpr
    from narwhals._plan.polars.lazyframe import PolarsLazyFrame
    from narwhals._plan.polars.series import PolarsSeries

Incomplete: TypeAlias = Any

NativeLazy_co = TypeVar("NativeLazy_co", covariant=True)


class HasLazyFrame(Protocol[NativeLazy]):
    @property
    def LazyFrame(self) -> type[CompliantLazyFrame[NativeLazy]]:
        """Required, but has a much smaller footprint than on main."""
        ...


class HasNamespace(Protocol):
    @property
    def Namespace(self) -> type[CompliantNamespace[Incomplete, Incomplete, Incomplete]]:
        """Required, but the protocol should not include types that are exported here."""
        ...


class HasExpr(Protocol):
    """This isn't particularly useful on its own.

    - `ExprDispatch` is the important part *from the outside*
    - Had to split out from `Expr` and `Scalar`, since `dispatch` can do any of:
        - `Self`   -> `Self`
        - `Expr`   -> `Scalar`
        - `Scalar` -> `Expr`
    """

    @property
    def Expr(self) -> type[CompliantExpr[Incomplete]]:
        """Required, but barely implemented for `polars` yet."""
        ...


class HasPlanEvaluator(Protocol[NativeLazy_co]):
    """Can translate a `ResolvedPlan` into `CompliantLazyFrame` operations."""

    @property
    def PlanEvaluator(self) -> type[ResolvedToCompliant[NativeLazy_co]] | Incomplete:
        """Required, but not implemented for `pyarrow` yet."""


class HasDataFrame(Protocol[NativeEager, NativeSeries_co]):
    @property
    def DataFrame(self) -> type[CompliantDataFrame[NativeEager, NativeSeries_co]]: ...


class HasSeries(Protocol[NativeSeries_co]):
    @property
    def Series(self) -> type[CompliantSeries[NativeSeries_co]]: ...


class HasPlanResolver(Protocol):
    """Overrides the default `LogicalPlan` -> `ResolvedPlan` translation."""

    @property
    def PlanResolver(self) -> type[LogicalToResolved] | None:
        """Optional, can default to `_plan.plans.conversion.Resolver`."""
        ...


class HasScalar(Protocol):
    """Has extra glue to mimic polars' scalar expressions.

    `CompliantScalar` implements many of the special cases, with guidance on the logic needed
    to fill in the gaps.

    Most other (non-accessor) expressions should behave identically to the `*Expr` version.
    """

    @property
    def Scalar(self) -> type[CompliantScalar[Incomplete]] | None:
        """Optional for *at-least* `polars`."""
        ...


@unstable
class LazyPackage(
    HasExpr,
    HasNamespace,
    HasLazyFrame[NativeLazy],
    HasPlanEvaluator[NativeLazy],
    Protocol[NativeLazy],
): ...


@unstable
class EagerPackage(
    HasExpr,
    HasNamespace,
    HasDataFrame[NativeEager, NativeSeries_co],
    HasSeries[NativeSeries_co],
    Protocol[NativeEager, NativeSeries_co],
): ...


@unstable
class HybridPackage(
    HasExpr,
    HasNamespace,
    HasDataFrame[NativeEager, NativeSeries_co],
    HasLazyFrame[NativeLazy],
    HasPlanEvaluator[NativeLazy],
    HasSeries[NativeSeries_co],
    Protocol[NativeLazy, NativeEager, NativeSeries_co],
): ...


@unstable
class FullPackage(
    HasPlanResolver, HasScalar, HybridPackage[NativeLazy, NativeEager, NativeSeries]
): ...


LazyPackageT = TypeVar("LazyPackageT", bound=LazyPackage[Any])
HybridPackageT = TypeVar("HybridPackageT", bound=HybridPackage[Any, Any, Any])


def try_lazy_package(package: LazyPackage[NativeLazy]) -> LazyPackage[NativeLazy]:
    return package


def try_hybrid_package(
    package: HybridPackage[NativeLazy, NativeEager, NativeSeries],
) -> HybridPackage[NativeLazy, NativeEager, NativeSeries]:
    return package


def try_hybrid_package_t(package: HybridPackageT) -> HybridPackageT:
    return package


def rewrap_as_plugin(package: LazyPackageT) -> LazyPackageT:
    """Figure out how the translation will work.

    - Make a split for eager vs lazy
    - Find any optional impls
    - Move into some structure that doesn't need checking again
    - The original classes should remain unchanged!

    """
    from_native_lazyframe = package.LazyFrame.from_native  # noqa: F841
    lazy = package.LazyFrame.from_narwhals  # noqa: F841
    collect = package.PlanEvaluator.collect  # noqa: F841
    sink_parquet = package.PlanEvaluator.sink_parquet  # noqa: F841
    Expr = package.Expr  # noqa: F841
    Namespace = package.Namespace  # noqa: F841
    LazyFrame = package.LazyFrame  # noqa: F841
    PlanEvaluator = package.PlanEvaluator  # noqa: F841
    return package


def try_arrow() -> LazyPackage[pa.Table]:
    import narwhals._plan.arrow

    return try_lazy_package(narwhals._plan.arrow)


# NOTE: This version reveals (<package-type>, (<native-types>, ...))
def try_arrow_1() -> tuple[
    HybridPackage[pa.Table, pa.Table, pa.ChunkedArray[Any]],
    tuple[
        type[CompliantLazyFrame[pa.Table]],
        type[CompliantDataFrame[pa.Table, pa.ChunkedArray[Any]]],
        type[CompliantSeries[pa.ChunkedArray[Any]]],
        type[CompliantExpr[Incomplete]],
    ],
]:
    import narwhals._plan.arrow

    package = try_hybrid_package(narwhals._plan.arrow)
    return package, (package.LazyFrame, package.DataFrame, package.Series, package.Expr)


# NOTE: This version reveals (<opaque>, (<compliant-types>, ...))
def try_arrow_2() -> tuple[
    ModuleType,
    tuple[type[ArrowLazyFrame], type[ArrowDataFrame], type[ArrowSeries], type[ArrowExpr]],
]:
    import narwhals._plan.arrow

    module = try_hybrid_package_t(narwhals._plan.arrow)
    return module, (module.LazyFrame, module.DataFrame, module.Series, module.Expr)


def try_polars_1() -> tuple[
    HybridPackage[pl.LazyFrame, pl.DataFrame, pl.Series],
    type[CompliantLazyFrame[pl.LazyFrame]],
    type[CompliantDataFrame[pl.DataFrame, pl.Series]],
    type[CompliantSeries[pl.Series]],
    type[CompliantExpr[Incomplete]],
]:
    import narwhals._plan.polars

    package = try_hybrid_package(narwhals._plan.polars)
    return package, package.LazyFrame, package.DataFrame, package.Series, package.Expr


def try_polars_2() -> tuple[
    ModuleType,
    type[PolarsLazyFrame],
    type[PolarsDataFrame],
    type[PolarsSeries],
    type[PolarsExpr],
]:
    import narwhals._plan.polars

    out = try_hybrid_package_t(narwhals._plan.polars)
    return out, out.LazyFrame, out.DataFrame, out.Series, out.Expr


def try_polars() -> LazyPackage[pl.LazyFrame]:
    import narwhals._plan.polars

    return try_lazy_package(narwhals._plan.polars)
