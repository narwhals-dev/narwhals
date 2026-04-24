"""Defines the expected shape of an `__init__.py` for a backend.

Rather than using a `Namespace` and accessing classes via properties with inline imports:
why not just use the import system?

TypeVar(s) would have an outer scope to avoid cycles


Could be something we resolve *once*, and then treat everything like a plugin?

We'd take these exports and transform them into something that describes capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.compliant.typing import (
    Native as NativeLazy,
    Native_co as NativeLazy_co,
)
from narwhals._plan.typing import (
    NativeDataFrameT_co as NativeEager_co,
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
    from narwhals._plan.arrow.v1 import (
        ArrowDataFrameV1,
        ArrowExprV1,
        ArrowLazyFrameV1,
        ArrowScalarV1,
        ArrowSeriesV1,
    )
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
    from narwhals._plan.polars.v1 import (
        PolarsDataFrameV1,
        PolarsEvaluatorV1,
        PolarsExprV1,
        PolarsLazyFrameV1,
        PolarsSeriesV1,
    )

Incomplete: TypeAlias = Any


class HasLazyFrame(Protocol[NativeLazy_co]):
    @property
    def LazyFrame(self) -> type[CompliantLazyFrame[NativeLazy_co]]:
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


class HasPlanEvaluator(Protocol[NativeLazy]):
    """Can translate a `ResolvedPlan` into `CompliantLazyFrame` operations."""

    @property
    def PlanEvaluator(self) -> type[ResolvedToCompliant[NativeLazy]]:
        """Required, but not implemented for `pyarrow` yet."""
        ...


class MaybeHasPlanEvaluator(Protocol[NativeLazy]):
    @property
    def PlanEvaluator(self) -> type[ResolvedToCompliant[NativeLazy]] | None: ...


class HasDataFrame(Protocol[NativeEager_co, NativeSeries_co]):
    @property
    def DataFrame(self) -> type[CompliantDataFrame[NativeEager_co, NativeSeries_co]]: ...


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
    def Scalar(
        self,
    ) -> type[CompliantScalar[Incomplete] | CompliantExpr[Incomplete]] | None:
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
    HasDataFrame[NativeEager_co, NativeSeries_co],
    HasSeries[NativeSeries_co],
    Protocol[NativeEager_co, NativeSeries_co],
): ...


@unstable
class HybridPackage(
    HasExpr,
    HasNamespace,
    HasDataFrame[NativeEager_co, NativeSeries_co],
    HasLazyFrame[NativeLazy],
    MaybeHasPlanEvaluator[NativeLazy],
    HasSeries[NativeSeries_co],
    Protocol[NativeLazy, NativeEager_co, NativeSeries_co],
): ...


@unstable
class FullPackage(
    HasPlanResolver,
    HasScalar,
    HybridPackage[NativeLazy, NativeEager_co, NativeSeries_co],
    Protocol[NativeLazy, NativeEager_co, NativeSeries_co],
): ...


LazyPackageT = TypeVar("LazyPackageT", bound=LazyPackage[Any])
HybridPackageT = TypeVar("HybridPackageT", bound=HybridPackage[Any, Any, Any])
FullPackageT_co = TypeVar(
    "FullPackageT_co", bound=FullPackage[Any, Any, Any], covariant=True
)

AnyPackageT_co = TypeVar(
    "AnyPackageT_co",
    bound="LazyPackage[Any] | HybridPackage[Any, Any, Any] | FullPackage[Any, Any, Any]",
    covariant=True,
)


class HasV1(Protocol[AnyPackageT_co]):
    """A new scope for a new set of types."""

    @property
    def v1(self) -> AnyPackageT_co: ...


def try_lazy_package(package: LazyPackage[NativeLazy]) -> LazyPackage[NativeLazy]:
    return package


def try_hybrid_package(
    package: HybridPackage[NativeLazy, NativeEager_co, NativeSeries_co],
) -> HybridPackage[NativeLazy, NativeEager_co, NativeSeries_co]:
    return package


def try_hybrid_package_t(package: HybridPackageT) -> HybridPackageT:
    return package


def try_v1_package(
    package: HasV1[HybridPackage[NativeLazy, NativeEager_co, NativeSeries_co]],
) -> HasV1[HybridPackage[NativeLazy, NativeEager_co, NativeSeries_co]]:
    return package


def try_v1_package_t(package: HasV1[FullPackageT_co]) -> HasV1[FullPackageT_co]:
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

    return try_lazy_package(narwhals._plan.arrow)  # pyright: ignore[reportArgumentType]


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


def try_arrow_v1_1() -> tuple[
    HybridPackage[pa.Table, pa.Table, pa.ChunkedArray[Any]],
    tuple[
        type[CompliantLazyFrame[pa.Table]],
        type[CompliantDataFrame[pa.Table, pa.ChunkedArray[Any]]],
        type[CompliantSeries[pa.ChunkedArray[Any]]],
        type[CompliantExpr[Incomplete]],
    ],
]:
    import narwhals._plan.arrow

    outer_package = try_v1_package(narwhals._plan.arrow)
    v1 = outer_package.v1
    return v1, (v1.LazyFrame, v1.DataFrame, v1.Series, v1.Expr)


def try_arrow_v1_2() -> tuple[
    ModuleType,
    tuple[
        type[ArrowLazyFrameV1],
        type[ArrowDataFrameV1],
        type[ArrowSeriesV1],
        type[ArrowExprV1],
        type[ArrowScalarV1],
    ],
]:
    import narwhals._plan.arrow

    outer = try_v1_package_t(narwhals._plan.arrow)
    inner = outer.v1
    return inner, (
        inner.LazyFrame,
        inner.DataFrame,
        inner.Series,
        inner.Expr,
        inner.Scalar,
    )


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
    type[PolarsExpr],
]:
    import narwhals._plan.polars

    out = try_hybrid_package_t(narwhals._plan.polars)
    return out, out.LazyFrame, out.DataFrame, out.Series, out.Expr, out.Scalar


def try_polars() -> LazyPackage[pl.LazyFrame]:
    import narwhals._plan.polars

    return try_lazy_package(narwhals._plan.polars)


def try_polars_v1_1() -> tuple[
    HybridPackage[pl.LazyFrame, pl.DataFrame, pl.Series],
    tuple[
        type[CompliantLazyFrame[pl.LazyFrame]],
        type[CompliantDataFrame[pl.DataFrame, pl.Series]],
        type[CompliantSeries[pl.Series]],
        type[CompliantExpr[Any]],
        type[ResolvedToCompliant[pl.LazyFrame]],
    ],
]:
    import narwhals._plan.polars

    outer_package = try_v1_package(narwhals._plan.polars)
    v1 = outer_package.v1

    if v1.PlanEvaluator is None:
        raise NotImplementedError
    return v1, (v1.LazyFrame, v1.DataFrame, v1.Series, v1.Expr, v1.PlanEvaluator)


def try_polars_v1_2() -> tuple[
    ModuleType,
    tuple[
        type[PolarsLazyFrameV1],
        type[PolarsDataFrameV1],
        type[PolarsSeriesV1],
        type[PolarsExprV1],
        type[PolarsEvaluatorV1],
        type[PolarsExprV1],
    ],
]:
    import narwhals._plan.polars

    outer = try_v1_package_t(narwhals._plan.polars)
    v1 = outer.v1
    return v1, (
        v1.LazyFrame,
        v1.DataFrame,
        v1.Series,
        v1.Expr,
        v1.PlanEvaluator,
        v1.Scalar,
    )
