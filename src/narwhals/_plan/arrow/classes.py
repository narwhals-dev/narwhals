from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Final

from narwhals._utils import Version

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._plan.arrow import DataFrame, Expr, LazyFrame, Scalar, Series, v1, v2


# TODO @dangotbanned: Convert into a descriptor for classmethod access
# - Should also navigate *down* a level at subclass-time
# - Overhead : `ArrowExprV1.__narwhals_classes__.v1.series` -> `ArrowSeriesV1`
# - Incorrect: `ArrowExprV1.__narwhals_classes__.series`    -> `ArrowSeries`
# - Ideal    : `ArrowExprV1.__narwhals_classes__.series`    -> `ArrowSeriesV1`
class ArrowClasses:
    """TODO @dangotbanned: Convert into a descriptor for classmethod access."""

    __slots__ = ()
    version: ClassVar[Version] = Version.MAIN

    @property
    def dataframe(self) -> type[DataFrame]:
        from narwhals._plan.arrow.dataframe import ArrowDataFrame

        return ArrowDataFrame

    @property
    def lazyframe(self) -> type[LazyFrame]:
        from narwhals._plan.arrow.lazyframe import ArrowLazyFrame

        return ArrowLazyFrame

    @property
    def expr(self) -> type[Expr]:
        from narwhals._plan.arrow.expr import ArrowExpr

        return ArrowExpr

    @property
    def scalar(self) -> type[Scalar]:
        from narwhals._plan.arrow.expr import ArrowScalar

        return ArrowScalar

    @property
    def series(self) -> type[Series]:
        from narwhals._plan.arrow.series import ArrowSeries

        return ArrowSeries

    @property
    def v1(self) -> ArrowClassesV1:
        return ArrowClassesV1()

    @property
    def v2(self) -> ArrowClassesV2:
        return ArrowClassesV2()


class ArrowClassesV1:
    __slots__ = ()
    version: ClassVar[Version] = Version.V1

    @property
    def dataframe(self) -> type[v1.DataFrame]:
        from narwhals._plan.arrow.v1 import DataFrame

        return DataFrame

    @property
    def lazyframe(self) -> type[v1.LazyFrame]:
        from narwhals._plan.arrow.v1 import LazyFrame

        return LazyFrame

    @property
    def expr(self) -> type[v1.Expr]:
        from narwhals._plan.arrow.v1 import Expr

        return Expr

    @property
    def scalar(self) -> type[v1.Scalar]:
        from narwhals._plan.arrow.v1 import Scalar

        return Scalar

    @property
    def series(self) -> type[v1.Series]:
        from narwhals._plan.arrow.v1 import Series

        return Series


class ArrowClassesV2:
    __slots__ = ()
    version: ClassVar[Version] = Version.V2

    @property
    def dataframe(self) -> type[v2.DataFrame]:
        from narwhals._plan.arrow.v2 import DataFrame

        return DataFrame

    @property
    def lazyframe(self) -> type[v2.LazyFrame]:
        from narwhals._plan.arrow.v2 import LazyFrame

        return LazyFrame

    @property
    def expr(self) -> type[v2.Expr]:
        from narwhals._plan.arrow.v2 import Expr

        return Expr

    @property
    def scalar(self) -> type[v2.Scalar]:
        from narwhals._plan.arrow.v2 import Scalar

        return Scalar

    @property
    def series(self) -> type[v2.Series]:
        from narwhals._plan.arrow.v2 import Series

        return Series


_CLASSES: Final[
    Mapping[Version, type[ArrowClasses | ArrowClassesV1 | ArrowClassesV2]]
] = {Version.MAIN: ArrowClasses, Version.V1: ArrowClassesV1, Version.V2: ArrowClassesV2}


class Versioned:
    """Enforce specifying `version=...` for each subclass.

    `Versioned` should be used to the left of any protocols in the bases.
    """

    __slots__ = ()
    version: ClassVar[Version]
    # TODO @dangotbanned: Fix this *properly* (likely by adding a generic base for each to inherit from)
    if TYPE_CHECKING:

        @property
        def __narwhals_classes__(self) -> ArrowClasses:
            return ArrowClasses()

    def __init_subclass__(cls, *, version: Version, **kwds: Any) -> None:
        super().__init_subclass__(**kwds)
        cls.version = version
        if not TYPE_CHECKING:
            cls.__narwhals_classes__ = _CLASSES[version]()
