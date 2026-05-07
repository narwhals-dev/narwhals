from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from narwhals._plan._version import into_version
from narwhals._plan.arrow import io
from narwhals._plan.compliant.namespace import EagerNamespace
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from narwhals._plan.arrow.dataframe import ArrowDataFrame as Frame
    from narwhals._plan.arrow.expr import ArrowExpr as Expr, ArrowScalar as Scalar
    from narwhals._plan.arrow.lazyframe import ArrowLazyFrame as LazyFrame
    from narwhals._plan.arrow.series import ArrowSeries as Series
    from narwhals._plan.arrow.typing import IOSource
    from narwhals.schema import Schema
    from narwhals.typing import FileSource


class ArrowNamespace(EagerNamespace["Frame", "Series", "Expr", "Scalar"]):
    __slots__ = ()
    implementation = Implementation.PYARROW
    version: ClassVar[Version] = Version.MAIN

    @property
    def _expr(self) -> type[Expr]:
        from narwhals._plan.arrow.expr import ArrowExpr

        return ArrowExpr

    @property
    def _scalar(self) -> type[Scalar]:
        from narwhals._plan.arrow.expr import ArrowScalar

        return ArrowScalar

    @property
    def _series(self) -> type[Series]:
        from narwhals._plan.arrow.series import ArrowSeries

        return ArrowSeries

    @property
    def _dataframe(self) -> type[Frame]:
        from narwhals._plan.arrow.dataframe import ArrowDataFrame

        return ArrowDataFrame

    @property
    def _lazyframe(self) -> type[LazyFrame]:
        from narwhals._plan.arrow.lazyframe import ArrowLazyFrame

        return ArrowLazyFrame

    def read_csv_schema(self, source: FileSource, /, **kwds: Any) -> Schema:
        return into_version(self).schema.from_arrow(io.read_csv_schema(source, **kwds))

    def read_parquet_schema(self, source: IOSource, /, **kwds: Any) -> Schema:
        native = io.read_parquet_schema(source, **kwds)
        return into_version(self).schema.from_arrow(native)
