from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, overload

import polars as pl

from narwhals._plan._version import into_version
from narwhals._plan.common import todo
from narwhals._plan.compliant.namespace import CompliantNamespace
from narwhals._plan.expressions.literal import is_literal_scalar
from narwhals._polars.utils import narwhals_to_native_dtype as _dtype_native
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan.polars.dataframe import PolarsDataFrame as DataFrame
    from narwhals._plan.polars.expr import PolarsExpr as Expr
    from narwhals._plan.polars.lazyframe import PolarsLazyFrame as LazyFrame
    from narwhals._plan.series import Series as NwSeries
    from narwhals.schema import Schema
    from narwhals.typing import IntoDType, NonNestedLiteral

Incomplete: TypeAlias = Any


@overload
def dtype_native(dtype: IntoDType, /, version: Version) -> pl.DataType: ...
@overload
def dtype_native(dtype: None, /, version: Version) -> None: ...
@overload
def dtype_native(dtype: IntoDType | None, /, version: Version) -> pl.DataType | None: ...
def dtype_native(dtype: IntoDType | None, /, version: Version) -> pl.DataType | None:
    """Convert a Narwhals `DType` to a `polars.DataType`, or passthrough `None`."""
    return dtype if dtype is None else _dtype_native(dtype, version)


class PolarsNamespace(CompliantNamespace[Incomplete, Incomplete, Incomplete]):
    __slots__ = ("_version",)
    _version: Version
    implementation: ClassVar = Implementation.POLARS

    def __init__(self, version: Version = Version.MAIN) -> None:
        self._version = version

    @property
    def version(self) -> Version:
        return self._version

    @property
    def _dataframe(self) -> type[DataFrame]:
        from narwhals._plan.polars.dataframe import PolarsDataFrame

        return PolarsDataFrame

    @property
    def _lazyframe(self) -> type[LazyFrame]:
        from narwhals._plan.polars.lazyframe import PolarsLazyFrame

        return PolarsLazyFrame

    @property
    def _expr(self) -> type[Expr]:
        from narwhals._plan.polars.expr import PolarsExpr

        return PolarsExpr

    _scalar = todo()  # type: ignore[assignment]
    _frame = todo()  # type: ignore[assignment]

    def read_csv(self, source: str, /, **kwds: Any) -> DataFrame:
        return self._dataframe.from_native(pl.read_csv(source, **kwds), self.version)

    def read_csv_schema(self, source: str, /, **kwds: Any) -> Schema:
        schema = pl.scan_csv(source, **kwds).collect_schema()
        return into_version(self.version).schema.from_polars(schema)

    def read_parquet(self, source: str, /, **kwds: Any) -> DataFrame:
        return self._dataframe.from_native(pl.read_parquet(source, **kwds), self.version)

    def read_parquet_schema(self, source: str, /, **kwds: Any) -> Schema:
        schema = pl.read_parquet_schema(source, **kwds)
        return into_version(self.version).schema.from_polars(schema)

    def scan_csv(self, source: str, /, **kwds: Any) -> LazyFrame:
        return self._lazyframe.from_native(pl.scan_csv(source, **kwds), self.version)

    def scan_parquet(self, source: str, /, **kwds: Any) -> LazyFrame:
        return self._lazyframe.from_native(pl.scan_parquet(source, **kwds), self.version)

    def col(self, node: ir.Column, frame: Any, name: str) -> Expr:
        return self._expr.from_native(pl.col(node.name), name, self.version)

    def len(self, node: ir.Len, frame: Any, name: str) -> Expr:
        return self._expr.from_native(pl.len(), name, self.version)

    def lit(
        self,
        node: ir.Literal[NonNestedLiteral] | ir.Literal[NwSeries[pl.Series]],
        frame: Any,
        name: str,
    ) -> Expr:
        version = self.version
        if not is_literal_scalar(node):
            series = node.unwrap().to_native()
            return self._expr.from_native(pl.lit(series), name, version)
        return self._expr.from_python(
            node.unwrap(), name, dtype=node.dtype, version=version
        )

    all_horizontal = todo()
    any_horizontal = todo()
    concat_str = todo()
    coalesce = todo()
    date_range = todo()
    int_range = todo()
    linear_space = todo()
    max_horizontal = todo()
    mean_horizontal = todo()
    min_horizontal = todo()
    sum_horizontal = todo()


PolarsNamespace()
