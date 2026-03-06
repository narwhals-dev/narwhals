from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

import polars as pl

from narwhals._plan._version import into_version
from narwhals._plan.common import todo
from narwhals._plan.compliant.namespace import CompliantNamespace
from narwhals._plan.compliant.translate import FromDict, FromIterable
from narwhals._plan.expressions.literal import is_literal_scalar
from narwhals._polars.utils import (
    narwhals_to_native_dtype as _dtype_native,
    native_to_narwhals_dtype as _dtype_from_native,
)
from narwhals._utils import Implementation, Version
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterable, Mapping

    from typing_extensions import TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions.ranges import IntRange
    from narwhals._plan.polars.dataframe import PolarsDataFrame as DataFrame
    from narwhals._plan.polars.expr import PolarsExpr as Expr
    from narwhals._plan.polars.lazyframe import PolarsLazyFrame as LazyFrame
    from narwhals._plan.polars.series import PolarsSeries as Series
    from narwhals._plan.polars.typing import CompliantDataFrame
    from narwhals._plan.series import Series as NwSeries
    from narwhals.dtypes import Date, DType, FloatType, IntegerType
    from narwhals.schema import Schema
    from narwhals.typing import (
        ClosedInterval,
        ConcatMethod,
        IntoDType,
        IntoSchema,
        NonNestedLiteral,
    )

Incomplete: TypeAlias = Any
MAIN = Version.MAIN
Int64 = MAIN.dtypes.Int64()


@overload
def dtype_to_native(dtype: IntoDType, /, version: Version) -> pl.DataType: ...
@overload
def dtype_to_native(dtype: None, /, version: Version) -> None: ...
@overload
def dtype_to_native(
    dtype: IntoDType | None, /, version: Version
) -> pl.DataType | None: ...
def dtype_to_native(dtype: IntoDType | None, /, version: Version) -> pl.DataType | None:
    """Convert a Narwhals `DType` to a `polars.DataType`, or passthrough `None`."""
    return dtype if dtype is None else _dtype_native(dtype, version)


def dtype_to_native_fast(dtype: IntegerType | FloatType | Date) -> Any:
    name = dtype.__class__.__name__
    if native := getattr(pl, name, None):
        return native
    # NOTE: Purely an error path for 128-bit ints
    return dtype_to_native(dtype, MAIN)


def dtype_from_native(dtype: pl.DataType, version: Version, /) -> DType:
    """Convert a `polars.DataType` to a Narwhals `DType`."""
    return _dtype_from_native(dtype, version)


# TODO @dangotbanned: Add version branching for False in `Explode.options`
# Default is backwards compatible
def explode_todo(
    *, empty_as_null: bool, keep_nulls: bool
) -> tuple[Literal[True], Literal[True]]:
    if not (empty_as_null and keep_nulls):
        msg = f"TODO @dangotbanned: Add version branching for False in `Explode.options`, got: {empty_as_null=}, {keep_nulls=}"
        raise NotImplementedError(msg)
    return True, True


class PolarsNamespace(
    FromIterable[pl.Series],
    FromDict[pl.DataFrame, pl.Series],
    CompliantNamespace[Incomplete, "Expr", Incomplete],
):
    __slots__ = ("_version",)
    _version: Version
    implementation: ClassVar = Implementation.POLARS

    def __init__(self, version: Version = MAIN) -> None:
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

    @property
    def _series(self) -> type[Series]:
        from narwhals._plan.polars.series import PolarsSeries

        return PolarsSeries

    _scalar = todo()  # type: ignore[assignment]
    _frame = todo()  # type: ignore[assignment]

    def from_dict(
        self,
        data: Mapping[str, Any],
        /,
        *,
        schema: IntoSchema | None = None,
        version: Version = MAIN,
    ) -> DataFrame:
        return self._dataframe.from_dict(data, schema=schema, version=version)

    def from_iterable(
        self,
        data: Iterable[Any],
        *,
        name: str = "",
        dtype: IntoDType | None = None,
        version: Version = MAIN,
    ) -> Series:
        return self._series.from_iterable(data, name=name, dtype=dtype, version=version)

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

    def col(self, node: ir.Column, frame: Incomplete, name: str) -> Expr:
        return self._expr.from_native(pl.col(node.name), name, self.version)

    def len(self, node: ir.Len, frame: Incomplete, name: str) -> Expr:
        return self._expr.from_native(pl.len(), name, self.version)

    def lit(
        self,
        node: ir.Literal[NonNestedLiteral] | ir.Literal[NwSeries[pl.Series]],
        frame: Incomplete,
        name: str,
    ) -> Expr:
        version = self.version
        if not is_literal_scalar(node):
            series = node.unwrap().to_native()
            return self._expr.from_native(pl.lit(series), name, version)
        return self._expr.from_python(
            node.unwrap(), name, dtype=node.dtype, version=version
        )

    def int_range(
        self, node: ir.RangeExpr[IntRange], frame: Incomplete, name: str
    ) -> Expr:
        start, end = node.function.unwrap_input(node)
        if is_literal_scalar(start) and is_literal_scalar(end):
            start_, end_ = start.unwrap(), end.unwrap()
            if isinstance(start_, int) and isinstance(end_, int):
                dtype = dtype_to_native_fast(node.function.dtype)
                native = pl.int_range(start_, end_, node.function.step, dtype=dtype)
                return self._expr.from_native(native, name, self.version)
            msg = f"All inputs for `{node.function}()` must resolve to int, but got \n{start_!r}\n{end_!r}"
            raise InvalidOperationError(msg)
        msg = f"TODO @dangotbanned: `{self.int_range.__qualname__}()` w/ non-`ScalarLiteral` inputs, got \n{start!r}\n{end!r}"
        raise NotImplementedError(msg)

    def int_range_eager(
        self,
        start: int,
        end: int,
        step: int = 1,
        *,
        dtype: IntegerType = Int64,
        name: str = "literal",
    ) -> Series:
        dtype_ = dtype_to_native_fast(dtype)
        native = pl.int_range(start, end, step, dtype=dtype_, eager=True)
        return self._series.from_native(native, name, version=self.version)

    def date_range_eager(
        self,
        start: dt.date,
        end: dt.date,
        interval: int = 1,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> Series:
        native = pl.date_range(start, end, f"{interval}d", closed=closed, eager=True)
        return self._series.from_native(native, name, version=self.version)

    def linear_space_eager(
        self,
        start: float,
        end: float,
        num_samples: int,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> Series:
        native = pl.linear_space(start, end, num_samples, closed=closed, eager=True)
        return self._series.from_native(native, name, version=self.version)

    def concat_df(
        self, dfs: Iterable[CompliantDataFrame], /, how: ConcatMethod = "vertical"
    ) -> DataFrame:
        result = pl.concat((df.native for df in dfs), how=how)
        return self._dataframe.from_native(result, version=self.version)

    concat_df_vertical = concat_df

    def concat_df_horizontal(self, dfs: Iterable[CompliantDataFrame], /) -> DataFrame:
        return self.concat_df(dfs, how="horizontal")

    def concat_df_diagonal(self, dfs: Iterable[CompliantDataFrame], /) -> DataFrame:
        return self.concat_df(dfs, how="diagonal")

    all_horizontal = todo()
    any_horizontal = todo()
    concat_str = todo()
    coalesce = todo()
    date_range = todo()
    linear_space = todo()
    max_horizontal = todo()
    mean_horizontal = todo()
    min_horizontal = todo()
    sum_horizontal = todo()


PolarsNamespace()
