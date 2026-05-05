from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

import polars as pl

from narwhals._plan._version import into_version
from narwhals._plan.common import todo
from narwhals._plan.compliant.namespace import CompliantNamespace
from narwhals._polars.utils import (
    narwhals_to_native_dtype as _dtype_native,
    native_to_narwhals_dtype as _dtype_from_native,
)
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    import datetime as dt

    from typing_extensions import TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions.ranges import IntRange
    from narwhals._plan.polars.dataframe import PolarsDataFrame as DataFrame
    from narwhals._plan.polars.expr import PolarsExpr as Expr
    from narwhals._plan.polars.lazyframe import (
        PolarsEvaluator as Evaluator,
        PolarsLazyFrame as LazyFrame,
    )
    from narwhals._plan.polars.series import PolarsSeries as Series
    from narwhals.dtypes import Date, DType, FloatType, IntegerType
    from narwhals.schema import Schema
    from narwhals.typing import ClosedInterval, IntoDType, PythonLiteral

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


class PolarsNamespace(CompliantNamespace["DataFrame", "Expr", "Expr"]):
    __slots__ = ()
    version: ClassVar[Version] = MAIN
    implementation: ClassVar = Implementation.POLARS

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
    def _scalar(self) -> type[Expr]:
        return self._expr

    @property
    def _series(self) -> type[Series]:
        from narwhals._plan.polars.series import PolarsSeries

        return PolarsSeries

    _frame = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]

    @property
    def _evaluator(self) -> type[Evaluator]:
        from narwhals._plan.polars.lazyframe import PolarsEvaluator

        return PolarsEvaluator

    def read_csv_schema(self, source: str, /, **kwds: Any) -> Schema:
        schema = pl.scan_csv(source, **kwds).collect_schema()
        return into_version(self.version).schema.from_polars(schema)

    def read_parquet_schema(self, source: str, /, **kwds: Any) -> Schema:
        schema = pl.read_parquet_schema(source, **kwds)
        return into_version(self.version).schema.from_polars(schema)

    def scan_csv(self, source: str, /, **kwds: Any) -> LazyFrame:
        return self._lazyframe.from_native(pl.scan_csv(source, **kwds))

    def scan_parquet(self, source: str, /, **kwds: Any) -> LazyFrame:
        return self._lazyframe.from_native(pl.scan_parquet(source, **kwds))

    def col(self, node: ir.Column, frame: Incomplete, name: str) -> Expr:
        return self._expr.from_native(pl.col(node.name), name)

    def len(self, node: ir.Len, frame: Incomplete, name: str) -> Expr:
        return self._expr.from_native(pl.len(), name)

    def lit(self, node: ir.Lit[PythonLiteral], frame: Incomplete, name: str) -> Expr:
        return self._expr.from_python(node.value, name, dtype=node.dtype)

    def lit_series(
        self, node: ir.LitSeries[pl.Series], frame: Incomplete, name: str
    ) -> Expr:
        return self._expr.from_native(pl.lit(node.native), name)

    def int_range(
        self, node: ir.RangeExpr[IntRange], frame: Incomplete, name: str
    ) -> Expr:
        func = node.function
        if fastpath := func.try_unwrap_literals(node):
            dtype = dtype_to_native_fast(func.dtype)
            native = pl.int_range(*fastpath, func.step, dtype=dtype)
            return self._expr.from_native(native, name)
        msg = f"TODO @dangotbanned: `{self.int_range.__qualname__}()` w/ non-`Lit` inputs, got \n{node.input[0]!r}\n{node.input[1]!r}"
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
        return self._series.from_native(native, name)

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
        return self._series.from_native(native, name)

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
        return self._series.from_native(native, name)

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
