from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import cast
from typing import overload

import polars as pl

from narwhals._polars.expr import PolarsExpr
from narwhals._polars.series import PolarsSeries
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import narwhals_to_native_dtype
from narwhals.dependencies import is_numpy_array_2d
from narwhals.dtypes import DType
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from datetime import timezone

    from typing_extensions import Self
    from typing_extensions import TypeAlias

    from narwhals._compliant import CompliantSelectorNamespace
    from narwhals._compliant import CompliantWhen
    from narwhals._polars.dataframe import Method
    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame
    from narwhals._polars.typing import FrameT
    from narwhals.schema import Schema
    from narwhals.typing import Into1DArray
    from narwhals.typing import TimeUnit
    from narwhals.typing import _2DArray
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

    Incomplete: TypeAlias = Any


class PolarsNamespace:
    all: Method[PolarsExpr]
    col: Method[PolarsExpr]
    exclude: Method[PolarsExpr]
    all_horizontal: Method[PolarsExpr]
    any_horizontal: Method[PolarsExpr]
    sum_horizontal: Method[PolarsExpr]
    min_horizontal: Method[PolarsExpr]
    max_horizontal: Method[PolarsExpr]
    # NOTE: `PolarsSeries`, `PolarsExpr` still have gaps
    when: Method[CompliantWhen[PolarsDataFrame, Incomplete, Incomplete]]

    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._implementation = Implementation.POLARS
        self._version = version

    def __getattr__(self: Self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._expr(
                getattr(pl, attr)(*args, **kwargs),
                version=self._version,
                backend_version=self._backend_version,
            )

        return func

    @property
    def _dataframe(self) -> type[PolarsDataFrame]:
        from narwhals._polars.dataframe import PolarsDataFrame

        return PolarsDataFrame

    @property
    def _expr(self) -> type[PolarsExpr]:
        return PolarsExpr

    @property
    def _series(self) -> type[PolarsSeries]:
        return PolarsSeries

    @overload
    def from_numpy(
        self,
        data: Into1DArray,
        /,
        schema: None = ...,
    ) -> PolarsSeries: ...

    @overload
    def from_numpy(
        self,
        data: _2DArray,
        /,
        schema: Mapping[str, DType] | Schema | Sequence[str] | None,
    ) -> PolarsDataFrame: ...

    def from_numpy(
        self,
        data: Into1DArray | _2DArray,
        /,
        schema: Mapping[str, DType] | Schema | Sequence[str] | None = None,
    ) -> PolarsDataFrame | PolarsSeries:
        if is_numpy_array_2d(data):
            return self._dataframe.from_numpy(data, schema=schema, context=self)
        return self._series.from_numpy(data, context=self)  # pragma: no cover

    def nth(self: Self, *indices: int) -> PolarsExpr:
        if self._backend_version < (1, 0, 0):
            msg = "`nth` is only supported for Polars>=1.0.0. Please use `col` for columns selection instead."
            raise AttributeError(msg)
        return self._expr(
            pl.nth(*indices), version=self._version, backend_version=self._backend_version
        )

    def len(self: Self) -> PolarsExpr:
        if self._backend_version < (0, 20, 5):
            return self._expr(
                pl.count().alias("len"),
                version=self._version,
                backend_version=self._backend_version,
            )
        return self._expr(
            pl.len(), version=self._version, backend_version=self._backend_version
        )

    def concat(
        self: Self,
        items: Iterable[FrameT],
        *,
        how: Literal["vertical", "horizontal", "diagonal"],
    ) -> PolarsDataFrame | PolarsLazyFrame:
        from narwhals._polars.dataframe import PolarsLazyFrame

        result = pl.concat((item.native for item in items), how=how)
        if isinstance(result, pl.DataFrame):
            return self._dataframe(
                result, backend_version=self._backend_version, version=self._version
            )
        return PolarsLazyFrame(
            result, backend_version=self._backend_version, version=self._version
        )

    def lit(self: Self, value: Any, dtype: DType | type[DType] | None) -> PolarsExpr:
        if dtype is not None:
            return self._expr(
                pl.lit(
                    value,
                    dtype=narwhals_to_native_dtype(
                        dtype, self._version, self._backend_version
                    ),
                ),
                version=self._version,
                backend_version=self._backend_version,
            )
        return self._expr(
            pl.lit(value), version=self._version, backend_version=self._backend_version
        )

    def mean_horizontal(self: Self, *exprs: PolarsExpr) -> PolarsExpr:
        if self._backend_version < (0, 20, 8):
            return self._expr(
                pl.sum_horizontal(e._native_expr for e in exprs)
                / pl.sum_horizontal(1 - e.is_null()._native_expr for e in exprs),
                version=self._version,
                backend_version=self._backend_version,
            )

        return self._expr(
            pl.mean_horizontal(e._native_expr for e in exprs),
            version=self._version,
            backend_version=self._backend_version,
        )

    def concat_str(
        self: Self,
        *exprs: PolarsExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> PolarsExpr:
        pl_exprs: list[pl.Expr] = [expr._native_expr for expr in exprs]

        if self._backend_version < (0, 20, 6):
            null_mask = [expr.is_null() for expr in pl_exprs]
            sep = pl.lit(separator)

            if not ignore_nulls:
                null_mask_result = pl.any_horizontal(*null_mask)
                output_expr = pl.reduce(
                    lambda x, y: x.cast(pl.String()) + sep + y.cast(pl.String()),  # type: ignore[arg-type,return-value]
                    pl_exprs,
                )
                result = pl.when(~null_mask_result).then(output_expr)
            else:
                init_value, *values = [
                    pl.when(nm).then(pl.lit("")).otherwise(expr.cast(pl.String()))
                    for expr, nm in zip(pl_exprs, null_mask)
                ]
                separators = [
                    pl.when(~nm).then(sep).otherwise(pl.lit("")) for nm in null_mask[:-1]
                ]

                result = pl.fold(  # type: ignore[assignment]
                    acc=init_value,
                    function=operator.add,
                    exprs=[s + v for s, v in zip(separators, values)],
                )

            return self._expr(
                result, version=self._version, backend_version=self._backend_version
            )

        return self._expr(
            pl.concat_str(
                pl_exprs,
                separator=separator,
                ignore_nulls=ignore_nulls,
            ),
            version=self._version,
            backend_version=self._backend_version,
        )

    # NOTE: Implementation is too different to annotate correctly (vs other `*SelectorNamespace`)
    # 1. Others have lots of private stuff for code reuse
    #    i. None of that is useful here
    # 2. We don't have a `PolarsSelector` abstraction, and just use `PolarsExpr`
    # 3. `PolarsExpr` still has it's own gaps in the spec
    @property
    def selectors(self: Self) -> CompliantSelectorNamespace[Any, Any]:
        return cast("CompliantSelectorNamespace[Any, Any]", PolarsSelectorNamespace(self))


class PolarsSelectorNamespace:
    def __init__(self: Self, context: _FullContext, /) -> None:
        self._implementation = context._implementation
        self._backend_version = context._backend_version
        self._version = context._version

    def by_dtype(self: Self, dtypes: Iterable[DType]) -> PolarsExpr:
        native_dtypes = [
            narwhals_to_native_dtype(
                dtype, self._version, self._backend_version
            ).__class__
            if isinstance(dtype, type) and issubclass(dtype, DType)
            else narwhals_to_native_dtype(dtype, self._version, self._backend_version)
            for dtype in dtypes
        ]
        return PolarsExpr(
            pl.selectors.by_dtype(native_dtypes),
            version=self._version,
            backend_version=self._backend_version,
        )

    def matches(self: Self, pattern: str) -> PolarsExpr:
        return PolarsExpr(
            pl.selectors.matches(pattern=pattern),
            version=self._version,
            backend_version=self._backend_version,
        )

    def numeric(self: Self) -> PolarsExpr:
        return PolarsExpr(
            pl.selectors.numeric(),
            version=self._version,
            backend_version=self._backend_version,
        )

    def boolean(self: Self) -> PolarsExpr:
        return PolarsExpr(
            pl.selectors.boolean(),
            version=self._version,
            backend_version=self._backend_version,
        )

    def string(self: Self) -> PolarsExpr:
        return PolarsExpr(
            pl.selectors.string(),
            version=self._version,
            backend_version=self._backend_version,
        )

    def categorical(self: Self) -> PolarsExpr:
        return PolarsExpr(
            pl.selectors.categorical(),
            version=self._version,
            backend_version=self._backend_version,
        )

    def all(self: Self) -> PolarsExpr:
        return PolarsExpr(
            pl.selectors.all(),
            version=self._version,
            backend_version=self._backend_version,
        )

    def datetime(
        self: Self,
        time_unit: TimeUnit | Iterable[TimeUnit] | None,
        time_zone: str | timezone | Iterable[str | timezone | None] | None,
    ) -> PolarsExpr:
        return PolarsExpr(
            pl.selectors.datetime(time_unit=time_unit, time_zone=time_zone),  # type: ignore[arg-type]
            version=self._version,
            backend_version=self._backend_version,
        )
