from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import NoReturn

from narwhals import dtypes
from narwhals._dask.expr import DaskExpr
from narwhals._expression_parsing import parse_into_exprs
from narwhals.dependencies import get_dask_dataframe
from narwhals.dependencies import get_pandas

if TYPE_CHECKING:
    from narwhals._dask.dataframe import DaskLazyFrame
    from narwhals._dask.typing import IntoDaskExpr


class DaskNamespace:
    Int64 = dtypes.Int64
    Int32 = dtypes.Int32
    Int16 = dtypes.Int16
    Int8 = dtypes.Int8
    UInt64 = dtypes.UInt64
    UInt32 = dtypes.UInt32
    UInt16 = dtypes.UInt16
    UInt8 = dtypes.UInt8
    Float64 = dtypes.Float64
    Float32 = dtypes.Float32
    Boolean = dtypes.Boolean
    Object = dtypes.Object
    Unknown = dtypes.Unknown
    Categorical = dtypes.Categorical
    Enum = dtypes.Enum
    String = dtypes.String
    Datetime = dtypes.Datetime
    Duration = dtypes.Duration
    Date = dtypes.Date

    def __init__(self, *, backend_version: tuple[int, ...]) -> None:
        self._backend_version = backend_version

    def all(self) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[Any]:
            return [
                df._native_dataframe.loc[:, column_name] for column_name in df.columns
            ]

        return DaskExpr(
            func,
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
            returns_scalar=False,
            backend_version=self._backend_version,
        )

    def col(self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names,
            backend_version=self._backend_version,
        )

    def min(self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names,
            backend_version=self._backend_version,
        ).min()

    def max(self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names,
            backend_version=self._backend_version,
        ).max()

    def mean(self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names,
            backend_version=self._backend_version,
        ).mean()

    def sum(self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names,
            backend_version=self._backend_version,
        ).sum()

    def len(self) -> DaskExpr:
        pd = get_pandas()
        dd = get_dask_dataframe()

        def func(df: DaskLazyFrame) -> list[Any]:
            if not df.columns:
                return [dd.from_pandas(pd.Series([0], name="len"))]
            return [
                df._native_dataframe.loc[:, df.columns[0]].size.to_series().rename("len")
            ]

        # coverage bug? this is definitely hit
        return DaskExpr(  # pragma: no cover
            func,
            depth=0,
            function_name="len",
            root_names=None,
            output_names=["len"],
            returns_scalar=True,
            backend_version=self._backend_version,
        )

    def all_horizontal(self, *exprs: IntoDaskExpr) -> DaskExpr:
        return reduce(lambda x, y: x & y, parse_into_exprs(*exprs, namespace=self))

    def any_horizontal(self, *exprs: IntoDaskExpr) -> DaskExpr:
        return reduce(lambda x, y: x | y, parse_into_exprs(*exprs, namespace=self))

    def _create_expr_from_series(self, _: Any) -> NoReturn:
        msg = "`_create_expr_from_series` for DaskNamespace exists only for compatibility"
        raise NotImplementedError(msg)

    def _create_compliant_series(self, _: Any) -> NoReturn:
        msg = "`_create_compliant_series` for DaskNamespace exists only for compatibility"
        raise NotImplementedError(msg)

    def _create_series_from_scalar(self, *_: Any) -> NoReturn:
        msg = (
            "`_create_series_from_scalar` for DaskNamespace exists only for compatibility"
        )
        raise NotImplementedError(msg)

    def _create_expr_from_callable(  # pragma: no cover
        self,
        func: Callable[[DaskLazyFrame], list[DaskExpr]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
    ) -> DaskExpr:
        msg = (
            "`_create_expr_from_callable` for DaskNamespace exists only for compatibility"
        )
        raise NotImplementedError(msg)
