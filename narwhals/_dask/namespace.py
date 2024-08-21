from __future__ import annotations

from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import NoReturn

from narwhals import dtypes
from narwhals._dask.dataframe import DaskLazyFrame
from narwhals._dask.expr import DaskExpr
from narwhals._dask.selectors import DaskSelectorNamespace
from narwhals._expression_parsing import parse_into_exprs

if TYPE_CHECKING:
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

    @property
    def selectors(self) -> DaskSelectorNamespace:
        return DaskSelectorNamespace(backend_version=self._backend_version)

    def __init__(self, *, backend_version: tuple[int, ...]) -> None:
        self._backend_version = backend_version

    def all(self) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[Any]:
            return [df._native_frame.loc[:, column_name] for column_name in df.columns]

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

    def lit(self, value: Any, dtype: dtypes.DType | None) -> DaskExpr:
        # TODO @FBruzzesi: cast to dtype once `narwhals_to_native_dtype` is implemented.
        # It should be enough to add `.astype(narwhals_to_native_dtype(dtype))`
        return DaskExpr(
            lambda df: [df._native_frame.assign(lit=value).loc[:, "lit"]],
            depth=0,
            function_name="lit",
            root_names=None,
            output_names=["lit"],
            returns_scalar=False,
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
        import dask.dataframe as dd  # ignore-banned-import
        import pandas as pd  # ignore-banned-import

        def func(df: DaskLazyFrame) -> list[Any]:
            if not df.columns:
                return [
                    dd.from_pandas(
                        pd.Series([0], name="len"),
                        npartitions=df._native_frame.npartitions,
                    )
                ]
            return [df._native_frame.loc[:, df.columns[0]].size.to_series().rename("len")]

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

    def sum_horizontal(self, *exprs: IntoDaskExpr) -> DaskExpr:
        return reduce(lambda x, y: x + y, parse_into_exprs(*exprs, namespace=self))

    def concat(
        self,
        items: Iterable[DaskLazyFrame],
        *,
        how: str = "vertical",
    ) -> DaskLazyFrame:
        import dask.dataframe as dd  # ignore-banned-import

        if len(list(items)) == 0:
            msg = "No items to concatenate"
            raise ValueError(msg)
        native_frames = [i._native_frame for i in items]
        axis: int
        join: str
        if how == "vertical":
            if not all(
                tuple(i.columns) == tuple(native_frames[0].columns) for i in native_frames
            ):
                msg = "unable to vstack with non-matching columns"
                raise TypeError(msg)
            axis = 0
            join = "inner"
        elif how == "horizontal":
            all_column_names: list[str] = list(chain(*[i.columns for i in native_frames]))
            if len(all_column_names) != len(set(all_column_names)):
                duplicates = [
                    i for i in all_column_names if all_column_names.count(i) > 1
                ]
                msg = (
                    f"Columns with name(s): {', '.join(duplicates)} "
                    "have more than one occurrence"
                )
                raise TypeError(msg)
            axis = 1
            join = "outer"
        else:
            msg = (
                "Only valid options for concat are 'vertical' and 'horizontal' "
                f"({how} not recognised)"
            )
            raise NotImplementedError(msg)
        return DaskLazyFrame(
            dd.concat(native_frames, axis=axis, join=join),
            backend_version=self._backend_version,
        )

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
