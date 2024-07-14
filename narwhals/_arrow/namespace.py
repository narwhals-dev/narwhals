from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable

from narwhals import dtypes
from narwhals._arrow.dataframe import ArrowDataFrame
from narwhals._arrow.expr import ArrowExpr
from narwhals._expression_parsing import parse_into_exprs
from narwhals.dependencies import get_pyarrow
from narwhals.utils import flatten

if TYPE_CHECKING:
    from typing import Callable

    from narwhals._arrow.expr import ArrowExpr
    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import IntoArrowExpr


class ArrowNamespace:
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

    def _create_expr_from_callable(
        self,
        func: Callable[[ArrowDataFrame], list[ArrowSeries]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
    ) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr

        return ArrowExpr(
            func,
            depth=depth,
            function_name=function_name,
            root_names=root_names,
            output_names=output_names,
            backend_version=self._backend_version,
        )

    def _create_expr_from_series(self, series: ArrowSeries) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr

        return ArrowExpr(
            lambda _df: [series],
            depth=0,
            function_name="series",
            root_names=None,
            output_names=None,
            backend_version=self._backend_version,
        )

    def _create_series_from_scalar(self, value: Any, series: ArrowSeries) -> ArrowSeries:
        from narwhals._arrow.series import ArrowSeries

        if self._backend_version < (13,) and hasattr(value, "as_py"):  # pragma: no cover
            value = value.as_py()
        return ArrowSeries._from_iterable(
            [value],
            name=series.name,
            backend_version=self._backend_version,
        )

    def _create_compliant_series(self, value: Any) -> ArrowSeries:
        from narwhals._arrow.series import ArrowSeries

        pa = get_pyarrow()
        return ArrowSeries(
            native_series=pa.chunked_array([value]),
            name="",
            backend_version=self._backend_version,
        )

    # --- not in spec ---
    def __init__(self, *, backend_version: tuple[int, ...]) -> None:
        self._backend_version = backend_version

    # --- selection ---
    def col(self, *column_names: str | Iterable[str]) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr

        return ArrowExpr.from_column_names(
            *flatten(column_names), backend_version=self._backend_version
        )

    def all(self) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr
        from narwhals._arrow.series import ArrowSeries

        return ArrowExpr(
            lambda df: [
                ArrowSeries(
                    df._native_dataframe[column_name],
                    name=column_name,
                    backend_version=df._backend_version,
                )
                for column_name in df.columns
            ],
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
            backend_version=self._backend_version,
        )

    def all_horizontal(self, *exprs: IntoArrowExpr) -> ArrowExpr:
        return reduce(
            lambda x, y: x & y,
            parse_into_exprs(*exprs, namespace=self),
        )

    def sum_horizontal(self, *exprs: IntoArrowExpr) -> ArrowExpr:
        return reduce(
            lambda x, y: x + y,
            parse_into_exprs(
                *exprs,
                namespace=self,
            ),
        )

    def concat(
        self,
        items: Iterable[ArrowDataFrame],
        *,
        how: str = "vertical",
    ) -> ArrowDataFrame:
        dfs: list[Any] = [item._native_dataframe for item in items]

        if len(dfs) == 0:
            msg = "No items to concatenate"
            raise ValueError(msg)

        pa = get_pyarrow()
        if how == "horizontal":
            names = [name for df in dfs for name in df.column_names]
            arrays = [a for df in dfs for a in df]

            if len(set(names)) < len(names):  # pragma: no cover
                msg = "Expected unique column names"
                raise ValueError(msg)

            return ArrowDataFrame(
                pa.Table.from_arrays(arrays, names=names),
                backend_version=self._backend_version,
            )
        if how == "vertical":
            cols = set(dfs[0].column_names)
            for df in dfs:
                cols_current = set(df.column_names)
                if cols_current != cols:
                    msg = "unable to vstack, column names don't match"
                    raise TypeError(msg)

            return ArrowDataFrame(
                pa.concat_tables(dfs).combine_chunks(),
                backend_version=self._backend_version,
            )
        raise NotImplementedError
