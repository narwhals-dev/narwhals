from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import NoReturn

from narwhals import dtypes
from narwhals._dask.expr import DaskExpr
from narwhals.utils import flatten

if TYPE_CHECKING:
    from narwhals._dask.dataframe import DaskLazyFrame

if TYPE_CHECKING:
    from typing import Callable

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

    def col(self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names,
            backend_version=self._backend_version,
        )

    def all_horizontal(self, *exprs: IntoDaskExpr) -> DaskExpr:
        # coverage shows 55->exit as uncovered?
        return reduce(lambda x, y: x & y, flatten(exprs))  # type: ignore[no-any-return]  # pragma: no cover

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
        return DaskExpr(
            call=func,
            depth=depth,
            function_name=function_name,
            root_names=root_names,
            output_names=output_names,
            backend_version=self._backend_version,
        )
