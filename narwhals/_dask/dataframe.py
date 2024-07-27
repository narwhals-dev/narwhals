from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._dask.utils import parse_exprs_and_named_exprs
from narwhals.dependencies import get_dask_dataframe

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._dask.expr import DaskExpr
    from narwhals._dask.namespace import DaskNamespace


class DaskLazyFrame:
    def __init__(
        self, native_dataframe: Any, *, backend_version: tuple[int, ...]
    ) -> None:
        self._native_dataframe = native_dataframe
        self._backend_version = backend_version

    def __native_namespace__(self) -> Any:
        return get_dask_dataframe()

    def __narwhals_namespace__(self) -> DaskNamespace:
        from narwhals._dask.namespace import DaskNamespace

        return DaskNamespace(backend_version=self._backend_version)

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def _from_native_dataframe(self, df: Any) -> Self:
        return self.__class__(df, backend_version=self._backend_version)

    def with_columns(self, *exprs: DaskExpr, **named_exprs: DaskExpr) -> Self:
        df = self._native_dataframe
        new_series = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)
        df = df.assign(**new_series)
        return self._from_native_dataframe(df)
