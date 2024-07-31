from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._dask.utils import parse_exprs_and_named_exprs
from narwhals.dependencies import get_dask_dataframe
from narwhals.dependencies import get_pandas
from narwhals.utils import Implementation
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._dask.expr import DaskExpr
    from narwhals._dask.namespace import DaskNamespace
    from narwhals._dask.typing import IntoDaskExpr


class DaskLazyFrame:
    def __init__(
        self, native_dataframe: Any, *, backend_version: tuple[int, ...]
    ) -> None:
        self._native_dataframe = native_dataframe
        self._backend_version = backend_version

    def __native_namespace__(self) -> Any:  # pragma: no cover
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

    def collect(self) -> Any:
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        result = self._native_dataframe.compute()
        return PandasLikeDataFrame(
            result,
            implementation=Implementation.PANDAS,
            backend_version=parse_version(get_pandas().__version__),
        )

    @property
    def columns(self) -> list[str]:
        return self._native_dataframe.columns.tolist()  # type: ignore[no-any-return]

    def filter(
        self,
        *predicates: DaskExpr,
    ) -> Self:
        from narwhals._dask.namespace import DaskNamespace

        plx = DaskNamespace(backend_version=self._backend_version)
        expr = plx.all_horizontal(*predicates)
        # Safety: all_horizontal's expression only returns a single column.
        mask = expr._call(self)[0]
        return self._from_native_dataframe(self._native_dataframe.loc[mask])

    def lazy(self) -> Self:
        return self

    def select(
        self: Self,
        *exprs: IntoDaskExpr,
        **named_exprs: IntoDaskExpr,
    ) -> Self:
        dd = get_dask_dataframe()

        if exprs and all(isinstance(x, str) for x in exprs) and not named_exprs:
            # This is a simple slice => fastpath!
            return self._from_native_dataframe(self._native_dataframe.loc[:, exprs])

        new_series = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)
        if not new_series:
            # return empty dataframe, like Polars does
            pd = get_pandas()
            return self._from_native_dataframe(dd.from_pandas(pd.DataFrame()))
        df = dd.concat([val.rename(name) for name, val in new_series.items()], axis=1)
        return self._from_native_dataframe(df)

    def drop_nulls(self) -> Self:
        return self._from_native_dataframe(self._native_dataframe.dropna())
