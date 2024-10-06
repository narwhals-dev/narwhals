from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn

from narwhals._pandas_like.expr import PandasLikeExpr

if TYPE_CHECKING:
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.series import PandasLikeSeries
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes
    from narwhals.utils import Implementation


class PandasSelectorNamespace:
    def __init__(
        self,
        *,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> None:
        self._implementation = implementation
        self._backend_version = backend_version
        self._dtypes = dtypes

    def by_dtype(self, dtypes: list[DType | type[DType]]) -> PandasSelector:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            return [df[col] for col in df.columns if df.schema[col] in dtypes]

        return PandasSelector(
            func,
            depth=0,
            function_name="type_selector",
            root_names=None,
            output_names=None,
            implementation=self._implementation,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def numeric(self) -> PandasSelector:
        return self.by_dtype(
            [
                self._dtypes.Int64,
                self._dtypes.Int32,
                self._dtypes.Int16,
                self._dtypes.Int8,
                self._dtypes.UInt64,
                self._dtypes.UInt32,
                self._dtypes.UInt16,
                self._dtypes.UInt8,
                self._dtypes.Float64,
                self._dtypes.Float32,
            ],
        )

    def categorical(self) -> PandasSelector:
        return self.by_dtype([self._dtypes.Categorical])

    def string(self) -> PandasSelector:
        return self.by_dtype([self._dtypes.String])

    def boolean(self) -> PandasSelector:
        return self.by_dtype([self._dtypes.Boolean])

    def all(self) -> PandasSelector:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            return [df[col] for col in df.columns]

        return PandasSelector(
            func,
            depth=0,
            function_name="type_selector",
            root_names=None,
            output_names=None,
            implementation=self._implementation,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )


class PandasSelector(PandasLikeExpr):
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PandasSelector("
            f"depth={self._depth}, "
            f"function_name={self._function_name}, "
            f"root_names={self._root_names}, "
            f"output_names={self._output_names}"
        )

    def _to_expr(self) -> PandasLikeExpr:
        return PandasLikeExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            root_names=self._root_names,
            output_names=self._output_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def __sub__(self, other: PandasSelector | Any) -> PandasSelector | Any:
        if isinstance(other, PandasSelector):

            def call(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
                lhs = self._call(df)
                rhs = other._call(df)
                return [x for x in lhs if x.name not in {x.name for x in rhs}]

            return PandasSelector(
                call,
                depth=0,
                function_name="type_selector",
                root_names=None,
                output_names=None,
                implementation=self._implementation,
                backend_version=self._backend_version,
                dtypes=self._dtypes,
            )
        else:
            return self._to_expr() - other

    def __or__(self, other: PandasSelector | Any) -> PandasSelector | Any:
        if isinstance(other, PandasSelector):

            def call(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
                lhs = self._call(df)
                rhs = other._call(df)
                return [x for x in lhs if x.name not in {x.name for x in rhs}] + rhs

            return PandasSelector(
                call,
                depth=0,
                function_name="type_selector",
                root_names=None,
                output_names=None,
                implementation=self._implementation,
                backend_version=self._backend_version,
                dtypes=self._dtypes,
            )
        else:
            return self._to_expr() | other

    def __and__(self, other: PandasSelector | Any) -> PandasSelector | Any:
        if isinstance(other, PandasSelector):

            def call(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
                lhs = self._call(df)
                rhs = other._call(df)
                return [x for x in lhs if x.name in {x.name for x in rhs}]

            return PandasSelector(
                call,
                depth=0,
                function_name="type_selector",
                root_names=None,
                output_names=None,
                implementation=self._implementation,
                backend_version=self._backend_version,
                dtypes=self._dtypes,
            )
        else:
            return self._to_expr() & other

    def __invert__(self) -> PandasSelector:
        return (
            PandasSelectorNamespace(
                implementation=self._implementation,
                backend_version=self._backend_version,
                dtypes=self._dtypes,
            ).all()
            - self
        )

    def __rsub__(self, other: Any) -> NoReturn:
        raise NotImplementedError

    def __rand__(self, other: Any) -> NoReturn:
        raise NotImplementedError

    def __ror__(self, other: Any) -> NoReturn:
        raise NotImplementedError
