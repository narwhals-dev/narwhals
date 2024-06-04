from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn

from narwhals import dtypes
from narwhals._pandas_like.expr import PandasExpr

if TYPE_CHECKING:
    from narwhals._pandas_like.dataframe import PandasDataFrame
    from narwhals._pandas_like.series import PandasSeries
    from narwhals.dtypes import DType


class PandasSelectorNamespace:
    def __init__(self, implementation: str) -> None:
        self._implementation = implementation

    def by_dtype(self, dtypes: list[DType | type[DType]]) -> PandasSelector:
        def func(df: PandasDataFrame) -> list[PandasSeries]:
            return [df[col] for col in df.columns if df.schema[col] in dtypes]

        return PandasSelector(
            func,
            depth=0,
            function_name="type_selector",
            root_names=None,
            output_names=None,
            implementation=self._implementation,
        )

    def numeric(self) -> PandasSelector:
        return self.by_dtype(
            [
                dtypes.Int64,
                dtypes.Int32,
                dtypes.Int16,
                dtypes.Int8,
                dtypes.UInt64,
                dtypes.UInt32,
                dtypes.UInt16,
                dtypes.UInt8,
                dtypes.Float64,
                dtypes.Float32,
            ],
        )

    def categorical(self) -> PandasSelector:
        return self.by_dtype([dtypes.Categorical])

    def string(self) -> PandasSelector:
        return self.by_dtype([dtypes.String])

    def boolean(self) -> PandasSelector:
        return self.by_dtype([dtypes.Boolean])

    def all(self) -> PandasSelector:
        def func(df: PandasDataFrame) -> list[PandasSeries]:
            return [df[col] for col in df.columns]

        return PandasSelector(
            func,
            depth=0,
            function_name="type_selector",
            root_names=None,
            output_names=None,
            implementation=self._implementation,
        )


class PandasSelector(PandasExpr):
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PandasSelector("
            f"depth={self._depth}, "
            f"function_name={self._function_name}, "
            f"root_names={self._root_names}, "
            f"output_names={self._output_names}"
        )

    def _to_expr(self) -> PandasExpr:
        return PandasExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            root_names=self._root_names,
            output_names=self._output_names,
            implementation=self._implementation,
        )

    def __sub__(self, other: PandasSelector | Any) -> PandasSelector | Any:
        if isinstance(other, PandasSelector):

            def call(df: PandasDataFrame) -> list[PandasSeries]:
                lhs = self._call(df)
                rhs = other._call(df)
                return [x for x in lhs if x.name not in [x.name for x in rhs]]

            return PandasSelector(
                call,
                depth=0,
                function_name="type_selector",
                root_names=None,
                output_names=None,
                implementation=self._implementation,
            )
        else:
            return self._to_expr() - other

    def __or__(self, other: PandasSelector | Any) -> PandasSelector | Any:
        if isinstance(other, PandasSelector):

            def call(df: PandasDataFrame) -> list[PandasSeries]:
                lhs = self._call(df)
                rhs = other._call(df)
                return [x for x in lhs if x.name not in [x.name for x in rhs]] + rhs

            return PandasSelector(
                call,
                depth=0,
                function_name="type_selector",
                root_names=None,
                output_names=None,
                implementation=self._implementation,
            )
        else:
            return self._to_expr() | other

    def __and__(self, other: PandasSelector | Any) -> PandasSelector | Any:
        if isinstance(other, PandasSelector):

            def call(df: PandasDataFrame) -> list[PandasSeries]:
                lhs = self._call(df)
                rhs = other._call(df)
                return [x for x in lhs if x.name in [x.name for x in rhs]]

            return PandasSelector(
                call,
                depth=0,
                function_name="type_selector",
                root_names=None,
                output_names=None,
                implementation=self._implementation,
            )
        else:
            return self._to_expr() & other

    def __invert__(self) -> PandasSelector:
        return PandasSelectorNamespace(self._implementation).all() - self

    def __rsub__(self, other: Any) -> NoReturn:
        raise NotImplementedError

    def __rand__(self, other: Any) -> NoReturn:
        raise NotImplementedError

    def __ror__(self, other: Any) -> NoReturn:
        raise NotImplementedError
