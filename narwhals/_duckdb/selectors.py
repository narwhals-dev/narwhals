from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn

from duckdb import ColumnExpression
from duckdb import Expression

from narwhals._duckdb.expr import DuckDBExpr
from narwhals._duckdb.utils import get_column_name
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class DuckDBSelectorNamespace:
    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._version = version

    def by_dtype(self: Self, dtypes: list[DType | type[DType]]) -> DuckDBSelector:
        def func(df: DuckDBLazyFrame) -> list[Expression]:
            return [
                ColumnExpression(col) for col in df.columns if df.schema[col] in dtypes
            ]

        return DuckDBSelector(
            func,
            depth=0,
            function_name="type_selector",
            root_names=None,
            output_names=None,
            backend_version=self._backend_version,
            returns_scalar=False,
            version=self._version,
            kwargs={},
        )

    def numeric(self: Self) -> DuckDBSelector:
        dtypes = import_dtypes_module(self._version)
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

    def categorical(self: Self) -> DuckDBSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.Categorical])

    def string(self: Self) -> DuckDBSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.String])

    def boolean(self: Self) -> DuckDBSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.Boolean])

    def all(self: Self) -> DuckDBSelector:
        def func(df: DuckDBLazyFrame) -> list[Any]:
            return [ColumnExpression(col) for col in df.columns]

        return DuckDBSelector(
            func,
            depth=0,
            function_name="type_selector",
            root_names=None,
            output_names=None,
            backend_version=self._backend_version,
            returns_scalar=False,
            version=self._version,
            kwargs={},
        )


class DuckDBSelector(DuckDBExpr):
    def __repr__(self: Self) -> str:  # pragma: no cover
        return (
            f"DuckDBSelector("
            f"depth={self._depth}, "
            f"function_name={self._function_name}, "
            f"root_names={self._root_names}, "
            f"output_names={self._output_names}"
        )

    def _to_expr(self: Self) -> DuckDBExpr:
        return DuckDBExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            root_names=self._root_names,
            output_names=self._output_names,
            backend_version=self._backend_version,
            returns_scalar=self._returns_scalar,
            version=self._version,
            kwargs={},
        )

    def __sub__(self: Self, other: DuckDBSelector | Any) -> DuckDBSelector | Any:
        if isinstance(other, DuckDBSelector):

            def call(df: DuckDBLazyFrame) -> list[Any]:
                lhs = self._call(df)
                rhs = other._call(df)
                lhs_names = [
                    get_column_name(df, x, returns_scalar=self._returns_scalar)
                    for x in lhs
                ]
                rhs_names = {
                    get_column_name(df, x, returns_scalar=other._returns_scalar)
                    for x in rhs
                }
                return [col for col, name in zip(lhs, lhs_names) if name not in rhs_names]

            return DuckDBSelector(
                call,
                depth=0,
                function_name="type_selector",
                root_names=None,
                output_names=None,
                backend_version=self._backend_version,
                returns_scalar=self._returns_scalar,
                version=self._version,
                kwargs={},
            )
        else:
            return self._to_expr() - other

    def __or__(self: Self, other: DuckDBSelector | Any) -> DuckDBSelector | Any:
        if isinstance(other, DuckDBSelector):

            def call(df: DuckDBLazyFrame) -> list[Column]:
                lhs = self._call(df)
                rhs = other._call(df)
                lhs_names = [
                    get_column_name(df, x, returns_scalar=self._returns_scalar)
                    for x in lhs
                ]
                rhs_names = [
                    get_column_name(df, x, returns_scalar=other._returns_scalar)
                    for x in rhs
                ]
                return [
                    *(col for col, name in zip(lhs, lhs_names) if name not in rhs_names),
                    *rhs,
                ]

            return DuckDBSelector(
                call,
                depth=0,
                function_name="type_selector",
                root_names=None,
                output_names=None,
                backend_version=self._backend_version,
                returns_scalar=self._returns_scalar,
                version=self._version,
                kwargs={},
            )
        else:
            return self._to_expr() | other

    def __and__(self: Self, other: DuckDBSelector | Any) -> DuckDBSelector | Any:
        if isinstance(other, DuckDBSelector):

            def call(df: DuckDBLazyFrame) -> list[Any]:
                lhs = self._call(df)
                rhs = other._call(df)
                lhs_names = [
                    get_column_name(df, x, returns_scalar=self._returns_scalar)
                    for x in lhs
                ]
                rhs_names = {
                    get_column_name(df, x, returns_scalar=other._returns_scalar)
                    for x in rhs
                }
                return [col for col, name in zip(lhs, lhs_names) if name in rhs_names]

            return DuckDBSelector(
                call,
                depth=0,
                function_name="type_selector",
                root_names=None,
                output_names=None,
                backend_version=self._backend_version,
                returns_scalar=self._returns_scalar,
                version=self._version,
                kwargs={},
            )
        else:
            return self._to_expr() & other

    def __invert__(self: Self) -> DuckDBSelector:
        return (
            DuckDBSelectorNamespace(
                backend_version=self._backend_version, version=self._version
            ).all()
            - self
        )

    def __rsub__(self: Self, other: Any) -> NoReturn:
        raise NotImplementedError

    def __rand__(self: Self, other: Any) -> NoReturn:
        raise NotImplementedError

    def __ror__(self: Self, other: Any) -> NoReturn:
        raise NotImplementedError
