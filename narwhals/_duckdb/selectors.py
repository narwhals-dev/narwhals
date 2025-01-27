from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from duckdb import ColumnExpression

from narwhals._duckdb.expr import DuckDBExpr
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    import duckdb
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
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            return [
                ColumnExpression(col) for col in df.columns if df.schema[col] in dtypes
            ]

        def evalute_output_names(df: DuckDBLazyFrame) -> Sequence[str]:
            return [col for col in df.columns if df.schema[col] in dtypes]

        return DuckDBSelector(
            func,
            depth=0,
            function_name="selector",
            evaluate_output_names=evalute_output_names,
            alias_output_names=None,
            backend_version=self._backend_version,
            returns_scalar=False,
            version=self._version,
        )

    def numeric(self: Self) -> DuckDBSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype(
            [
                dtypes.Int128,
                dtypes.Int64,
                dtypes.Int32,
                dtypes.Int16,
                dtypes.Int8,
                dtypes.UInt128,
                dtypes.UInt64,
                dtypes.UInt32,
                dtypes.UInt16,
                dtypes.UInt8,
                dtypes.Float64,
                dtypes.Float32,
            ],
        )

    def categorical(self: Self) -> DuckDBSelector:  # pragma: no cover
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.Categorical])

    def string(self: Self) -> DuckDBSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.String])

    def boolean(self: Self) -> DuckDBSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.Boolean])

    def all(self: Self) -> DuckDBSelector:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            return [ColumnExpression(col) for col in df.columns]

        return DuckDBSelector(
            func,
            depth=0,
            function_name="selector",
            evaluate_output_names=lambda df: df.columns,
            alias_output_names=None,
            backend_version=self._backend_version,
            returns_scalar=False,
            version=self._version,
        )


class DuckDBSelector(DuckDBExpr):
    def __repr__(self: Self) -> str:  # pragma: no cover
        return (
            f"DuckDBSelector("
            f"depth={self._depth}, "
            f"function_name={self._function_name})"
        )

    def _to_expr(self: Self) -> DuckDBExpr:
        return DuckDBExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            returns_scalar=self._returns_scalar,
            version=self._version,
        )

    def __sub__(self: Self, other: DuckDBSelector | Any) -> DuckDBSelector | Any:
        if isinstance(other, DuckDBSelector):

            def call(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                return [x for x, name in zip(lhs, lhs_names) if name not in rhs_names]

            def evaluate_output_names(df: DuckDBLazyFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [x for x in lhs_names if x not in rhs_names]

            return DuckDBSelector(
                call,
                depth=0,
                function_name="selector",
                evaluate_output_names=evaluate_output_names,
                alias_output_names=None,
                backend_version=self._backend_version,
                returns_scalar=self._returns_scalar,
                version=self._version,
            )
        else:
            return self._to_expr() - other

    def __or__(self: Self, other: DuckDBSelector | Any) -> DuckDBSelector | Any:
        if isinstance(other, DuckDBSelector):

            def call(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                rhs = other._call(df)
                return [
                    *(x for x, name in zip(lhs, lhs_names) if name not in rhs_names),
                    *rhs,
                ]

            def evaluate_output_names(df: DuckDBLazyFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [*(x for x in lhs_names if x not in rhs_names), *rhs_names]

            return DuckDBSelector(
                call,
                depth=0,
                function_name="selector",
                evaluate_output_names=evaluate_output_names,
                alias_output_names=None,
                backend_version=self._backend_version,
                returns_scalar=self._returns_scalar,
                version=self._version,
            )
        else:
            return self._to_expr() | other

    def __and__(self: Self, other: DuckDBSelector | Any) -> DuckDBSelector | Any:
        if isinstance(other, DuckDBSelector):

            def call(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                return [x for x, name in zip(lhs, lhs_names) if name in rhs_names]

            def evaluate_output_names(df: DuckDBLazyFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [x for x in lhs_names if x in rhs_names]

            return DuckDBSelector(
                call,
                depth=0,
                function_name="selector",
                evaluate_output_names=evaluate_output_names,
                alias_output_names=None,
                backend_version=self._backend_version,
                returns_scalar=self._returns_scalar,
                version=self._version,
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
