from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from pyspark.sql import functions as F  # noqa: N812

from narwhals._spark_like.expr import SparkLikeExpr
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class SparkLikeSelectorNamespace:
    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._version = version

    def by_dtype(self: Self, dtypes: list[DType | type[DType]]) -> SparkLikeSelector:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [F.col(col) for col in df.columns if df.schema[col] in dtypes]

        def evalute_output_names(df: SparkLikeLazyFrame) -> Sequence[str]:
            return [col for col in df.columns if df.schema[col] in dtypes]

        return SparkLikeSelector(
            func,
            depth=0,
            function_name="selector",
            evaluate_output_names=evalute_output_names,
            alias_output_names=None,
            backend_version=self._backend_version,
            returns_scalar=False,
            version=self._version,
        )

    def numeric(self: Self) -> SparkLikeSelector:
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

    def categorical(self: Self) -> SparkLikeSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.Categorical])

    def string(self: Self) -> SparkLikeSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.String])

    def boolean(self: Self) -> SparkLikeSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.Boolean])

    def all(self: Self) -> SparkLikeSelector:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [F.col(col) for col in df.columns]

        return SparkLikeSelector(
            func,
            depth=0,
            function_name="selector",
            evaluate_output_names=lambda df: df.columns,
            alias_output_names=None,
            backend_version=self._backend_version,
            returns_scalar=False,
            version=self._version,
        )


class SparkLikeSelector(SparkLikeExpr):
    def __repr__(self: Self) -> str:  # pragma: no cover
        return (
            f"SparkLikeSelector("
            f"depth={self._depth}, "
            f"function_name={self._function_name})"
        )

    def _to_expr(self: Self) -> SparkLikeExpr:
        return SparkLikeExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            returns_scalar=self._returns_scalar,
            version=self._version,
        )

    def __sub__(self: Self, other: SparkLikeSelector | Any) -> SparkLikeSelector | Any:
        if isinstance(other, SparkLikeSelector):

            def call(df: SparkLikeLazyFrame) -> list[Column]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                return [x for x, name in zip(lhs, lhs_names) if name not in rhs_names]

            def evaluate_output_names(df: SparkLikeLazyFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [x for x in lhs_names if x not in rhs_names]

            return SparkLikeSelector(
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

    def __or__(self: Self, other: SparkLikeSelector | Any) -> SparkLikeSelector | Any:
        if isinstance(other, SparkLikeSelector):

            def call(df: SparkLikeLazyFrame) -> list[Column]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                rhs = other._call(df)
                return [
                    *(x for x, name in zip(lhs, lhs_names) if name not in rhs_names),
                    *rhs,
                ]

            def evaluate_output_names(df: SparkLikeLazyFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [*(x for x in lhs_names if x not in rhs_names), *rhs_names]

            return SparkLikeSelector(
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

    def __and__(self: Self, other: SparkLikeSelector | Any) -> SparkLikeSelector | Any:
        if isinstance(other, SparkLikeSelector):

            def call(df: SparkLikeLazyFrame) -> list[Column]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                return [x for x, name in zip(lhs, lhs_names) if name in rhs_names]

            def evaluate_output_names(df: SparkLikeLazyFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [x for x in lhs_names if x in rhs_names]

            return SparkLikeSelector(
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

    def __invert__(self: Self) -> SparkLikeSelector:
        return (
            SparkLikeSelectorNamespace(
                backend_version=self._backend_version, version=self._version
            ).all()
            - self
        )
