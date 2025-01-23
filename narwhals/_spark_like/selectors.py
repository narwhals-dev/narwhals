from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from pyspark.sql import functions as F  # noqa: N812

from narwhals._spark_like.expr import SparkLikeExpr
from narwhals._spark_like.utils import get_column_name
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
        def func(df: SparkLikeLazyFrame) -> list[Any]:
            return [
                df._native_frame[col] for col in df.columns if df.schema[col] in dtypes
            ]

        return SparkLikeSelector(
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

    def numeric(self: Self) -> SparkLikeSelector:
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

    def string(self: Self) -> SparkLikeSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.String])

    def boolean(self: Self) -> SparkLikeSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.Boolean])

    def all(self: Self) -> SparkLikeSelector:
        def func(df: SparkLikeLazyFrame) -> list[Any]:
            return [df._native_frame[col] for col in df.columns]

        return SparkLikeSelector(
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


class SparkLikeSelector(SparkLikeExpr):
    def __repr__(self: Self) -> str:  # pragma: no cover
        return (
            f"SparkLikeSelector("
            f"depth={self._depth}, "
            f"function_name={self._function_name}, "
            f"root_names={self._root_names}, "
            f"output_names={self._output_names}"
        )

    def _to_expr(self: Self) -> SparkLikeExpr:
        return SparkLikeExpr(
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

    def __sub__(self: Self, other: SparkLikeSelector | Any) -> SparkLikeSelector | Any:
        if isinstance(other, SparkLikeSelector):

            def call(df: SparkLikeLazyFrame) -> list[Any]:
                lhs = self._call(df)
                rhs = other._call(df)
                lhs_names = [get_column_name(df, x) for x in lhs]
                rhs_names = {get_column_name(df, x) for x in rhs}
                return [col for col, name in zip(lhs, lhs_names) if name not in rhs_names]

            return SparkLikeSelector(
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
            return self._to_expr() - F.lit(other)

    def __or__(self: Self, other: SparkLikeSelector | Any) -> SparkLikeSelector | Any:
        if isinstance(other, SparkLikeSelector):

            def call(df: SparkLikeLazyFrame) -> list[Column]:
                lhs = self._call(df)
                rhs = other._call(df)
                lhs_names = [get_column_name(df, x) for x in lhs]
                rhs_names = [get_column_name(df, x) for x in rhs]
                return [
                    *(col for col, name in zip(lhs, lhs_names) if name not in rhs_names),
                    *rhs,
                ]

            return SparkLikeSelector(
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
            return self._to_expr() | F.lit(other)

    def __and__(self: Self, other: SparkLikeSelector | Any) -> SparkLikeSelector | Any:
        if isinstance(other, SparkLikeSelector):

            def call(df: SparkLikeLazyFrame) -> list[Any]:
                lhs = self._call(df)
                rhs = other._call(df)
                lhs_names = [get_column_name(df, x) for x in lhs]
                rhs_names = {get_column_name(df, x) for x in rhs}
                return [col for col, name in zip(lhs, lhs_names) if name in rhs_names]

            return SparkLikeSelector(
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
            return self._to_expr() & F.lit(other)

    def __invert__(self: Self) -> SparkLikeSelector:
        return (
            SparkLikeSelectorNamespace(
                backend_version=self._backend_version, version=self._version
            ).all()
            - self
        )
