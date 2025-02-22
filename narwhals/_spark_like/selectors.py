from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterator

from narwhals._selectors import CompliantSelector
from narwhals._selectors import LazySelectorNamespace
from narwhals._spark_like.expr import SparkLikeExpr

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._selectors import EvalNames
    from narwhals._selectors import EvalSeries
    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals.utils import _FullContext


class SparkLikeSelectorNamespace(LazySelectorNamespace["SparkLikeLazyFrame", "Column"]):
    def _iter_columns(self, df: SparkLikeLazyFrame) -> Iterator[Column]:
        for col in df.columns:
            yield df._F.col(col)

    def _selector(
        self,
        context: _FullContext,
        call: EvalSeries[SparkLikeLazyFrame, Column],
        evaluate_output_names: EvalNames[SparkLikeLazyFrame],
        /,
    ) -> SparkLikeSelector:
        return SparkLikeSelector(
            call,
            function_name="selector",
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
            implementation=context._implementation,
        )

    def __init__(self: Self, context: _FullContext, /) -> None:
        self._backend_version = context._backend_version
        self._version = context._version
        self._implementation = context._implementation


class SparkLikeSelector(CompliantSelector["SparkLikeLazyFrame", "Column"], SparkLikeExpr):  # type: ignore[misc]
    @property
    def selectors(self: Self) -> SparkLikeSelectorNamespace:
        return SparkLikeSelectorNamespace(self)

    def _to_expr(self: Self) -> SparkLikeExpr:
        return SparkLikeExpr(
            self._call,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )
