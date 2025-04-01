from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import CompliantSelector
from narwhals._compliant import LazySelectorNamespace
from narwhals._spark_like.expr import SparkLikeExpr

if TYPE_CHECKING:
    from sqlframe.base.column import Column
    from typing_extensions import Self

    from narwhals._compliant import EvalNames
    from narwhals._compliant import EvalSeries
    from narwhals._spark_like.dataframe import SparkLikeLazyFrame


class SparkLikeSelectorNamespace(LazySelectorNamespace["SparkLikeLazyFrame", "Column"]):
    def _selector(
        self,
        call: EvalSeries[SparkLikeLazyFrame, Column],
        evaluate_output_names: EvalNames[SparkLikeLazyFrame],
        /,
    ) -> SparkLikeSelector:
        return SparkLikeSelector(
            call,
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )


class SparkLikeSelector(CompliantSelector["SparkLikeLazyFrame", "Column"], SparkLikeExpr):  # type: ignore[misc]
    def _to_expr(self: Self) -> SparkLikeExpr:
        return SparkLikeExpr(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )
