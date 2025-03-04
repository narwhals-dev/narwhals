from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprStructNamespace:
    def __init__(self: Self, expr: SparkLikeExpr) -> None:
        self._compliant_expr = expr

    def field(self: Self, *names: str) -> SparkLikeExpr:
        expr = self._compliant_expr

        def funcs(df: SparkLikeLazyFrame) -> list[Column]:
            native_series_list = expr._call(df)
            cols = []
            for series in native_series_list:
                for name in names:
                    cols += [series.getField(name)]
            return cols

        return expr.__class__(
            funcs,
            function_name=f"{expr._function_name}->field({names})",
            evaluate_output_names=lambda _df: names,
            alias_output_names=None,
            backend_version=expr._backend_version,
            version=expr._version,
            implementation=expr._implementation,
        )
