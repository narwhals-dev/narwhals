from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Sequence

from narwhals._compliant import LazyGroupBy

if TYPE_CHECKING:
    from sqlframe.base.column import Column  # noqa: F401

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeLazyGroupBy(LazyGroupBy["SparkLikeLazyFrame", "SparkLikeExpr", "Column"]):
    def __init__(
        self,
        df: SparkLikeLazyFrame,
        keys: Sequence[SparkLikeExpr],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._keys = df._evaluate_aliases(*keys)
        self._compliant_frame = df.drop_nulls(subset=self._keys) if drop_null_keys else df

    def agg(self, *exprs: SparkLikeExpr) -> SparkLikeLazyFrame:
        if agg_columns := list(self._evaluate_exprs(exprs)):
            return self.compliant._with_native(
                self.compliant.native.groupBy(*self._keys).agg(*agg_columns)
            )
        return self.compliant._with_native(
            self.compliant.native.select(*self._keys).dropDuplicates()
        )
