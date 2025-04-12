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
        frame, self._keys, self._output_key_names = self._init_parsing(
            compliant_frame=df, keys=keys
        )
        self._compliant_frame = (
            frame.drop_nulls(subset=self._keys) if drop_null_keys else frame
        )

    def agg(self, *exprs: SparkLikeExpr) -> SparkLikeLazyFrame:
        result = (
            self.compliant.native.groupBy(*self._keys).agg(*agg_columns)
            if (agg_columns := list(self._evaluate_exprs(exprs)))
            else self.compliant.native.select(*self._keys).dropDuplicates()
        )

        return self.compliant._with_native(result).rename(
            dict(zip(self._keys, self._output_key_names))
        )
