from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Sequence

from narwhals._compliant import LazyGroupBy

if TYPE_CHECKING:
    from sqlframe.base.column import Column  # noqa: F401
    from typing_extensions import Self

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeLazyGroupBy(LazyGroupBy["SparkLikeLazyFrame", "SparkLikeExpr", "Column"]):
    def __init__(
        self: Self,
        compliant_frame: SparkLikeLazyFrame,
        keys: Sequence[SparkLikeExpr],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        compliant_frame = compliant_frame.with_columns(*keys)
        self._keys = [
            key._alias_output_names(key._evaluate_output_names(compliant_frame))[0]
            if key._alias_output_names is not None
            else key._evaluate_output_names(compliant_frame)[0]
            for key in keys
        ]
        self._compliant_frame = (
            compliant_frame.drop_nulls(subset=self._keys)
            if drop_null_keys
            else compliant_frame
        )

    def agg(self: Self, *exprs: SparkLikeExpr) -> SparkLikeLazyFrame:
        if agg_columns := list(self._evaluate_exprs(exprs)):
            return self.compliant._with_native(
                self.compliant.native.groupBy(*self._keys).agg(*agg_columns)
            )
        return self.compliant._with_native(
            self.compliant.native.select(*self._keys).dropDuplicates()
        )
