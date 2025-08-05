from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from narwhals._spark_like.utils import strptime_to_pyspark_format
from narwhals._sql.expr_str import SQLExprStringNamespace
from narwhals._utils import _is_naive_format, not_implemented

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprStringNamespace(SQLExprStringNamespace["SparkLikeExpr"]):
    def slice(self, offset: int, length: int | None) -> SparkLikeExpr:
        # From the docs: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.substring.html
        # The position is not zero based, but 1 based index.
        def func(expr: Column) -> Column:
            col_length = self.compliant._F.char_length(expr)

            _offset = (
                col_length + self.compliant._F.lit(offset + 1)
                if offset < 0
                else self.compliant._F.lit(offset + 1)
            )
            _length = self.compliant._F.lit(length) if length is not None else col_length
            return expr.substr(_offset, _length)

        return self.compliant._with_elementwise(func)

    def to_datetime(self, format: str | None) -> SparkLikeExpr:
        F = self.compliant._F  # noqa: N806
        if not format:
            function = F.to_timestamp
        elif _is_naive_format(format):
            function = partial(
                F.to_timestamp_ntz, format=F.lit(strptime_to_pyspark_format(format))
            )
        else:
            format = strptime_to_pyspark_format(format)
            function = partial(F.to_timestamp, format=format)
        return self.compliant._with_elementwise(
            lambda expr: function(F.replace(expr, F.lit("T"), F.lit(" ")))
        )

    def to_date(self, format: str | None) -> SparkLikeExpr:
        F = self.compliant._F  # noqa: N806
        return self.compliant._with_elementwise(
            lambda expr: F.to_date(expr, format=strptime_to_pyspark_format(format))
        )

    replace = not_implemented()
