from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprDateTimeNamespace:
    def __init__(self, expr: SparkLikeExpr) -> None:
        self._compliant_expr = expr

    def date(self: Self) -> SparkLikeExpr:
        from pyspark.sql import functions as F  # noqa: N812

        return self._compliant_expr._from_call(
            F.to_date,
            "date",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def year(self: Self) -> SparkLikeExpr:
        from pyspark.sql import functions as F  # noqa: N812

        return self._compliant_expr._from_call(
            F.year,
            "year",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def month(self: Self) -> SparkLikeExpr:
        from pyspark.sql import functions as F  # noqa: N812

        return self._compliant_expr._from_call(
            F.month,
            "month",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def day(self: Self) -> SparkLikeExpr:
        from pyspark.sql import functions as F  # noqa: N812

        return self._compliant_expr._from_call(
            F.day,
            "day",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def hour(self: Self) -> SparkLikeExpr:
        from pyspark.sql import functions as F  # noqa: N812

        return self._compliant_expr._from_call(
            F.hour,
            "hour",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def minute(self: Self) -> SparkLikeExpr:
        from pyspark.sql import functions as F  # noqa: N812

        return self._compliant_expr._from_call(
            F.minute,
            "minute",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def second(self: Self) -> SparkLikeExpr:
        from pyspark.sql import functions as F  # noqa: N812

        return self._compliant_expr._from_call(
            F.second,
            "second",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def millisecond(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            lambda _input: (_input.cast("double") * 1_000) % 1_000,
            "millisecond",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def microsecond(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            lambda _input: ((_input.cast("double") * 1_000) % 1_000) * 1_000,
            "microsecond",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def nanosecond(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            lambda _input: ((_input.cast("double") * 1_000) % 1_000) * 1_000_000,
            "nanosecond",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def ordinal_day(self: Self) -> SparkLikeExpr:
        from pyspark.sql import functions as F  # noqa: N812

        return self._compliant_expr._from_call(
            F.dayofyear,
            "ordinal_day",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def weekday(self: Self) -> SparkLikeExpr:
        from pyspark.sql import functions as F  # noqa: N812

        def _weekday(_input: Column) -> Column:
            # From F.dayofweek docstring:
            # Ranges from 1 for a Sunday through to 7 for a Saturday
            _tmp = F.dayofweek(_input) - 1  # (0 Sunday -> 6 Saturday)
            return F.when(_tmp == 0, F.lit(7)).otherwise(_tmp)

        return self._compliant_expr._from_call(
            _weekday,
            "weekday",
            returns_scalar=self._compliant_expr._returns_scalar,
        )
