from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F  # noqa: N812

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprDateTimeNamespace:
    def __init__(self: Self, expr: SparkLikeExpr) -> None:
        self._compliant_expr = expr

    def date(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.to_date,
            "date",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def year(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.year,
            "year",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def month(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.month,
            "month",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def day(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.day,
            "day",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def hour(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.hour,
            "hour",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def minute(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.minute,
            "minute",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def second(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.second,
            "second",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def millisecond(self: Self) -> SparkLikeExpr:
        def _millisecond(_input: Column) -> Column:
            return F.floor((F.unix_micros(_input) % 1_000_000) / 1000)

        return self._compliant_expr._from_call(
            _millisecond,
            "millisecond",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def microsecond(self: Self) -> SparkLikeExpr:
        def _microsecond(_input: Column) -> Column:
            return F.unix_micros(_input) % 1_000_000

        return self._compliant_expr._from_call(
            _microsecond,
            "microsecond",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def nanosecond(self: Self) -> SparkLikeExpr:
        def _nanosecond(_input: Column) -> Column:
            return (F.unix_micros(_input) % 1_000_000) * 1000

        return self._compliant_expr._from_call(
            _nanosecond,
            "nanosecond",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def ordinal_day(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.dayofyear,
            "ordinal_day",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def weekday(self: Self) -> SparkLikeExpr:
        def _weekday(_input: Column) -> Column:
            # PySpark's dayofweek returns 1-7 for Sunday-Saturday
            return (F.dayofweek(_input) + 6) % 7

        return self._compliant_expr._from_call(
            _weekday,
            "weekday",
            returns_scalar=self._compliant_expr._returns_scalar,
        )
