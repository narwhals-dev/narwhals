from __future__ import annotations

from typing import TYPE_CHECKING

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
            expr_kind=self._compliant_expr._expr_kind,
        )

    def year(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.year,
            "year",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def month(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.month,
            "month",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def day(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.day,
            "day",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def hour(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.hour,
            "hour",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def minute(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.minute,
            "minute",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def second(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.second,
            "second",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def millisecond(self: Self) -> SparkLikeExpr:
        def _millisecond(_input: Column) -> Column:
            return F.floor((F.unix_micros(_input) % 1_000_000) / 1000)

        return self._compliant_expr._from_call(
            _millisecond,
            "millisecond",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def microsecond(self: Self) -> SparkLikeExpr:
        def _microsecond(_input: Column) -> Column:
            return F.unix_micros(_input) % 1_000_000

        return self._compliant_expr._from_call(
            _microsecond,
            "microsecond",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def nanosecond(self: Self) -> SparkLikeExpr:
        def _nanosecond(_input: Column) -> Column:
            return (F.unix_micros(_input) % 1_000_000) * 1000

        return self._compliant_expr._from_call(
            _nanosecond,
            "nanosecond",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def ordinal_day(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            F.dayofyear,
            "ordinal_day",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def weekday(self: Self) -> SparkLikeExpr:
        def _weekday(_input: Column) -> Column:
            # PySpark's dayofweek returns 1-7 for Sunday-Saturday
            return (F.dayofweek(_input) + 6) % 7

        return self._compliant_expr._from_call(
            _weekday,
            "weekday",
            expr_kind=self._compliant_expr._expr_kind,
        )
