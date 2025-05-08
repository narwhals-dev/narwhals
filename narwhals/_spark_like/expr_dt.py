from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._duration import parse_interval_string
from narwhals._spark_like.utils import UNITS_DICT

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprDateTimeNamespace:
    def __init__(self, expr: SparkLikeExpr) -> None:
        self._compliant_expr = expr

    def date(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.to_date)

    def year(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.year)

    def month(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.month)

    def day(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.day)

    def hour(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.hour)

    def minute(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.minute)

    def second(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.second)

    def millisecond(self) -> SparkLikeExpr:
        def _millisecond(_input: Column) -> Column:
            return self._compliant_expr._F.floor(
                (self._compliant_expr._F.unix_micros(_input) % 1_000_000) / 1000
            )

        return self._compliant_expr._with_callable(_millisecond)

    def microsecond(self) -> SparkLikeExpr:
        def _microsecond(_input: Column) -> Column:
            return self._compliant_expr._F.unix_micros(_input) % 1_000_000

        return self._compliant_expr._with_callable(_microsecond)

    def nanosecond(self) -> SparkLikeExpr:
        def _nanosecond(_input: Column) -> Column:
            return (self._compliant_expr._F.unix_micros(_input) % 1_000_000) * 1000

        return self._compliant_expr._with_callable(_nanosecond)

    def ordinal_day(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.dayofyear)

    def weekday(self) -> SparkLikeExpr:
        def _weekday(_input: Column) -> Column:
            # PySpark's dayofweek returns 1-7 for Sunday-Saturday
            return (self._compliant_expr._F.dayofweek(_input) + 6) % 7

        return self._compliant_expr._with_callable(_weekday)

    def truncate(self, every: str) -> SparkLikeExpr:
        multiple, unit = parse_interval_string(every)
        if multiple != 1:
            msg = f"Only multiple 1 is currently supported for Spark-like.\nGot {multiple!s}."
            raise ValueError(msg)
        if unit == "ns":
            msg = "Truncating to nanoseconds is not yet supported for Spark-like."
            raise NotImplementedError(msg)
        format = UNITS_DICT[unit]

        def _truncate(_input: Column) -> Column:
            return self._compliant_expr._F.date_trunc(format, _input)

        return self._compliant_expr._with_callable(_truncate)
