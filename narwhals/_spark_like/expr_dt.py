from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._spark_like.utils import strptime_to_pyspark_format

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprDateTimeNamespace:
    def __init__(self: Self, expr: SparkLikeExpr) -> None:
        self._compliant_expr = expr

    def to_string(self: Self, format: str) -> SparkLikeExpr:  # noqa: A002
        F = self._compliant_expr._F  # noqa: N806

        def _format_iso_week_with_day(_input: Column) -> Column:
            """Format datetime as ISO week string with day."""
            year = F.date_format(_input, "yyyy")
            week = F.lpad(F.weekofyear(_input).cast("string"), 2, "0")
            day = F.dayofweek(_input)
            # Adjust Sunday from 1 to 7
            day = F.when(day == 1, 7).otherwise(day - 1)
            return F.concat(year, F.lit("-W"), week, F.lit("-"), day.cast("string"))

        def _format_iso_week(_input: Column) -> Column:
            """Format datetime as ISO week string."""
            year = F.date_format(_input, "yyyy")
            week = F.lpad(F.weekofyear(_input).cast("string"), 2, "0")
            return F.concat(year, F.lit("-W"), week)

        def _format_iso_datetime(_input: Column) -> Column:
            """Format datetime as ISO datetime with microseconds."""
            date_part = F.date_format(_input, "yyyy-MM-dd")
            time_part = F.date_format(_input, "HH:mm:ss")
            micros = F.unix_micros(_input) % 1_000_000
            micros_str = F.lpad(micros.cast("string"), 6, "0")
            return F.concat(date_part, F.lit("T"), time_part, F.lit("."), micros_str)

        def _to_string(_input: Column) -> Column:
            # Handle special formats
            if format == "%G-W%V":
                return _format_iso_week(_input)
            if format == "%G-W%V-%u":
                return _format_iso_week_with_day(_input)
            if format in {"%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S%.f"}:
                return _format_iso_datetime(_input)

            # Convert Python format to PySpark format
            pyspark_fmt = strptime_to_pyspark_format(format)
            return F.date_format(_input, pyspark_fmt)

        return self._compliant_expr._from_call(_to_string, "to_string")

    def date(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(self._compliant_expr._F.to_date, "date")

    def year(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(self._compliant_expr._F.year, "year")

    def month(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(self._compliant_expr._F.month, "month")

    def day(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(self._compliant_expr._F.day, "day")

    def hour(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(self._compliant_expr._F.hour, "hour")

    def minute(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(self._compliant_expr._F.minute, "minute")

    def second(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(self._compliant_expr._F.second, "second")

    def millisecond(self: Self) -> SparkLikeExpr:
        def _millisecond(_input: Column) -> Column:
            return self._compliant_expr._F.floor(
                (self._compliant_expr._F.unix_micros(_input) % 1_000_000) / 1000
            )

        return self._compliant_expr._from_call(_millisecond, "millisecond")

    def microsecond(self: Self) -> SparkLikeExpr:
        def _microsecond(_input: Column) -> Column:
            return self._compliant_expr._F.unix_micros(_input) % 1_000_000

        return self._compliant_expr._from_call(_microsecond, "microsecond")

    def nanosecond(self: Self) -> SparkLikeExpr:
        def _nanosecond(_input: Column) -> Column:
            return (self._compliant_expr._F.unix_micros(_input) % 1_000_000) * 1000

        return self._compliant_expr._from_call(_nanosecond, "nanosecond")

    def ordinal_day(self: Self) -> SparkLikeExpr:
        return self._compliant_expr._from_call(
            self._compliant_expr._F.dayofyear, "ordinal_day"
        )

    def weekday(self: Self) -> SparkLikeExpr:
        def _weekday(_input: Column) -> Column:
            # PySpark's dayofweek returns 1-7 for Sunday-Saturday
            return (self._compliant_expr._F.dayofweek(_input) + 6) % 7

        return self._compliant_expr._from_call(_weekday, "weekday")
