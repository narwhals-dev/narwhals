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

    def to_string(self: Self, format: str) -> SparkLikeExpr:  # noqa: A002
        def _format_iso_week_with_day(_input: Column) -> Column:
            """Format datetime as ISO week string with day."""
            year = F.date_format(_input, "YYYY")
            week = F.lpad(F.weekofyear(_input).cast("string"), 2, "0")
            day = F.dayofweek(_input)
            # Adjust Sunday from 1 to 7
            day = F.when(day == 1, 7).otherwise(day - 1)
            return F.concat(year, F.lit("-W"), week, F.lit("-"), day.cast("string"))

        def _format_iso_week(_input: Column) -> Column:
            """Format datetime as ISO week string."""
            year = F.date_format(_input, "YYYY")
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
            if format in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S%.f"):
                return _format_iso_datetime(_input)

            # Standard format conversions
            java_fmt = (
                format.replace("%Y", "yyyy")
                .replace("%m", "MM")
                .replace("%d", "dd")
                .replace("%H", "HH")
                .replace("%M", "mm")
                .replace("%S", "ss")
            )
            return F.date_format(_input, java_fmt)

        return self._compliant_expr._from_call(
            _to_string,
            "to_string",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

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
