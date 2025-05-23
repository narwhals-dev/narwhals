from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._duration import parse_interval_string
from narwhals._spark_like.utils import UNITS_DICT, strptime_to_pyspark_format

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprDateTimeNamespace:
    def __init__(self, expr: SparkLikeExpr) -> None:
        self._compliant_expr = expr

    def to_string(self, format: str) -> SparkLikeExpr:
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

        return self._compliant_expr._with_callable(_to_string)

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
        def _millisecond(expr: Column) -> Column:
            return self._compliant_expr._F.floor(
                (self._compliant_expr._F.unix_micros(expr) % 1_000_000) / 1000
            )

        return self._compliant_expr._with_callable(_millisecond)

    def microsecond(self) -> SparkLikeExpr:
        def _microsecond(expr: Column) -> Column:
            return self._compliant_expr._F.unix_micros(expr) % 1_000_000

        return self._compliant_expr._with_callable(_microsecond)

    def nanosecond(self) -> SparkLikeExpr:
        def _nanosecond(expr: Column) -> Column:
            return (self._compliant_expr._F.unix_micros(expr) % 1_000_000) * 1000

        return self._compliant_expr._with_callable(_nanosecond)

    def ordinal_day(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.dayofyear)

    def weekday(self) -> SparkLikeExpr:
        def _weekday(expr: Column) -> Column:
            # PySpark's dayofweek returns 1-7 for Sunday-Saturday
            return (self._compliant_expr._F.dayofweek(expr) + 6) % 7

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

        def _truncate(expr: Column) -> Column:
            return self._compliant_expr._F.date_trunc(format, expr)

        return self._compliant_expr._with_callable(_truncate)

    def replace_time_zone(self, time_zone: str | None) -> SparkLikeExpr:
        if time_zone is None:
            return self._compliant_expr._with_callable(
                lambda _input: _input.cast("timestamp_ntz")
            )
        else:  # pragma: no cover
            msg = "`replace_time_zone` with non-null `time_zone` not yet implemented for spark-like"
            raise NotImplementedError(msg)
