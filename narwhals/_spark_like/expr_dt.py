from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from narwhals._duration import parse_interval_string
from narwhals._spark_like.utils import (
    UNITS_DICT,
    fetch_session_time_zone,
    strptime_to_pyspark_format,
)

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprDateTimeNamespace:
    def __init__(self, expr: SparkLikeExpr) -> None:
        self._compliant_expr = expr

    def to_string(self, format: str) -> SparkLikeExpr:
        F = self._compliant_expr._F  # noqa: N806

        def _to_string(_input: Column) -> Column:
            # Handle special formats
            if format == "%G-W%V":
                return self._format_iso_week(_input)
            if format == "%G-W%V-%u":
                return self._format_iso_week_with_day(_input)

            format_, suffix = self._format_microseconds(_input, format)

            # Convert Python format to PySpark format
            pyspark_fmt = strptime_to_pyspark_format(format_)

            result = F.date_format(_input, pyspark_fmt)
            if "T" in format_:
                # `strptime_to_pyspark_format` replaces "T" with " " since pyspark
                # does not support the literal "T" in `date_format`.
                # If no other spaces are in the given format, then we can revert this
                # operation, otherwise we raise an exception.
                if " " not in format_:
                    result = F.replace(result, F.lit(" "), F.lit("T"))
                else:  # pragma: no cover
                    msg = (
                        "`dt.to_string` with a format that contains both spaces and "
                        " the literal 'T' is not supported for spark-like backends."
                    )
                    raise NotImplementedError(msg)

            return F.concat(result, *suffix)

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

    def _no_op_time_zone(self, time_zone: str) -> SparkLikeExpr:  # pragma: no cover
        def func(df: SparkLikeLazyFrame) -> Sequence[Column]:
            native_series_list = self._compliant_expr(df)
            conn_time_zone = fetch_session_time_zone(df.native.sparkSession)
            if conn_time_zone != time_zone:
                msg = (
                    "PySpark stores the time zone in the session, rather than in the "
                    f"data type, so changing the timezone to anything other than {conn_time_zone} "
                    " (the current session time zone) is not supported."
                )
                raise NotImplementedError(msg)
            return native_series_list

        return self._compliant_expr.__class__(
            func,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=self._compliant_expr._alias_output_names,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            implementation=self._compliant_expr._implementation,
        )

    def convert_time_zone(self, time_zone: str) -> SparkLikeExpr:  # pragma: no cover
        return self._no_op_time_zone(time_zone)

    def replace_time_zone(
        self, time_zone: str | None
    ) -> SparkLikeExpr:  # pragma: no cover
        if time_zone is None:
            return self._compliant_expr._with_callable(
                lambda _input: _input.cast("timestamp_ntz")
            )
        else:
            return self._no_op_time_zone(time_zone)

    def _format_iso_week_with_day(self, _input: Column) -> Column:
        """Format datetime as ISO week string with day."""
        F = self._compliant_expr._F  # noqa: N806

        year = F.date_format(_input, "yyyy")
        week = F.lpad(F.weekofyear(_input).cast("string"), 2, "0")
        day = F.dayofweek(_input)
        # Adjust Sunday from 1 to 7
        day = F.when(day == 1, 7).otherwise(day - 1)
        return F.concat(year, F.lit("-W"), week, F.lit("-"), day.cast("string"))

    def _format_iso_week(self, _input: Column) -> Column:
        """Format datetime as ISO week string."""
        F = self._compliant_expr._F  # noqa: N806

        year = F.date_format(_input, "yyyy")
        week = F.lpad(F.weekofyear(_input).cast("string"), 2, "0")
        return F.concat(year, F.lit("-W"), week)

    def _format_microseconds(
        self, _input: Column, format: str
    ) -> tuple[str, tuple[Column, ...]]:
        """Format microseconds if present in format, else it's a no-op."""
        F = self._compliant_expr._F  # noqa: N806

        suffix: tuple[Column, ...]
        if format.endswith((".%f", "%.f")):
            import re

            micros = F.unix_micros(_input) % 1_000_000
            micros_str = F.lpad(micros.cast("string"), 6, "0")
            suffix = (F.lit("."), micros_str)
            format_ = re.sub(r"(.%|%.)f$", "", format)
            return format_, suffix

        return format, ()
