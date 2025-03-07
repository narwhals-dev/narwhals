from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.expr import ArrowExpr
    from narwhals.typing import TimeUnit


class ArrowExprDateTimeNamespace:
    def __init__(self: Self, expr: ArrowExpr) -> None:
        self._compliant_expr = expr

    def to_string(self: Self, format: str) -> ArrowExpr:  # noqa: A002
        return self._compliant_expr._reuse_series_namespace(
            "dt", "to_string", format=format
        )

    def replace_time_zone(self: Self, time_zone: str | None) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace(
            "dt", "replace_time_zone", time_zone=time_zone
        )

    def convert_time_zone(self: Self, time_zone: str) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace(
            "dt", "convert_time_zone", time_zone=time_zone
        )

    def timestamp(self: Self, time_unit: TimeUnit) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace(
            "dt", "timestamp", time_unit=time_unit
        )

    def date(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "date")

    def year(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "year")

    def month(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "month")

    def day(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "day")

    def hour(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "hour")

    def minute(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "minute")

    def second(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "second")

    def millisecond(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "millisecond")

    def microsecond(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "microsecond")

    def nanosecond(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "nanosecond")

    def ordinal_day(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "ordinal_day")

    def weekday(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "weekday")

    def total_minutes(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "total_minutes")

    def total_seconds(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "total_seconds")

    def total_milliseconds(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "total_milliseconds")

    def total_microseconds(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "total_microseconds")

    def total_nanoseconds(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("dt", "total_nanoseconds")
