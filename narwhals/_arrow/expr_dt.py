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
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "to_string", format=format
        )

    def replace_time_zone(self: Self, time_zone: str | None) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "replace_time_zone", time_zone=time_zone
        )

    def convert_time_zone(self: Self, time_zone: str) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "convert_time_zone", time_zone=time_zone
        )

    def timestamp(self: Self, time_unit: TimeUnit) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "timestamp", time_unit=time_unit
        )

    def date(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation("dt", "date")

    def year(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation("dt", "year")

    def month(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation("dt", "month")

    def day(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation("dt", "day")

    def hour(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation("dt", "hour")

    def minute(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation("dt", "minute")

    def second(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation("dt", "second")

    def millisecond(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "millisecond"
        )

    def microsecond(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "microsecond"
        )

    def nanosecond(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "nanosecond"
        )

    def ordinal_day(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "ordinal_day"
        )

    def weekday(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "weekday"
        )

    def total_minutes(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "total_minutes"
        )

    def total_seconds(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "total_seconds"
        )

    def total_milliseconds(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "total_milliseconds"
        )

    def total_microseconds(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "total_microseconds"
        )

    def total_nanoseconds(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "dt", "total_nanoseconds"
        )
