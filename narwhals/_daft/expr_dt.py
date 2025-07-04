from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import DateTimeNamespace
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from narwhals._daft.expr import DaftExpr


class DaftExprDateTimeNamespace(
    LazyExprNamespace["DaftExpr"], DateTimeNamespace["DaftExpr"]
):
    def date(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.date())

    def year(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.year())

    def month(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.month())

    def day(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.day())

    def hour(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.hour())

    def minute(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.minute())

    def second(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.second())

    def millisecond(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.millisecond())

    def microsecond(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.microsecond())

    def nanosecond(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.nanosecond())

    def weekday(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.day_of_week() + 1)

    def to_string(self, format: str | None) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.strftime(format))

    def ordinal_day(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: expr.dt.day_of_year())

    convert_time_zone = not_implemented()
    replace_time_zone = not_implemented()
    timestamp = not_implemented()
    truncate = not_implemented()
    total_hours = not_implemented()
    total_minutes = not_implemented()
    total_seconds = not_implemented()
    total_milliseconds = not_implemented()
    total_microseconds = not_implemented()
    total_nanoseconds = not_implemented()
