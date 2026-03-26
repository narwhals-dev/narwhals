from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.expressions.namespace import ExprNamespace
from narwhals._plan.expressions.temporal import IRDateTimeNamespace

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr
    from narwhals.typing import TimeUnit


class ExprDateTimeNamespace(ExprNamespace[IRDateTimeNamespace]):
    @property
    def _ir_namespace(self) -> type[IRDateTimeNamespace]:
        return IRDateTimeNamespace

    def date(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.date())

    def year(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.year())

    def month(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.month())

    def day(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.day())

    def hour(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.hour())

    def minute(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.minute())

    def second(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.second())

    def millisecond(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.millisecond())

    def microsecond(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.microsecond())

    def nanosecond(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.nanosecond())

    def ordinal_day(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.ordinal_day())

    def weekday(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.weekday())

    def total_minutes(self) -> Expr:
        return self._with_unary(self._ir.total_minutes())

    def total_seconds(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.total_seconds())

    def total_milliseconds(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.total_milliseconds())

    def total_microseconds(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.total_microseconds())

    def total_nanoseconds(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.total_nanoseconds())

    def to_string(self, format: str) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.to_string(format=format))

    def replace_time_zone(self, time_zone: str | None) -> Expr:
        return self._with_unary(self._ir.replace_time_zone(time_zone=time_zone))

    def convert_time_zone(self, time_zone: str) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.convert_time_zone(time_zone=time_zone))

    def timestamp(self, time_unit: TimeUnit = "us") -> Expr:
        return self._with_unary(self._ir.timestamp(time_unit))

    def truncate(self, every: str) -> Expr:
        return self._with_unary(self._ir.truncate(every))

    def offset_by(self, by: str) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.offset_by(by))
