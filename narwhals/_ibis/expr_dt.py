from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._ibis.expr import IbisExpr


class IbisExprDateTimeNamespace:
    def __init__(self: Self, expr: IbisExpr) -> None:
        self._compliant_expr = expr

    def year(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.year(),
        )

    def month(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.month(),
        )

    def day(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.day(),
        )

    def hour(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.hour(),
        )

    def minute(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.minute(),
        )

    def second(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.second(),
        )

    def millisecond(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.millisecond(),
        )

    def microsecond(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.microsecond(),
        )

    def nanosecond(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.microsecond() * 1000,
        )

    def to_string(self: Self, format: str) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.strftime(format),
        )

    def weekday(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.day_of_week.index()
            + 1,  # Ibis uses 0-6 for Monday-Sunday. Add 1 to match polars.
        )

    def ordinal_day(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.day_of_year(),
        )

    def date(self: Self) -> IbisExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.date(),
        )

    def total_minutes(self: Self) -> IbisExpr:
        msg = "`total_minutes` is not implemented for Ibis"
        raise NotImplementedError(msg)

    def total_seconds(self: Self) -> IbisExpr:
        msg = "`total_seconds` is not implemented for Ibis"
        raise NotImplementedError(msg)

    def total_milliseconds(self: Self) -> IbisExpr:
        msg = "`total_milliseconds` is not implemented for Ibis"
        raise NotImplementedError(msg)

    def total_microseconds(self: Self) -> IbisExpr:
        msg = "`total_microseconds` is not implemented for Ibis"
        raise NotImplementedError(msg)

    def total_nanoseconds(self: Self) -> IbisExpr:
        msg = "`total_nanoseconds` is not implemented for Ibis"
        raise NotImplementedError(msg)
