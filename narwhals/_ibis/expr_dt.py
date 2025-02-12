from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._ibis.expr import IbisExpr


class IbisExprDateTimeNamespace:
    def __init__(self: Self, expr: IbisExpr) -> None:
        self._compliant_expr = expr

    def year(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.year(),
            "year",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def month(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.month(),
            "month",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def day(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.day(),
            "day",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def hour(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.hour(),
            "hour",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def minute(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.minute(),
            "minute",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def second(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.second(),
            "second",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def millisecond(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.millisecond(),
            "millisecond",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def microsecond(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.microsecond(),
            "microsecond",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def nanosecond(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.microsecond() * 1000,
            "nanosecond",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def to_string(self: Self, format: str) -> IbisExpr:  # noqa: A002
        return self._compliant_expr._from_call(
            lambda _input: _input.strftime(format),
            "to_string",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def weekday(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.day_of_week.index()
            + 1,  # Ibis uses 0-6 for Monday-Sunday. Add 1 to match polars.
            "weekday",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def ordinal_day(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.day_of_year(),
            "ordinal_day",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def date(self: Self) -> IbisExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.date(),
            "date",
            expr_kind=self._compliant_expr._expr_kind,
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
