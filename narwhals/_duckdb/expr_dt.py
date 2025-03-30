from __future__ import annotations

from typing import TYPE_CHECKING

from duckdb import FunctionExpression

from narwhals._duckdb.utils import lit

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprDateTimeNamespace:
    def __init__(self: Self, expr: DuckDBExpr) -> None:
        self._compliant_expr = expr

    def year(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("year", _input)
        )

    def month(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("month", _input)
        )

    def day(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("day", _input)
        )

    def hour(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("hour", _input)
        )

    def minute(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("minute", _input)
        )

    def second(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("second", _input)
        )

    def millisecond(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("millisecond", _input)
            - FunctionExpression("second", _input) * lit(1_000),
        )

    def microsecond(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("microsecond", _input)
            - FunctionExpression("second", _input) * lit(1_000_000),
        )

    def nanosecond(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("nanosecond", _input)
            - FunctionExpression("second", _input) * lit(1_000_000_000),
        )

    def to_string(self: Self, format: str) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("strftime", _input, lit(format)),
        )

    def weekday(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("isodow", _input)
        )

    def ordinal_day(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("dayofyear", _input)
        )

    def date(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.cast("date"))

    def total_minutes(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: FunctionExpression("datepart", lit("minute"), _input),
        )

    def total_seconds(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: lit(60) * FunctionExpression("datepart", lit("minute"), _input)
            + FunctionExpression("datepart", lit("second"), _input),
        )

    def total_milliseconds(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: lit(60_000)
            * FunctionExpression("datepart", lit("minute"), _input)
            + FunctionExpression("datepart", lit("millisecond"), _input),
        )

    def total_microseconds(self: Self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda _input: lit(60_000_000)
            * FunctionExpression("datepart", lit("minute"), _input)
            + FunctionExpression("datepart", lit("microsecond"), _input),
        )

    def total_nanoseconds(self: Self) -> DuckDBExpr:
        msg = "`total_nanoseconds` is not implemented for DuckDB"
        raise NotImplementedError(msg)
