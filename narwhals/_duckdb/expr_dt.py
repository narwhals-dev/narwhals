from __future__ import annotations

from typing import TYPE_CHECKING

from duckdb import FunctionExpression

from narwhals._duckdb.utils import lit

if TYPE_CHECKING:
    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprDateTimeNamespace:
    def __init__(self, expr: DuckDBExpr) -> None:
        self._compliant_expr = expr

    def year(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("year", expr)
        )

    def month(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("month", expr)
        )

    def day(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("day", expr)
        )

    def hour(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("hour", expr)
        )

    def minute(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("minute", expr)
        )

    def second(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("second", expr)
        )

    def millisecond(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("millisecond", expr)
            - FunctionExpression("second", expr) * lit(1_000),
        )

    def microsecond(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("microsecond", expr)
            - FunctionExpression("second", expr) * lit(1_000_000),
        )

    def nanosecond(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("nanosecond", expr)
            - FunctionExpression("second", expr) * lit(1_000_000_000),
        )

    def to_string(self, format: str) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("strftime", expr, lit(format)),
        )

    def weekday(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("isodow", expr)
        )

    def ordinal_day(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("dayofyear", expr)
        )

    def date(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.cast("date"))

    def total_minutes(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("datepart", lit("minute"), expr),
        )

    def total_seconds(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: lit(60) * FunctionExpression("datepart", lit("minute"), expr)
            + FunctionExpression("datepart", lit("second"), expr),
        )

    def total_milliseconds(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: lit(60_000) * FunctionExpression("datepart", lit("minute"), expr)
            + FunctionExpression("datepart", lit("millisecond"), expr),
        )

    def total_microseconds(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: lit(60_000_000)
            * FunctionExpression("datepart", lit("minute"), expr)
            + FunctionExpression("datepart", lit("microsecond"), expr),
        )

    def total_nanoseconds(self) -> DuckDBExpr:
        msg = "`total_nanoseconds` is not implemented for DuckDB"
        raise NotImplementedError(msg)
