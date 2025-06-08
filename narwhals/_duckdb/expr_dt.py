from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from duckdb import FunctionExpression

from narwhals._duckdb.utils import UNITS_DICT, fetch_rel_time_zone, lit
from narwhals._duration import parse_interval_string
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from duckdb import Expression

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
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
            - FunctionExpression("second", expr) * lit(1_000)
        )

    def microsecond(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("microsecond", expr)
            - FunctionExpression("second", expr) * lit(1_000_000)
        )

    def nanosecond(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("nanosecond", expr)
            - FunctionExpression("second", expr) * lit(1_000_000_000)
        )

    def to_string(self, format: str) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("strftime", expr, lit(format))
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
            lambda expr: FunctionExpression("datepart", lit("minute"), expr)
        )

    def total_seconds(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: lit(60) * FunctionExpression("datepart", lit("minute"), expr)
            + FunctionExpression("datepart", lit("second"), expr)
        )

    def total_milliseconds(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: lit(60_000) * FunctionExpression("datepart", lit("minute"), expr)
            + FunctionExpression("datepart", lit("millisecond"), expr)
        )

    def total_microseconds(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: lit(60_000_000)
            * FunctionExpression("datepart", lit("minute"), expr)
            + FunctionExpression("datepart", lit("microsecond"), expr)
        )

    def truncate(self, every: str) -> DuckDBExpr:
        multiple, unit = parse_interval_string(every)
        if multiple != 1:
            # https://github.com/duckdb/duckdb/issues/17554
            msg = f"Only multiple 1 is currently supported for DuckDB.\nGot {multiple!s}."
            raise ValueError(msg)
        if unit == "ns":
            msg = "Truncating to nanoseconds is not yet supported for DuckDB."
            raise NotImplementedError(msg)
        format = lit(UNITS_DICT[unit])

        def _truncate(expr: Expression) -> Expression:
            return FunctionExpression("date_trunc", format, expr)

        return self._compliant_expr._with_callable(_truncate)

    def _no_op_time_zone(self, time_zone: str) -> DuckDBExpr:
        def func(df: DuckDBLazyFrame) -> Sequence[Expression]:
            native_series_list = self._compliant_expr(df)
            conn_time_zone = fetch_rel_time_zone(df.native)
            if conn_time_zone != time_zone:
                msg = (
                    "DuckDB stores the time zone in the connection, rather than in the "
                    f"data type, so changing the timezone to anything other than {conn_time_zone} "
                    " (the current connection time zone) is not supported."
                )
                raise NotImplementedError(msg)
            return native_series_list

        return self._compliant_expr.__class__(
            func,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=self._compliant_expr._alias_output_names,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
        )

    def convert_time_zone(self, time_zone: str) -> DuckDBExpr:
        return self._no_op_time_zone(time_zone)

    def replace_time_zone(self, time_zone: str | None) -> DuckDBExpr:
        if time_zone is None:
            return self._compliant_expr._with_callable(
                lambda _input: _input.cast("timestamp")
            )
        else:
            return self._no_op_time_zone(time_zone)

    total_nanoseconds = not_implemented()
