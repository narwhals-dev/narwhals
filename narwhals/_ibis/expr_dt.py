from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from narwhals._duration import parse_interval_string
from narwhals._ibis.utils import UNITS_DICT_BUCKET, UNITS_DICT_TRUNCATE
from narwhals.utils import not_implemented

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from narwhals._ibis.expr import IbisExpr
    from narwhals._ibis.utils import BucketUnit, TruncateUnit


class IbisExprDateTimeNamespace:
    def __init__(self, expr: IbisExpr) -> None:
        self._compliant_expr = expr

    def year(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.year())

    def month(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.month())

    def day(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.day())

    def hour(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.hour())

    def minute(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.minute())

    def second(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.second())

    def millisecond(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.millisecond())

    def microsecond(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.microsecond())

    def to_string(self, format: str) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.strftime(format))

    def weekday(self) -> IbisExpr:
        # Ibis uses 0-6 for Monday-Sunday. Add 1 to match polars.
        return self._compliant_expr._with_callable(
            lambda expr: expr.day_of_week.index() + 1
        )

    def ordinal_day(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.day_of_year())

    def date(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda expr: expr.date())

    def _bucket(self, kwds: dict[BucketUnit, Any], /) -> Callable[..., ir.TimestampValue]:
        def fn(expr: ir.TimestampValue) -> ir.TimestampValue:
            return expr.bucket(**kwds)

        return fn

    def _truncate(self, unit: TruncateUnit, /) -> Callable[..., ir.TimestampValue]:
        def fn(expr: ir.TimestampValue) -> ir.TimestampValue:
            return expr.truncate(unit)

        return fn

    def truncate(self, every: str) -> IbisExpr:
        multiple, unit = parse_interval_string(every)
        if unit == "q":
            multiple, unit = 3 * multiple, "mo"
        if multiple != 1:
            if self._compliant_expr._backend_version < (7, 1):  # pragma: no cover
                msg = "Truncating datetimes with multiples of the unit is only supported in Ibis >= 7.1."
                raise NotImplementedError(msg)
            fn = self._bucket({UNITS_DICT_BUCKET[unit]: multiple})
        else:
            fn = self._truncate(UNITS_DICT_TRUNCATE[unit])
        return self._compliant_expr._with_callable(fn)

    def replace_time_zone(self, time_zone: str | None) -> IbisExpr:
        if time_zone is None:
            return self._compliant_expr._with_callable(
                lambda _input: _input.cast("timestamp")
            )
        else:  # pragma: no cover
            msg = "`replace_time_zone` with non-null `time_zone` not yet implemented for Ibis"
            raise NotImplementedError(msg)

    nanosecond = not_implemented()
    total_minutes = not_implemented()
    total_seconds = not_implemented()
    total_milliseconds = not_implemented()
    total_microseconds = not_implemented()
    total_nanoseconds = not_implemented()
