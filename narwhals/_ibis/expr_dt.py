from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._duration import Interval
from narwhals._ibis.utils import timedelta_to_ibis_interval
from narwhals._sql.expr_dt import SQLExprDateTimeNamesSpace
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from narwhals._ibis.expr import IbisExpr


class IbisExprDateTimeNamespace(SQLExprDateTimeNamesSpace["IbisExpr"]):
    def millisecond(self) -> IbisExpr:
        return self.compliant._with_callable(lambda expr: expr.millisecond())

    def microsecond(self) -> IbisExpr:
        return self.compliant._with_callable(lambda expr: expr.microsecond())

    def to_string(self, format: str) -> IbisExpr:
        return self.compliant._with_callable(lambda expr: expr.strftime(format))

    def weekday(self) -> IbisExpr:
        # Ibis uses 0-6 for Monday-Sunday. Add 1 to match polars.
        return self.compliant._with_callable(lambda expr: expr.day_of_week.index() + 1)

    def offset_by(self, by: str) -> IbisExpr:
        interval = Interval.parse_no_constraints(by)
        unit = interval.unit
        if unit in {"y", "q", "mo", "d", "ns"}:
            msg = f"Offsetting by {unit} is not yet supported for ibis."
            raise NotImplementedError(msg)
        offset = timedelta_to_ibis_interval(interval.to_timedelta())
        return self.compliant._with_callable(lambda expr: expr.add(offset))

    def replace_time_zone(self, time_zone: str | None) -> IbisExpr:
        if time_zone is None:
            return self.compliant._with_callable(lambda expr: expr.cast("timestamp"))
        msg = "`replace_time_zone` with non-null `time_zone` not yet implemented for Ibis"  # pragma: no cover
        raise NotImplementedError(msg)

    nanosecond = not_implemented()
    total_minutes = not_implemented()
    total_seconds = not_implemented()
    total_milliseconds = not_implemented()
    total_microseconds = not_implemented()
    total_nanoseconds = not_implemented()
    convert_time_zone = not_implemented()
    timestamp = not_implemented()
