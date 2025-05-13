from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from narwhals._duration import parse_interval_string
from narwhals._ibis.utils import UNITS_DICT_BUCKET
from narwhals._ibis.utils import UNITS_DICT_TRUNCATE
from narwhals.utils import not_implemented

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from narwhals._ibis.expr import IbisExpr
    from narwhals._ibis.utils import BucketUnit
    from narwhals._ibis.utils import TruncateUnit


class IbisExprDateTimeNamespace:
    def __init__(self, expr: IbisExpr) -> None:
        self._compliant_expr = expr

    def year(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.year())

    def month(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.month())

    def day(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.day())

    def hour(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.hour())

    def minute(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.minute())

    def second(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.second())

    def millisecond(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.millisecond())

    def microsecond(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.microsecond())

    def to_string(self, format: str) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.strftime(format))

    def weekday(self) -> IbisExpr:
        # Ibis uses 0-6 for Monday-Sunday. Add 1 to match polars.
        return self._compliant_expr._with_callable(
            lambda _input: _input.day_of_week.index() + 1
        )

    def ordinal_day(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.day_of_year())

    def date(self) -> IbisExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.date())

    def _bucket(self, kwds: dict[BucketUnit, Any], /) -> Callable[..., ir.TimestampValue]:
        def fn(_input: ir.TimestampValue) -> ir.TimestampValue:
            return _input.bucket(**kwds)

        return fn

    def _truncate(self, unit: TruncateUnit, /) -> Callable[..., ir.TimestampValue]:
        def fn(_input: ir.TimestampValue) -> ir.TimestampValue:
            return _input.truncate(unit)

        return fn

    def truncate(self, every: str) -> IbisExpr:
        multiple, unit = parse_interval_string(every)
        if multiple != 1:
            if self._compliant_expr._backend_version < (7, 1):  # pragma: no cover
                msg = "Truncating datetimes with multiples of the unit is only supported in Ibis >= 7.1."
                raise NotImplementedError(msg)
            fn = self._bucket({UNITS_DICT_BUCKET[unit]: multiple})
        else:
            fn = self._truncate(UNITS_DICT_TRUNCATE[unit])
        return self._compliant_expr._with_callable(fn)

    nanosecond = not_implemented()
    total_minutes = not_implemented()
    total_seconds = not_implemented()
    total_milliseconds = not_implemented()
    total_microseconds = not_implemented()
    total_nanoseconds = not_implemented()
