from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._pandas_like.utils import calculate_timestamp_date
from narwhals._pandas_like.utils import calculate_timestamp_datetime
from narwhals._pandas_like.utils import native_to_narwhals_dtype
from narwhals.utils import Implementation

if TYPE_CHECKING:
    import dask.dataframe.dask_expr as dx

    from narwhals._dask.expr import DaskExpr
    from narwhals.typing import TimeUnit


class DaskExprDateTimeNamespace:
    def __init__(self, expr: DaskExpr) -> None:
        self._compliant_expr = expr

    def date(self) -> DaskExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.dt.date, "date")

    def year(self) -> DaskExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.dt.year, "year")

    def month(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.month, "month"
        )

    def day(self) -> DaskExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.dt.day, "day")

    def hour(self) -> DaskExpr:
        return self._compliant_expr._with_callable(lambda _input: _input.dt.hour, "hour")

    def minute(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.minute, "minute"
        )

    def second(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.second, "second"
        )

    def millisecond(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.microsecond // 1000, "millisecond"
        )

    def microsecond(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.microsecond, "microsecond"
        )

    def nanosecond(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.microsecond * 1000 + _input.dt.nanosecond,
            "nanosecond",
        )

    def ordinal_day(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.dayofyear, "ordinal_day"
        )

    def weekday(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.weekday + 1,  # Dask is 0-6
            "weekday",
        )

    def to_string(self, format: str) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input, format: _input.dt.strftime(format.replace("%.f", ".%f")),
            "strftime",
            format=format,
        )

    def replace_time_zone(self, time_zone: str | None) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input, time_zone: _input.dt.tz_localize(None).dt.tz_localize(
                time_zone
            )
            if time_zone is not None
            else _input.dt.tz_localize(None),
            "tz_localize",
            time_zone=time_zone,
        )

    def convert_time_zone(self, time_zone: str) -> DaskExpr:
        def func(s: dx.Series, time_zone: str) -> dx.Series:
            dtype = native_to_narwhals_dtype(
                s.dtype, self._compliant_expr._version, Implementation.DASK
            )
            if dtype.time_zone is None:  # type: ignore[attr-defined]
                return s.dt.tz_localize("UTC").dt.tz_convert(time_zone)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                return s.dt.tz_convert(time_zone)  # pyright: ignore[reportAttributeAccessIssue]

        return self._compliant_expr._with_callable(
            func, "tz_convert", time_zone=time_zone
        )

    def timestamp(self, time_unit: TimeUnit) -> DaskExpr:
        def func(s: dx.Series, time_unit: TimeUnit) -> dx.Series:
            dtype = native_to_narwhals_dtype(
                s.dtype, self._compliant_expr._version, Implementation.DASK
            )
            is_pyarrow_dtype = "pyarrow" in str(dtype)
            mask_na = s.isna()
            dtypes = self._compliant_expr._version.dtypes
            if dtype == dtypes.Date:
                # Date is only supported in pandas dtypes if pyarrow-backed
                s_cast = s.astype("Int32[pyarrow]")
                result = calculate_timestamp_date(s_cast, time_unit)
            elif isinstance(dtype, dtypes.Datetime):
                original_time_unit = dtype.time_unit
                s_cast = (
                    s.astype("Int64[pyarrow]") if is_pyarrow_dtype else s.astype("int64")
                )
                result = calculate_timestamp_datetime(
                    s_cast, original_time_unit, time_unit
                )
            else:
                msg = "Input should be either of Date or Datetime type"
                raise TypeError(msg)
            return result.where(~mask_na)  # pyright: ignore[reportReturnType]

        return self._compliant_expr._with_callable(func, "datetime", time_unit=time_unit)

    def total_minutes(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.total_seconds() // 60, "total_minutes"
        )

    def total_seconds(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.total_seconds() // 1, "total_seconds"
        )

    def total_milliseconds(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.total_seconds() * 1000 // 1, "total_milliseconds"
        )

    def total_microseconds(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.total_seconds() * 1_000_000 // 1,
            "total_microseconds",
        )

    def total_nanoseconds(self) -> DaskExpr:
        return self._compliant_expr._with_callable(
            lambda _input: _input.dt.total_seconds() * 1_000_000_000 // 1,
            "total_nanoseconds",
        )
