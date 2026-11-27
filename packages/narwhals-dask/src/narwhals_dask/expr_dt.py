from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd

from narwhals.compliant import DateTimeNamespace
from narwhals.utils import parse_version
from narwhals_dask._vendored import (
    ALIAS_DICT,
    MS_PER_SECOND,
    NS_PER_SECOND,
    US_PER_SECOND,
    ExprNamespace,
    Interval,
    calculate_timestamp_date,
    calculate_timestamp_datetime,
    is_dtype_pyarrow,
)
from narwhals_dask.utils import native_to_narwhals_dtype

if TYPE_CHECKING:
    import dask.dataframe.dask_expr as dx

    from narwhals.typing import TimeUnit
    from narwhals_dask.expr import DaskExpr


def _microsecond(expr: dx.Series) -> dx.Series:
    if is_dtype_pyarrow(expr.dtype) and parse_version(pd) < (3,):
        # pandas-dev/pandas#59154: Arrow's microsecond field omits the milliseconds.
        def microsecond_partition(part: pd.Series) -> pd.Series:
            import pyarrow.compute as pc

            arr = cast("pd.arrays.ArrowExtensionArray", part.array).__arrow_array__()
            values = pc.add(pc.multiply(pc.millisecond(arr), 1000), pc.microsecond(arr))
            return pd.Series(
                pd.arrays.ArrowExtensionArray(values), index=part.index, name=part.name
            )

        return expr.map_partitions(
            microsecond_partition, meta=(expr.name, "int64[pyarrow]")
        )
    return expr.dt.microsecond


class DaskExprDateTimeNamespace(ExprNamespace["DaskExpr"], DateTimeNamespace["DaskExpr"]):
    def date(self) -> DaskExpr:
        return self.compliant._with_callable(lambda expr: expr.dt.date)

    def year(self) -> DaskExpr:
        return self.compliant._with_callable(lambda expr: expr.dt.year)

    def month(self) -> DaskExpr:
        return self.compliant._with_callable(lambda expr: expr.dt.month)

    def day(self) -> DaskExpr:
        return self.compliant._with_callable(lambda expr: expr.dt.day)

    def hour(self) -> DaskExpr:
        return self.compliant._with_callable(lambda expr: expr.dt.hour)

    def minute(self) -> DaskExpr:
        return self.compliant._with_callable(lambda expr: expr.dt.minute)

    def second(self) -> DaskExpr:
        return self.compliant._with_callable(lambda expr: expr.dt.second)

    def millisecond(self) -> DaskExpr:
        return self.compliant._with_callable(
            lambda expr: _microsecond(expr).floordiv(1000)
        )

    def microsecond(self) -> DaskExpr:
        return self.compliant._with_callable(_microsecond)

    def nanosecond(self) -> DaskExpr:
        return self.compliant._with_callable(
            lambda expr: _microsecond(expr).mul(1000) + expr.dt.nanosecond
        )

    def ordinal_day(self) -> DaskExpr:
        return self.compliant._with_callable(lambda expr: expr.dt.day_of_year)

    def weekday(self) -> DaskExpr:
        return self.compliant._with_callable(
            lambda expr: expr.dt.day_of_week + 1  # Dask is 0-6
        )

    def to_string(self, format: str) -> DaskExpr:
        def func(expr: dx.Series) -> dx.Series:
            if is_dtype_pyarrow(expr.dtype) and parse_version(pd) < (3, 2):
                # PyArrow interprets `'%S'` as "seconds, plus fractional seconds"
                # and doesn't support `%f`, until pandas 3.2 (pandas-dev/pandas#55001)
                fmt = format.replace("%S.%f", "%S").replace("%S%.f", "%S")
            else:
                fmt = format.replace("%.f", ".%f")
            return expr.dt.strftime(fmt)

        return self.compliant._with_callable(func)

    def replace_time_zone(self, time_zone: str | None) -> DaskExpr:
        return self.compliant._with_callable(
            lambda expr: (
                expr.dt.tz_localize(None).dt.tz_localize(time_zone)
                if time_zone is not None
                else expr.dt.tz_localize(None)
            )
        )

    def convert_time_zone(self, time_zone: str) -> DaskExpr:
        def func(s: dx.Series) -> dx.Series:
            dtype = native_to_narwhals_dtype(s.dtype, self.compliant._version)
            if dtype.time_zone is None:  # type: ignore[attr-defined]
                return s.dt.tz_localize("UTC").dt.tz_convert(time_zone)  # pyright: ignore[reportAttributeAccessIssue]
            return s.dt.tz_convert(time_zone)  # pyright: ignore[reportAttributeAccessIssue]

        return self.compliant._with_callable(func)

    # ignoring coverage due to https://github.com/narwhals-dev/narwhals/issues/2808.
    def timestamp(self, time_unit: TimeUnit) -> DaskExpr:  # pragma: no cover
        def func(s: dx.Series) -> dx.Series:
            dtype = native_to_narwhals_dtype(s.dtype, self.compliant._version)
            is_pyarrow_dtype = "pyarrow" in str(dtype)
            mask_na = s.isna()
            dtypes = self.compliant._version.dtypes
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

        return self.compliant._with_callable(func)

    def total_minutes(self) -> DaskExpr:
        return self.compliant._with_callable(lambda expr: expr.dt.total_seconds() // 60)

    def total_seconds(self) -> DaskExpr:
        return self.compliant._with_callable(lambda expr: expr.dt.total_seconds() // 1)

    def total_milliseconds(self) -> DaskExpr:
        return self.compliant._with_callable(
            lambda expr: expr.dt.total_seconds() * MS_PER_SECOND // 1
        )

    def total_microseconds(self) -> DaskExpr:
        return self.compliant._with_callable(
            lambda expr: expr.dt.total_seconds() * US_PER_SECOND // 1
        )

    def total_nanoseconds(self) -> DaskExpr:
        return self.compliant._with_callable(
            lambda expr: expr.dt.total_seconds() * NS_PER_SECOND // 1
        )

    def truncate(self, every: str) -> DaskExpr:
        interval = Interval.parse(every)
        unit = interval.unit
        if unit in {"mo", "q", "y"}:
            msg = f"Truncating to {unit} is not yet supported for dask."
            raise NotImplementedError(msg)
        freq = f"{interval.multiple}{ALIAS_DICT.get(unit, unit)}"
        return self.compliant._with_callable(lambda expr: expr.dt.floor(freq))

    def offset_by(self, by: str) -> DaskExpr:
        def func(s: dx.Series) -> dx.Series:
            interval = Interval.parse_no_constraints(by)
            unit = interval.unit
            if unit in {"y", "q", "mo", "d", "ns"}:
                msg = f"Offsetting by {unit} is not yet supported for dask."
                raise NotImplementedError(msg)
            offset = interval.to_timedelta()
            return s.add(offset)

        return self.compliant._with_callable(func)
