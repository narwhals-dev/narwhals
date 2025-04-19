from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc

from narwhals._arrow.utils import ArrowSeriesNamespace
from narwhals._arrow.utils import floordiv_compat
from narwhals._arrow.utils import lit
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    from narwhals._arrow.series import ArrowSeries
    from narwhals.dtypes import Datetime
    from narwhals.typing import TimeUnit


class ArrowSeriesDateTimeNamespace(ArrowSeriesNamespace):
    @property
    def unit(self) -> TimeUnit:  # NOTE: Unsafe (native).
        return cast("pa.TimestampType[TimeUnit, Any]", self.native.type).unit

    @property
    def time_zone(self) -> str | None:  # NOTE: Unsafe (narwhals).
        return cast("Datetime", self.compliant.dtype).time_zone

    def to_string(self, format: str) -> ArrowSeries:
        # PyArrow differs from other libraries in that %S also prints out
        # the fractional part of the second...:'(
        # https://arrow.apache.org/docs/python/generated/pyarrow.compute.strftime.html
        format = format.replace("%S.%f", "%S").replace("%S%.f", "%S")
        return self.with_native(pc.strftime(self.native, format))

    def replace_time_zone(self, time_zone: str | None) -> ArrowSeries:
        if time_zone is not None:
            result = pc.assume_timezone(pc.local_timestamp(self.native), time_zone)
        else:
            result = pc.local_timestamp(self.native)
        return self.with_native(result)

    def convert_time_zone(self, time_zone: str) -> ArrowSeries:
        ser = self.replace_time_zone("UTC") if self.time_zone is None else self.compliant
        return self.with_native(ser.native.cast(pa.timestamp(self.unit, time_zone)))

    def timestamp(self, time_unit: TimeUnit) -> ArrowSeries:
        ser = self.compliant
        dtypes = import_dtypes_module(ser._version)
        if isinstance(ser.dtype, dtypes.Datetime):
            unit = ser.dtype.time_unit
            s_cast = self.native.cast(pa.int64())
            if unit == "ns":
                if time_unit == "ns":
                    result_64 = s_cast
                elif time_unit == "us":
                    result_64 = floordiv_compat(s_cast, lit(1_000))
                else:
                    result_64 = floordiv_compat(s_cast, lit(1_000_000))
            elif unit == "us":
                if time_unit == "ns":
                    result_64 = pc.multiply(s_cast, lit(1_000))
                elif time_unit == "us":
                    result_64 = s_cast
                else:
                    result_64 = floordiv_compat(s_cast, lit(1_000))
            elif unit == "ms":
                if time_unit == "ns":
                    result_64 = pc.multiply(s_cast, lit(1_000_000))
                elif time_unit == "us":
                    result_64 = pc.multiply(s_cast, lit(1_000))
                else:
                    result_64 = s_cast
            elif unit == "s":
                if time_unit == "ns":
                    result_64 = pc.multiply(s_cast, lit(1_000_000_000))
                elif time_unit == "us":
                    result_64 = pc.multiply(s_cast, lit(1_000_000))
                else:
                    result_64 = pc.multiply(s_cast, lit(1_000))
            else:  # pragma: no cover
                msg = f"unexpected time unit {unit}, please report an issue at https://github.com/narwhals-dev/narwhals"
                raise AssertionError(msg)
            return self.with_native(result_64)
        elif isinstance(ser.dtype, dtypes.Date):
            time_s = pc.multiply(self.native.cast(pa.int32()), lit(86_400))
            if time_unit == "ns":
                result_32 = pc.multiply(time_s, lit(1_000_000_000))
            elif time_unit == "us":
                result_32 = pc.multiply(time_s, lit(1_000_000))
            else:
                result_32 = pc.multiply(time_s, lit(1_000))
            return self.with_native(result_32)
        else:
            msg = "Input should be either of Date or Datetime type"
            raise TypeError(msg)

    def date(self) -> ArrowSeries:
        return self.with_native(self.native.cast(pa.date32()))

    def year(self) -> ArrowSeries:
        return self.with_native(pc.year(self.native))

    def month(self) -> ArrowSeries:
        return self.with_native(pc.month(self.native))

    def day(self) -> ArrowSeries:
        return self.with_native(pc.day(self.native))

    def hour(self) -> ArrowSeries:
        return self.with_native(pc.hour(self.native))

    def minute(self) -> ArrowSeries:
        return self.with_native(pc.minute(self.native))

    def second(self) -> ArrowSeries:
        return self.with_native(pc.second(self.native))

    def millisecond(self) -> ArrowSeries:
        return self.with_native(pc.millisecond(self.native))

    def microsecond(self) -> ArrowSeries:
        arr = self.native
        result = pc.add(pc.multiply(pc.millisecond(arr), lit(1000)), pc.microsecond(arr))
        return self.with_native(result)

    def nanosecond(self) -> ArrowSeries:
        result = pc.add(
            pc.multiply(self.microsecond().native, lit(1000)), pc.nanosecond(self.native)
        )
        return self.with_native(result)

    def ordinal_day(self) -> ArrowSeries:
        return self.with_native(pc.day_of_year(self.native))

    def weekday(self) -> ArrowSeries:
        return self.with_native(pc.day_of_week(self.native, count_from_zero=False))

    def total_minutes(self) -> ArrowSeries:
        unit_to_minutes_factor = {
            "s": 60,  # seconds
            "ms": 60 * 1e3,  # milli
            "us": 60 * 1e6,  # micro
            "ns": 60 * 1e9,  # nano
        }
        factor = lit(unit_to_minutes_factor[self.unit], type=pa.int64())
        return self.with_native(pc.divide(self.native, factor).cast(pa.int64()))

    def total_seconds(self) -> ArrowSeries:
        unit_to_seconds_factor = {
            "s": 1,  # seconds
            "ms": 1e3,  # milli
            "us": 1e6,  # micro
            "ns": 1e9,  # nano
        }
        factor = lit(unit_to_seconds_factor[self.unit], type=pa.int64())
        return self.with_native(pc.divide(self.native, factor).cast(pa.int64()))

    def total_milliseconds(self) -> ArrowSeries:
        unit_to_milli_factor = {
            "s": 1e3,  # seconds
            "ms": 1,  # milli
            "us": 1e3,  # micro
            "ns": 1e6,  # nano
        }
        factor = lit(unit_to_milli_factor[self.unit], type=pa.int64())
        if self.unit == "s":
            return self.with_native(pc.multiply(self.native, factor).cast(pa.int64()))
        return self.with_native(pc.divide(self.native, factor).cast(pa.int64()))

    def total_microseconds(self) -> ArrowSeries:
        unit_to_micro_factor = {
            "s": 1e6,  # seconds
            "ms": 1e3,  # milli
            "us": 1,  # micro
            "ns": 1e3,  # nano
        }
        factor = lit(unit_to_micro_factor[self.unit], type=pa.int64())
        if self.unit in {"s", "ms"}:
            return self.with_native(pc.multiply(self.native, factor).cast(pa.int64()))
        return self.with_native(pc.divide(self.native, factor).cast(pa.int64()))

    def total_nanoseconds(self) -> ArrowSeries:
        unit_to_nano_factor = {
            "s": 1e9,  # seconds
            "ms": 1e6,  # milli
            "us": 1e3,  # micro
            "ns": 1,  # nano
        }
        factor = lit(unit_to_nano_factor[self.unit], type=pa.int64())
        return self.with_native(pc.multiply(self.native, factor).cast(pa.int64()))
