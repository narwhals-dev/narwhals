from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc

from narwhals._arrow.utils import ArrowSeriesNamespace
from narwhals._arrow.utils import floordiv_compat
from narwhals._arrow.utils import lit
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import ArrowChunkedArray
    from narwhals.typing import TimeUnit


class ArrowSeriesDateTimeNamespace(ArrowSeriesNamespace):
    def to_string(self: Self, format: str) -> ArrowSeries:  # noqa: A002
        # PyArrow differs from other libraries in that %S also prints out
        # the fractional part of the second...:'(
        # https://arrow.apache.org/docs/python/generated/pyarrow.compute.strftime.html
        format = format.replace("%S.%f", "%S").replace("%S%.f", "%S")
        return self.compliant._from_native_series(pc.strftime(self.native, format))

    def replace_time_zone(self: Self, time_zone: str | None) -> ArrowSeries:
        if time_zone is not None:
            result = pc.assume_timezone(pc.local_timestamp(self.native), time_zone)
        else:
            result = pc.local_timestamp(self.native)
        return self.compliant._from_native_series(result)

    def convert_time_zone(self: Self, time_zone: str) -> ArrowSeries:
        if self.compliant.dtype.time_zone is None:  # type: ignore[attr-defined]
            ser = self.replace_time_zone("UTC")
        else:
            ser = self.compliant
        native_type = pa.timestamp(ser._type.unit, time_zone)  # type: ignore[attr-defined]
        result = ser.native.cast(native_type)
        return self.compliant._from_native_series(result)

    def timestamp(self: Self, time_unit: TimeUnit) -> ArrowSeries:
        ser: ArrowSeries = self.compliant
        dtypes = import_dtypes_module(ser._version)
        if isinstance(ser.dtype, dtypes.Datetime):
            unit = ser.dtype.time_unit
            s_cast = self.native.cast(pa.int64())
            if unit == "ns":
                if time_unit == "ns":
                    result = s_cast
                elif time_unit == "us":
                    result = floordiv_compat(s_cast, 1_000)
                else:
                    result = floordiv_compat(s_cast, 1_000_000)
            elif unit == "us":
                if time_unit == "ns":
                    result = cast("ArrowChunkedArray", pc.multiply(s_cast, 1_000))
                elif time_unit == "us":
                    result = s_cast
                else:
                    result = floordiv_compat(s_cast, 1_000)
            elif unit == "ms":
                if time_unit == "ns":
                    result = cast("ArrowChunkedArray", pc.multiply(s_cast, 1_000_000))
                elif time_unit == "us":
                    result = cast("ArrowChunkedArray", pc.multiply(s_cast, 1_000))
                else:
                    result = s_cast
            elif unit == "s":
                if time_unit == "ns":
                    result = cast("ArrowChunkedArray", pc.multiply(s_cast, 1_000_000_000))
                elif time_unit == "us":
                    result = cast("ArrowChunkedArray", pc.multiply(s_cast, 1_000_000))
                else:
                    result = cast("ArrowChunkedArray", pc.multiply(s_cast, 1_000))
            else:  # pragma: no cover
                msg = f"unexpected time unit {unit}, please report an issue at https://github.com/narwhals-dev/narwhals"
                raise AssertionError(msg)
        elif isinstance(ser.dtype, dtypes.Date):
            time_s = pc.multiply(self.native.cast(pa.int32()), 86400)
            if time_unit == "ns":
                result = cast("ArrowChunkedArray", pc.multiply(time_s, 1_000_000_000))
            elif time_unit == "us":
                result = cast("ArrowChunkedArray", pc.multiply(time_s, 1_000_000))
            else:
                result = cast("ArrowChunkedArray", pc.multiply(time_s, 1_000))
        else:
            msg = "Input should be either of Date or Datetime type"
            raise TypeError(msg)
        return self.compliant._from_native_series(result)

    def date(self: Self) -> ArrowSeries:
        return self.compliant._from_native_series(self.native.cast(pa.date32()))

    def year(self: Self) -> ArrowSeries:
        return self.compliant._from_native_series(pc.year(self.native))

    def month(self: Self) -> ArrowSeries:
        return self.compliant._from_native_series(pc.month(self.native))

    def day(self: Self) -> ArrowSeries:
        return self.compliant._from_native_series(pc.day(self.native))

    def hour(self: Self) -> ArrowSeries:
        return self.compliant._from_native_series(pc.hour(self.native))

    def minute(self: Self) -> ArrowSeries:
        return self.compliant._from_native_series(pc.minute(self.native))

    def second(self: Self) -> ArrowSeries:
        return self.compliant._from_native_series(pc.second(self.native))

    def millisecond(self: Self) -> ArrowSeries:
        return self.compliant._from_native_series(pc.millisecond(self.native))

    def microsecond(self: Self) -> ArrowSeries:
        arr = self.native
        result = pc.add(pc.multiply(pc.millisecond(arr), lit(1000)), pc.microsecond(arr))
        return self.compliant._from_native_series(result)

    def nanosecond(self: Self) -> ArrowSeries:
        result = pc.add(
            pc.multiply(self.microsecond().native, lit(1000)), pc.nanosecond(self.native)
        )
        return self.compliant._from_native_series(result)

    def ordinal_day(self: Self) -> ArrowSeries:
        return self.compliant._from_native_series(pc.day_of_year(self.native))

    def weekday(self: Self) -> ArrowSeries:
        return self.compliant._from_native_series(
            pc.day_of_week(self.native, count_from_zero=False)
        )

    def total_minutes(self: Self) -> ArrowSeries:
        unit_to_minutes_factor = {
            "s": 60,  # seconds
            "ms": 60 * 1e3,  # milli
            "us": 60 * 1e6,  # micro
            "ns": 60 * 1e9,  # nano
        }
        unit = self.compliant._type.unit  # type: ignore[attr-defined]
        factor = lit(unit_to_minutes_factor[unit], type=pa.int64())
        return self.compliant._from_native_series(
            pc.divide(self.native, factor).cast(pa.int64())
        )

    def total_seconds(self: Self) -> ArrowSeries:
        unit_to_seconds_factor = {
            "s": 1,  # seconds
            "ms": 1e3,  # milli
            "us": 1e6,  # micro
            "ns": 1e9,  # nano
        }
        unit = self.compliant._type.unit  # type: ignore[attr-defined]
        factor = lit(unit_to_seconds_factor[unit], type=pa.int64())
        return self.compliant._from_native_series(
            pc.divide(self.native, factor).cast(pa.int64())
        )

    def total_milliseconds(self: Self) -> ArrowSeries:
        unit = self.compliant._type.unit  # type: ignore[attr-defined]
        unit_to_milli_factor = {
            "s": 1e3,  # seconds
            "ms": 1,  # milli
            "us": 1e3,  # micro
            "ns": 1e6,  # nano
        }
        factor = lit(unit_to_milli_factor[unit], type=pa.int64())
        if unit == "s":
            return self.compliant._from_native_series(
                pc.multiply(self.native, factor).cast(pa.int64())
            )
        return self.compliant._from_native_series(
            pc.divide(self.native, factor).cast(pa.int64())
        )

    def total_microseconds(self: Self) -> ArrowSeries:
        arr = self.native
        unit = self.compliant._type.unit  # type: ignore[attr-defined]
        unit_to_micro_factor = {
            "s": 1e6,  # seconds
            "ms": 1e3,  # milli
            "us": 1,  # micro
            "ns": 1e3,  # nano
        }
        factor = lit(unit_to_micro_factor[unit], type=pa.int64())
        if unit in {"s", "ms"}:
            return self.compliant._from_native_series(
                pc.multiply(arr, factor).cast(pa.int64())
            )
        return self.compliant._from_native_series(pc.divide(arr, factor).cast(pa.int64()))

    def total_nanoseconds(self: Self) -> ArrowSeries:
        unit_to_nano_factor = {
            "s": 1e9,  # seconds
            "ms": 1e6,  # milli
            "us": 1e3,  # micro
            "ns": 1,  # nano
        }

        unit = self.compliant._type.unit  # type: ignore[attr-defined]
        factor = lit(unit_to_nano_factor[unit], type=pa.int64())
        return self.compliant._from_native_series(
            pc.multiply(self.native, factor).cast(pa.int64())
        )
