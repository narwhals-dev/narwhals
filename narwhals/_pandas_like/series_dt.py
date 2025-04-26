from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._compliant.any_namespace import DateTimeNamespace
from narwhals._pandas_like.utils import PandasLikeSeriesNamespace
from narwhals._pandas_like.utils import calculate_timestamp_date
from narwhals._pandas_like.utils import calculate_timestamp_datetime
from narwhals._pandas_like.utils import int_dtype_mapper
from narwhals._pandas_like.utils import is_pyarrow_dtype_backend

if TYPE_CHECKING:
    from narwhals._pandas_like.series import PandasLikeSeries
    from narwhals.typing import TimeUnit


class PandasLikeSeriesDateTimeNamespace(
    PandasLikeSeriesNamespace, DateTimeNamespace["PandasLikeSeries"]
):
    def date(self) -> PandasLikeSeries:
        result = self.with_native(self.native.dt.date)
        if str(result.dtype).lower() == "object":
            msg = (
                "Accessing `date` on the default pandas backend "
                "will return a Series of type `object`."
                "\nThis differs from polars API and will prevent `.dt` chaining. "
                "Please switch to the `pyarrow` backend:"
                '\ndf.convert_dtypes(dtype_backend="pyarrow")'
            )
            raise NotImplementedError(msg)
        return result

    def year(self) -> PandasLikeSeries:
        return self.with_native(self.native.dt.year)

    def month(self) -> PandasLikeSeries:
        return self.with_native(self.native.dt.month)

    def day(self) -> PandasLikeSeries:
        return self.with_native(self.native.dt.day)

    def hour(self) -> PandasLikeSeries:
        return self.with_native(self.native.dt.hour)

    def minute(self) -> PandasLikeSeries:
        return self.with_native(self.native.dt.minute)

    def second(self) -> PandasLikeSeries:
        return self.with_native(self.native.dt.second)

    def millisecond(self) -> PandasLikeSeries:
        return self.microsecond() // 1000

    def microsecond(self) -> PandasLikeSeries:
        if self.backend_version < (3, 0, 0) and self._is_pyarrow():
            # crazy workaround for https://github.com/pandas-dev/pandas/issues/59154
            import pyarrow.compute as pc  # ignore-banned-import()

            from narwhals._arrow.utils import lit

            arr_ns = self.native.array
            arr = arr_ns.__arrow_array__()
            result_arr = pc.add(
                pc.multiply(pc.millisecond(arr), lit(1_000)), pc.microsecond(arr)
            )
            result = type(self.native)(type(arr_ns)(result_arr), name=self.native.name)
            return self.with_native(result)

        return self.with_native(self.native.dt.microsecond)

    def nanosecond(self) -> PandasLikeSeries:
        return self.microsecond() * 1_000 + self.native.dt.nanosecond

    def ordinal_day(self) -> PandasLikeSeries:
        year_start = self.native.dt.year
        result = (
            self.native.to_numpy().astype("datetime64[D]")
            - (year_start.to_numpy() - 1970).astype("datetime64[Y]")
        ).astype("int32") + 1
        dtype = "Int64[pyarrow]" if self._is_pyarrow() else "int32"
        return self.with_native(
            type(self.native)(result, dtype=dtype, name=year_start.name)
        )

    def weekday(self) -> PandasLikeSeries:
        # Pandas is 0-6 while Polars is 1-7
        return self.with_native(self.native.dt.weekday) + 1

    def _is_pyarrow(self) -> bool:
        return is_pyarrow_dtype_backend(self.native.dtype, self.implementation)

    def _get_total_seconds(self) -> Any:
        if hasattr(self.native.dt, "total_seconds"):
            return self.native.dt.total_seconds()
        else:  # pragma: no cover
            return (
                self.native.dt.days * 86400
                + self.native.dt.seconds
                + (self.native.dt.microseconds / 1e6)
                + (self.native.dt.nanoseconds / 1e9)
            )

    def total_minutes(self) -> PandasLikeSeries:
        s = self._get_total_seconds()
        # this calculates the sign of each series element
        s_sign = 2 * (s > 0).astype(int_dtype_mapper(s.dtype)) - 1
        s_abs = s.abs() // 60
        if ~s.isna().any():
            s_abs = s_abs.astype(int_dtype_mapper(s.dtype))
        return self.with_native(s_abs * s_sign)

    def total_seconds(self) -> PandasLikeSeries:
        s = self._get_total_seconds()
        # this calculates the sign of each series element
        s_sign = 2 * (s > 0).astype(int_dtype_mapper(s.dtype)) - 1
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int_dtype_mapper(s.dtype))
        return self.with_native(s_abs * s_sign)

    def total_milliseconds(self) -> PandasLikeSeries:
        s = self._get_total_seconds() * 1e3
        # this calculates the sign of each series element
        s_sign = 2 * (s > 0).astype(int_dtype_mapper(s.dtype)) - 1
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int_dtype_mapper(s.dtype))
        return self.with_native(s_abs * s_sign)

    def total_microseconds(self) -> PandasLikeSeries:
        s = self._get_total_seconds() * 1e6
        # this calculates the sign of each series element
        s_sign = 2 * (s > 0).astype(int_dtype_mapper(s.dtype)) - 1
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int_dtype_mapper(s.dtype))
        return self.with_native(s_abs * s_sign)

    def total_nanoseconds(self) -> PandasLikeSeries:
        s = self._get_total_seconds() * 1e9
        # this calculates the sign of each series element
        s_sign = 2 * (s > 0).astype(int_dtype_mapper(s.dtype)) - 1
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int_dtype_mapper(s.dtype))
        return self.with_native(s_abs * s_sign)

    def to_string(self, format: str) -> PandasLikeSeries:
        # Polars' parser treats `'%.f'` as pandas does `'.%f'`
        # PyArrow interprets `'%S'` as "seconds, plus fractional seconds"
        # and doesn't support `%f`
        if not self._is_pyarrow():
            format = format.replace("%S%.f", "%S.%f")
        else:
            format = format.replace("%S.%f", "%S").replace("%S%.f", "%S")
        return self.with_native(self.native.dt.strftime(format))

    def replace_time_zone(self, time_zone: str | None) -> PandasLikeSeries:
        de_zone = self.native.dt.tz_localize(None)
        result = de_zone.dt.tz_localize(time_zone) if time_zone is not None else de_zone
        return self.with_native(result)

    def convert_time_zone(self, time_zone: str) -> PandasLikeSeries:
        if self.compliant.dtype.time_zone is None:  # type: ignore[attr-defined]
            result = self.native.dt.tz_localize("UTC").dt.tz_convert(time_zone)
        else:
            result = self.native.dt.tz_convert(time_zone)
        return self.with_native(result)

    def timestamp(self, time_unit: TimeUnit) -> PandasLikeSeries:
        s = self.native
        dtype = self.compliant.dtype
        mask_na = s.isna()
        dtypes = self.version.dtypes
        if dtype == dtypes.Date:
            # Date is only supported in pandas dtypes if pyarrow-backed
            s_cast = s.astype("Int32[pyarrow]")
            result = calculate_timestamp_date(s_cast, time_unit)
        elif isinstance(dtype, dtypes.Datetime):
            fn = (
                s.view
                if (self.implementation.is_pandas() and self.backend_version < (2,))
                else s.astype
            )
            s_cast = fn("Int64[pyarrow]") if self._is_pyarrow() else fn("int64")
            result = calculate_timestamp_datetime(s_cast, dtype.time_unit, time_unit)
        else:
            msg = "Input should be either of Date or Datetime type"
            raise TypeError(msg)
        result[mask_na] = None
        return self.with_native(result)
