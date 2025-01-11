from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Literal
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.series import Series

SeriesT = TypeVar("SeriesT", bound="Series[Any]")


class SeriesDateTimeNamespace(Generic[SeriesT]):
    def __init__(self: Self, series: SeriesT) -> None:
        self._narwhals_series = series

    def date(self: Self) -> SeriesT:
        """Get the date in a datetime series.

        Returns:
            A new Series with the date portion of the datetime values.

        Raises:
            NotImplementedError: If pandas default backend is being used.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.date()
        )

    def year(self: Self) -> SeriesT:
        """Get the year in a datetime series.

        Returns:
            A new Series containing the year component of each datetime value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.year()
        )

    def month(self: Self) -> SeriesT:
        """Gets the month in a datetime series.

        Returns:
            A new Series containing the month component of each datetime value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.month()
        )

    def day(self: Self) -> SeriesT:
        """Extracts the day in a datetime series.

        Returns:
            A new Series containing the day component of each datetime value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.day()
        )

    def hour(self: Self) -> SeriesT:
        """Extracts the hour in a datetime series.

        Returns:
            A new Series containing the hour component of each datetime value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.hour()
        )

    def minute(self: Self) -> SeriesT:
        """Extracts the minute in a datetime series.

        Returns:
            A new Series containing the minute component of each datetime value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.minute()
        )

    def second(self: Self) -> SeriesT:
        """Extracts the seconds in a datetime series.

        Returns:
            A new Series containing the second component of each datetime value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.second()
        )

    def millisecond(self: Self) -> SeriesT:
        """Extracts the milliseconds in a datetime series.

        Returns:
            A new Series containing the millisecond component of each datetime value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.millisecond()
        )

    def microsecond(self: Self) -> SeriesT:
        """Extracts the microseconds in a datetime series.

        Returns:
            A new Series containing the microsecond component of each datetime value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.microsecond()
        )

    def nanosecond(self: Self) -> SeriesT:
        """Extract the nanoseconds in a date series.

        Returns:
            A new Series containing the nanosecond component of each datetime value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.nanosecond()
        )

    def ordinal_day(self: Self) -> SeriesT:
        """Get ordinal day.

        Returns:
            A new Series containing the ordinal day (day of year) for each datetime value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.ordinal_day()
        )

    def weekday(self: Self) -> SeriesT:
        """Extract the week day in a datetime series.

        Returns:
            A new Series containing the week day for each datetime value.
            Returns the ISO weekday number where monday = 1 and sunday = 7

        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.weekday()
        )

    def total_minutes(self: Self) -> SeriesT:
        """Get total minutes.

        Notes:
            The function outputs the total minutes in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Returns:
            A new Series containing the total number of minutes for each timedelta value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.total_minutes()
        )

    def total_seconds(self: Self) -> SeriesT:
        """Get total seconds.

        Notes:
            The function outputs the total seconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Returns:
            A new Series containing the total number of seconds for each timedelta value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.total_seconds()
        )

    def total_milliseconds(self: Self) -> SeriesT:
        """Get total milliseconds.

        Notes:
            The function outputs the total milliseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Returns:
            A new Series containing the total number of milliseconds for each timedelta value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.total_milliseconds()
        )

    def total_microseconds(self: Self) -> SeriesT:
        """Get total microseconds.

        Returns:
            A new Series containing the total number of microseconds for each timedelta value.

        Notes:
            The function outputs the total microseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.total_microseconds()
        )

    def total_nanoseconds(self: Self) -> SeriesT:
        """Get total nanoseconds.

        Notes:
            The function outputs the total nanoseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Returns:
            A new Series containing the total number of nanoseconds for each timedelta value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.total_nanoseconds()
        )

    def to_string(self: Self, format: str) -> SeriesT:  # noqa: A002
        """Convert a Date/Time/Datetime series into a String series with the given format.

        Arguments:
            format: Format string for converting the datetime to string.

        Returns:
            A new Series with the datetime values formatted as strings according to the specified format.

        Notes:
            Unfortunately, different libraries interpret format directives a bit
            differently.

            - Chrono, the library used by Polars, uses `"%.f"` for fractional seconds,
              whereas pandas and Python stdlib use `".%f"`.
            - PyArrow interprets `"%S"` as "seconds, including fractional seconds"
              whereas most other tools interpret it as "just seconds, as 2 digits".

            Therefore, we make the following adjustments:

            - for pandas-like libraries, we replace `"%S.%f"` with `"%S%.f"`.
            - for PyArrow, we replace `"%S.%f"` with `"%S"`.

            Workarounds like these don't make us happy, and we try to avoid them as
            much as possible, but here we feel like it's the best compromise.

            If you just want to format a date/datetime Series as a local datetime
            string, and have it work as consistently as possible across libraries,
            we suggest using:

            - `"%Y-%m-%dT%H:%M:%S%.f"` for datetimes
            - `"%Y-%m-%d"` for dates

            though note that, even then, different tools may return a different number
            of trailing zeros. Nonetheless, this is probably consistent enough for
            most applications.

            If you have an application where this is not enough, please open an issue
            and let us know.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.to_string(format)
        )

    def replace_time_zone(self: Self, time_zone: str | None) -> SeriesT:
        """Replace time zone.

        Arguments:
            time_zone: Target time zone.

        Returns:
            A new Series with the specified time zone.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.replace_time_zone(time_zone)
        )

    def convert_time_zone(self: Self, time_zone: str) -> SeriesT:
        """Convert time zone.

        If converting from a time-zone-naive column, then conversion happens
        as if converting from UTC.

        Arguments:
            time_zone: Target time zone.

        Returns:
            A new Series with the specified time zone.
        """
        if time_zone is None:
            msg = "Target `time_zone` cannot be `None` in `convert_time_zone`. Please use `replace_time_zone(None)` if you want to remove the time zone."
            raise TypeError(msg)
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.convert_time_zone(time_zone)
        )

    def timestamp(self: Self, time_unit: Literal["ns", "us", "ms"] = "us") -> SeriesT:
        """Return a timestamp in the given time unit.

        Arguments:
            time_unit: {'ns', 'us', 'ms'}
                Time unit.

        Returns:
            A new Series with timestamps in the specified time unit.
        """
        if time_unit not in {"ns", "us", "ms"}:
            msg = (
                "invalid `time_unit`"
                f"\n\nExpected one of {{'ns', 'us', 'ms'}}, got {time_unit!r}."
            )
            raise ValueError(msg)
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.timestamp(time_unit)
        )
