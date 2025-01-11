from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Generic
from typing import Literal
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.expr import Expr

ExprT = TypeVar("ExprT", bound="Expr")


class ExprDateTimeNamespace(Generic[ExprT]):
    def __init__(self: Self, expr: ExprT) -> None:
        self._expr = expr

    def date(self: Self) -> ExprT:
        """Extract the date from underlying DateTime representation.

        Returns:
            A new expression.

        Raises:
            NotImplementedError: If pandas default backend is being used.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.date()
        )

    def year(self: Self) -> ExprT:
        """Extract year from underlying DateTime representation.

        Returns the year number in the calendar date.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.year()
        )

    def month(self: Self) -> ExprT:
        """Extract month from underlying DateTime representation.

        Returns the month number starting from 1. The return value ranges from 1 to 12.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.month()
        )

    def day(self: Self) -> ExprT:
        """Extract day from underlying DateTime representation.

        Returns the day of month starting from 1. The return value ranges from 1 to 31. (The last day of month differs by months.)

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.day()
        )

    def hour(self: Self) -> ExprT:
        """Extract hour from underlying DateTime representation.

        Returns the hour number from 0 to 23.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.hour()
        )

    def minute(self: Self) -> ExprT:
        """Extract minutes from underlying DateTime representation.

        Returns the minute number from 0 to 59.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.minute()
        )

    def second(self: Self) -> ExprT:
        """Extract seconds from underlying DateTime representation.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.second()
        )

    def millisecond(self: Self) -> ExprT:
        """Extract milliseconds from underlying DateTime representation.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.millisecond()
        )

    def microsecond(self: Self) -> ExprT:
        """Extract microseconds from underlying DateTime representation.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.microsecond()
        )

    def nanosecond(self: Self) -> ExprT:
        """Extract Nanoseconds from underlying DateTime representation.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.nanosecond()
        )

    def ordinal_day(self: Self) -> ExprT:
        """Get ordinal day.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.ordinal_day()
        )

    def weekday(self: Self) -> ExprT:
        """Extract the week day from the underlying Date representation.

        Returns:
            Returns the ISO weekday number where monday = 1 and sunday = 7
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.weekday()
        )

    def total_minutes(self: Self) -> ExprT:
        """Get total minutes.

        Returns:
            A new expression.

        Notes:
            The function outputs the total minutes in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` and `cast` in this case.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.total_minutes()
        )

    def total_seconds(self: Self) -> ExprT:
        """Get total seconds.

        Returns:
            A new expression.

        Notes:
            The function outputs the total seconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` and `cast` in this case.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.total_seconds()
        )

    def total_milliseconds(self: Self) -> ExprT:
        """Get total milliseconds.

        Returns:
            A new expression.

        Notes:
            The function outputs the total milliseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` and `cast` in this case.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.total_milliseconds()
        )

    def total_microseconds(self: Self) -> ExprT:
        """Get total microseconds.

        Returns:
            A new expression.

        Notes:
            The function outputs the total microseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` and `cast` in this case.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.total_microseconds()
        )

    def total_nanoseconds(self: Self) -> ExprT:
        """Get total nanoseconds.

        Returns:
            A new expression.

        Notes:
            The function outputs the total nanoseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` and `cast` in this case.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.total_nanoseconds()
        )

    def to_string(self: Self, format: str) -> ExprT:  # noqa: A002
        """Convert a Date/Time/Datetime column into a String column with the given format.

        Arguments:
            format: Format to format temporal column with.

        Returns:
            A new expression.

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
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.to_string(format)
        )

    def replace_time_zone(self: Self, time_zone: str | None) -> ExprT:
        """Replace time zone.

        Arguments:
            time_zone: Target time zone.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.replace_time_zone(time_zone)
        )

    def convert_time_zone(self: Self, time_zone: str) -> ExprT:
        """Convert to a new time zone.

        If converting from a time-zone-naive column, then conversion happens
        as if converting from UTC.

        Arguments:
            time_zone: Target time zone.

        Returns:
            A new expression.
        """
        if time_zone is None:
            msg = "Target `time_zone` cannot be `None` in `convert_time_zone`. Please use `replace_time_zone(None)` if you want to remove the time zone."
            raise TypeError(msg)
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.convert_time_zone(time_zone)
        )

    def timestamp(self: Self, time_unit: Literal["ns", "us", "ms"] = "us") -> ExprT:
        """Return a timestamp in the given time unit.

        Arguments:
            time_unit: {'ns', 'us', 'ms'}
                Time unit.

        Returns:
            A new expression.
        """
        if time_unit not in {"ns", "us", "ms"}:
            msg = (
                "invalid `time_unit`"
                f"\n\nExpected one of {{'ns', 'us', 'ms'}}, got {time_unit!r}."
            )
            raise ValueError(msg)
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).dt.timestamp(time_unit)
        )
