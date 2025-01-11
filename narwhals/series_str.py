from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.series import Series

SeriesT = TypeVar("SeriesT", bound="Series[Any]")


class SeriesStringNamespace(Generic[SeriesT]):
    def __init__(self: Self, series: SeriesT) -> None:
        self._narwhals_series = series

    def len_chars(self: Self) -> SeriesT:
        r"""Return the length of each string as the number of characters.

        Returns:
            A new Series containing the length of each string in characters.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.len_chars()
        )

    def replace(
        self: Self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> SeriesT:
        r"""Replace first matching regex/literal substring with a new string value.

        Arguments:
            pattern: A valid regular expression pattern.
            value: String that will replace the matched substring.
            literal: Treat `pattern` as a literal string.
            n: Number of matches to replace.

        Returns:
            A new Series with the regex/literal pattern replaced with the specified value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.replace(
                pattern, value, literal=literal, n=n
            )
        )

    def replace_all(
        self: Self, pattern: str, value: str, *, literal: bool = False
    ) -> SeriesT:
        r"""Replace all matching regex/literal substring with a new string value.

        Arguments:
            pattern: A valid regular expression pattern.
            value: String that will replace the matched substring.
            literal: Treat `pattern` as a literal string.

        Returns:
            A new Series with all occurrences of pattern replaced with the specified value.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.replace_all(
                pattern, value, literal=literal
            )
        )

    def strip_chars(self: Self, characters: str | None = None) -> SeriesT:
        r"""Remove leading and trailing characters.

        Arguments:
            characters: The set of characters to be removed. All combinations of this set of characters will be stripped from the start and end of the string. If set to None (default), all leading and trailing whitespace is removed instead.

        Returns:
            A new Series with leading and trailing characters removed.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.strip_chars(characters)
        )

    def starts_with(self: Self, prefix: str) -> SeriesT:
        r"""Check if string values start with a substring.

        Arguments:
            prefix: prefix substring

        Returns:
            A new Series with boolean values indicating if each string starts with the prefix.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.starts_with(prefix)
        )

    def ends_with(self: Self, suffix: str) -> SeriesT:
        r"""Check if string values end with a substring.

        Arguments:
            suffix: suffix substring

        Returns:
            A new Series with boolean values indicating if each string ends with the suffix.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.ends_with(suffix)
        )

    def contains(self: Self, pattern: str, *, literal: bool = False) -> SeriesT:
        r"""Check if string contains a substring that matches a pattern.

        Arguments:
            pattern: A Character sequence or valid regular expression pattern.
            literal: If True, treats the pattern as a literal string.
                     If False, assumes the pattern is a regular expression.

        Returns:
            A new Series with boolean values indicating if each string contains the pattern.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.contains(pattern, literal=literal)
        )

    def slice(self: Self, offset: int, length: int | None = None) -> SeriesT:
        r"""Create subslices of the string values of a Series.

        Arguments:
            offset: Start index. Negative indexing is supported.
            length: Length of the slice. If set to `None` (default), the slice is taken to the
                end of the string.

        Returns:
            A new Series containing subslices of each string.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.slice(
                offset=offset, length=length
            )
        )

    def head(self: Self, n: int = 5) -> SeriesT:
        r"""Take the first n elements of each string.

        Arguments:
            n: Number of elements to take. Negative indexing is supported (see note (1.))

        Returns:
            A new Series containing the first n characters of each string.

        Notes:
            1. When the `n` input is negative, `head` returns characters up to the n-th from the end of the string.
                For example, if `n = -3`, then all characters except the last three are returned.
            2. If the length of the string has fewer than `n` characters, the full string is returned.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.slice(offset=0, length=n)
        )

    def tail(self: Self, n: int = 5) -> SeriesT:
        r"""Take the last n elements of each string.

        Arguments:
            n: Number of elements to take. Negative indexing is supported (see note (1.))

        Returns:
            A new Series containing the last n characters of each string.

        Notes:
            1. When the `n` input is negative, `tail` returns characters starting from the n-th from the beginning of
                the string. For example, if `n = -3`, then all characters except the first three are returned.
            2. If the length of the string has fewer than `n` characters, the full string is returned.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.slice(offset=-n, length=None)
        )

    def to_uppercase(self) -> SeriesT:
        r"""Transform string to uppercase variant.

        Returns:
            A new Series with values converted to uppercase.

        Notes:
            The PyArrow backend will convert 'ß' to 'ẞ' instead of 'SS'.
            For more info see: https://github.com/apache/arrow/issues/34599
            There may be other unicode-edge-case-related variations across implementations.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.to_uppercase()
        )

    def to_lowercase(self) -> SeriesT:
        r"""Transform string to lowercase variant.

        Returns:
            A new Series with values converted to lowercase.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.to_lowercase()
        )

    def to_datetime(self: Self, format: str | None = None) -> SeriesT:  # noqa: A002
        """Parse Series with strings to a Series with Datetime dtype.

        Notes:
            pandas defaults to nanosecond time unit, Polars to microsecond.
            Prior to pandas 2.0, nanoseconds were the only time unit supported
            in pandas, with no ability to set any other one. The ability to
            set the time unit in pandas, if the version permits, will arrive.

        Warning:
            As different backends auto-infer format in different ways, if `format=None`
            there is no guarantee that the result will be equal.

        Arguments:
            format: Format to use for conversion. If set to None (default), the format is
                inferred from the data.

        Returns:
            A new Series with datetime dtype.
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.to_datetime(format=format)
        )
