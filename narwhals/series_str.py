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

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s_native = pl.Series(["foo", "345", None])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.len_chars().to_native()  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [u32]
            [
                    3
                    3
                    null
            ]
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

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> s_native = pd.Series(["123abc", "abc abc123"])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.replace("abc", "").to_native()
            0        123
            1     abc123
            dtype: object
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

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> s_native = pd.Series(["123abc", "abc abc123"])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.replace_all("abc", "").to_native()
            0     123
            1     123
            dtype: object
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

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s_native = pl.Series(["apple", "\nmango"])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.strip_chars().to_native()  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [str]
            [
                    "apple"
                    "mango"
            ]
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

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> s_native = pd.Series(["apple", "mango", None])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.starts_with("app").to_native()
            0     True
            1    False
            2     None
            dtype: object
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

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> s_native = pd.Series(["apple", "mango", None])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.ends_with("ngo").to_native()
            0    False
            1     True
            2     None
            dtype: object
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

        Examples:
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> s_native = pa.chunked_array([["cat", "dog", "rabbit and parrot"]])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.contains(
            ...     "cat|parrot"
            ... ).to_native()  # doctest: +ELLIPSIS  +NORMALIZE_WHITESPACE
            <pyarrow.lib.ChunkedArray object at ...>
            [
            [
                true,
                false,
                true
            ]
            ]
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

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> s_native = pd.Series(["pear", None, "papaya"])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.slice(4, 3).to_native()  # doctest: +NORMALIZE_WHITESPACE
            0
            1    None
            2      ya
            dtype: object
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.slice(
                offset=offset, length=length
            )
        )

    def split(self: Self, by: str) -> SeriesT:
        r"""Split the string values of a Series by a substring.

        Arguments:
            by: Substring to split by.

        Returns:
            A new Series containing lists of strings.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s_native = pl.Series(["foo bar", "foo_bar"])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.split("_").to_native()  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [list[str]]
            [
                    ["foo bar"]
                    ["foo", "bar"]
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.split(by=by)
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

        Examples:
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> s_native = pa.chunked_array([["taata", "taatatata", "zukkyun"]])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.head().to_native()  # doctest: +ELLIPSIS  +NORMALIZE_WHITESPACE
            <pyarrow.lib.ChunkedArray object at ...>
            [
            [
                "taata",
                "taata",
                "zukky"
            ]
            ]
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

        Examples:
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> s_native = pa.chunked_array([["taata", "taatatata", "zukkyun"]])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.tail().to_native()  # doctest: +ELLIPSIS  +NORMALIZE_WHITESPACE
            <pyarrow.lib.ChunkedArray object at ...>
            [
            [
                "taata",
                "atata",
                "kkyun"
            ]
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.slice(offset=-n, length=None)
        )

    def to_uppercase(self: Self) -> SeriesT:
        r"""Transform string to uppercase variant.

        Returns:
            A new Series with values converted to uppercase.

        Notes:
            The PyArrow backend will convert 'ß' to 'ẞ' instead of 'SS'.
            For more info see: https://github.com/apache/arrow/issues/34599
            There may be other unicode-edge-case-related variations across implementations.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> s_native = pd.Series(["apple", None])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.to_uppercase().to_native()
            0    APPLE
            1     None
            dtype: object
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.to_uppercase()
        )

    def to_lowercase(self: Self) -> SeriesT:
        r"""Transform string to lowercase variant.

        Returns:
            A new Series with values converted to lowercase.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> s_native = pd.Series(["APPLE", None])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.to_lowercase().to_native()
            0    apple
            1     None
            dtype: object
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.to_lowercase()
        )

    def to_datetime(self: Self, format: str | None = None) -> SeriesT:  # noqa: A002
        """Parse Series with strings to a Series with Datetime dtype.

        Notes:
            - pandas defaults to nanosecond time unit, Polars to microsecond.
              Prior to pandas 2.0, nanoseconds were the only time unit supported
              in pandas, with no ability to set any other one. The ability to
              set the time unit in pandas, if the version permits, will arrive.
            - timezone-aware strings are all converted to and parsed as UTC.

        Warning:
            As different backends auto-infer format in different ways, if `format=None`
            there is no guarantee that the result will be equal.

        Arguments:
            format: Format to use for conversion. If set to None (default), the format is
                inferred from the data.

        Returns:
            A new Series with datetime dtype.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s_native = pl.Series(["2020-01-01", "2020-01-02"])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.to_datetime(
            ...     format="%Y-%m-%d"
            ... ).to_native()  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [datetime[μs]]
            [
                    2020-01-01 00:00:00
                    2020-01-02 00:00:00
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.to_datetime(format=format)
        )
