from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Generic
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.expr import Expr

ExprT = TypeVar("ExprT", bound="Expr")


class ExprStringNamespace(Generic[ExprT]):
    def __init__(self: Self, expr: ExprT) -> None:
        self._expr = expr

    def len_chars(self: Self) -> ExprT:
        r"""Return the length of each string as the number of characters.

        Returns:
            A new expression.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"words": ["foo", "345", None]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(words_len=nw.col("words").str.len_chars())
            ┌─────────────────────┐
            | Narwhals DataFrame  |
            |---------------------|
            |shape: (3, 2)        |
            |┌───────┬───────────┐|
            |│ words ┆ words_len │|
            |│ ---   ┆ ---       │|
            |│ str   ┆ u32       │|
            |╞═══════╪═══════════╡|
            |│ foo   ┆ 3         │|
            |│ 345   ┆ 3         │|
            |│ null  ┆ null      │|
            |└───────┴───────────┘|
            └─────────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.len_chars(),
            self._expr._metadata,
        )

    def replace(
        self: Self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> ExprT:
        r"""Replace first matching regex/literal substring with a new string value.

        Arguments:
            pattern: A valid regular expression pattern.
            value: String that will replace the matched substring.
            literal: Treat `pattern` as a literal string.
            n: Number of matches to replace.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"foo": ["123abc", "abc abc123"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(replaced=nw.col("foo").str.replace("abc", ""))
            ┌──────────────────────┐
            |  Narwhals DataFrame  |
            |----------------------|
            |          foo replaced|
            |0      123abc      123|
            |1  abc abc123   abc123|
            └──────────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.replace(
                pattern, value, literal=literal, n=n
            ),
            self._expr._metadata,
        )

    def replace_all(
        self: Self, pattern: str, value: str, *, literal: bool = False
    ) -> ExprT:
        r"""Replace all matching regex/literal substring with a new string value.

        Arguments:
            pattern: A valid regular expression pattern.
            value: String that will replace the matched substring.
            literal: Treat `pattern` as a literal string.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"foo": ["123abc", "abc abc123"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(replaced=nw.col("foo").str.replace_all("abc", ""))
            ┌──────────────────────┐
            |  Narwhals DataFrame  |
            |----------------------|
            |          foo replaced|
            |0      123abc      123|
            |1  abc abc123      123|
            └──────────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.replace_all(
                pattern, value, literal=literal
            ),
            self._expr._metadata,
        )

    def strip_chars(self: Self, characters: str | None = None) -> ExprT:
        r"""Remove leading and trailing characters.

        Arguments:
            characters: The set of characters to be removed. All combinations of this
                set of characters will be stripped from the start and end of the string.
                If set to None (default), all leading and trailing whitespace is removed
                instead.

        Returns:
            A new expression.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"fruits": ["apple", "\nmango"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(stripped=nw.col("fruits").str.strip_chars()).to_dict(
            ...     as_series=False
            ... )
            {'fruits': ['apple', '\nmango'], 'stripped': ['apple', 'mango']}
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.strip_chars(characters),
            self._expr._metadata,
        )

    def starts_with(self: Self, prefix: str) -> ExprT:
        r"""Check if string values start with a substring.

        Arguments:
            prefix: prefix substring

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"fruits": ["apple", "mango", None]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(has_prefix=nw.col("fruits").str.starts_with("app"))
            ┌───────────────────┐
            |Narwhals DataFrame |
            |-------------------|
            |  fruits has_prefix|
            |0  apple       True|
            |1  mango      False|
            |2   None       None|
            └───────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.starts_with(prefix),
            self._expr._metadata,
        )

    def ends_with(self: Self, suffix: str) -> ExprT:
        r"""Check if string values end with a substring.

        Arguments:
            suffix: suffix substring

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"fruits": ["apple", "mango", None]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(has_suffix=nw.col("fruits").str.ends_with("ngo"))
            ┌───────────────────┐
            |Narwhals DataFrame |
            |-------------------|
            |  fruits has_suffix|
            |0  apple      False|
            |1  mango       True|
            |2   None       None|
            └───────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.ends_with(suffix),
            self._expr._metadata,
        )

    def contains(self: Self, pattern: str, *, literal: bool = False) -> ExprT:
        r"""Check if string contains a substring that matches a pattern.

        Arguments:
            pattern: A Character sequence or valid regular expression pattern.
            literal: If True, treats the pattern as a literal string.
                     If False, assumes the pattern is a regular expression.

        Returns:
            A new expression.

        Examples:
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> df_native = pa.table({"pets": ["cat", "dog", "rabbit and parrot"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     default_match=nw.col("pets").str.contains("cat|parrot"),
            ...     case_insensitive_match=nw.col("pets").str.contains("cat|(?i)parrot"),
            ... ).to_native()
            pyarrow.Table
            pets: string
            default_match: bool
            case_insensitive_match: bool
            ----
            pets: [["cat","dog","rabbit and parrot"]]
            default_match: [[true,false,true]]
            case_insensitive_match: [[true,false,true]]
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.contains(
                pattern, literal=literal
            ),
            self._expr._metadata,
        )

    def slice(self: Self, offset: int, length: int | None = None) -> ExprT:
        r"""Create subslices of the string values of an expression.

        Arguments:
            offset: Start index. Negative indexing is supported.
            length: Length of the slice. If set to `None` (default), the slice is taken to the
                end of the string.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"s": ["pear", None, "papaya"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(s_sliced=nw.col("s").str.slice(4, length=3))
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        s s_sliced|
            |0    pear         |
            |1    None     None|
            |2  papaya       ya|
            └──────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.slice(
                offset=offset, length=length
            ),
            self._expr._metadata,
        )

    def split(self: Self, by: str) -> ExprT:
        r"""Split the string values of an expression by a substring.

        Arguments:
            by: Substring to split by.

        Returns:
            A new expression.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"s": ["foo bar", "foo_bar"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(nw.col("s").str.split("_").alias("s_split"))
            ┌────────────────────────────┐
            |     Narwhals DataFrame     |
            |----------------------------|
            |shape: (2, 2)               |
            |┌─────────┬────────────────┐|
            |│ s       ┆ s_split        │|
            |│ ---     ┆ ---            │|
            |│ str     ┆ list[str]      │|
            |╞═════════╪════════════════╡|
            |│ foo bar ┆ ["foo bar"]    │|
            |│ foo_bar ┆ ["foo", "bar"] │|
            |└─────────┴────────────────┘|
            └────────────────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.split(by=by),
            self._expr._metadata,
        )

    def head(self: Self, n: int = 5) -> ExprT:
        r"""Take the first n elements of each string.

        Arguments:
            n: Number of elements to take. Negative indexing is **not** supported.

        Returns:
            A new expression.

        Notes:
            If the length of the string has fewer than `n` characters, the full string is returned.

        Examples:
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> df_native = pa.table({"lyrics": ["taata", "taatatata", "zukkyun"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(lyrics_head=nw.col("lyrics").str.head()).to_native()
            pyarrow.Table
            lyrics: string
            lyrics_head: string
            ----
            lyrics: [["taata","taatatata","zukkyun"]]
            lyrics_head: [["taata","taata","zukky"]]
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.slice(0, n),
            self._expr._metadata,
        )

    def tail(self: Self, n: int = 5) -> ExprT:
        r"""Take the last n elements of each string.

        Arguments:
            n: Number of elements to take. Negative indexing is **not** supported.

        Returns:
            A new expression.

        Notes:
            If the length of the string has fewer than `n` characters, the full string is returned.

        Examples:
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> df_native = pa.table({"lyrics": ["taata", "taatatata", "zukkyun"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(lyrics_tail=nw.col("lyrics").str.tail()).to_native()
            pyarrow.Table
            lyrics: string
            lyrics_tail: string
            ----
            lyrics: [["taata","taatatata","zukkyun"]]
            lyrics_tail: [["taata","atata","kkyun"]]
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.slice(
                offset=-n, length=None
            ),
            self._expr._metadata,
        )

    def to_datetime(self: Self, format: str | None = None) -> ExprT:  # noqa: A002
        """Convert to Datetime dtype.

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
            A new expression.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": ["2020-01-01", "2020-01-02"]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a").str.to_datetime(format="%Y-%m-%d"))
            ┌───────────────────────┐
            |  Narwhals DataFrame   |
            |-----------------------|
            |shape: (2, 1)          |
            |┌─────────────────────┐|
            |│ a                   │|
            |│ ---                 │|
            |│ datetime[μs]        │|
            |╞═════════════════════╡|
            |│ 2020-01-01 00:00:00 │|
            |│ 2020-01-02 00:00:00 │|
            |└─────────────────────┘|
            └───────────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.to_datetime(format=format),
            self._expr._metadata,
        )

    def to_uppercase(self: Self) -> ExprT:
        r"""Transform string to uppercase variant.

        Returns:
            A new expression.

        Notes:
            The PyArrow backend will convert 'ß' to 'ẞ' instead of 'SS'.
            For more info see [the related issue](https://github.com/apache/arrow/issues/34599).
            There may be other unicode-edge-case-related variations across implementations.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"fruits": ["apple", None]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(upper_col=nw.col("fruits").str.to_uppercase())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |  fruits upper_col|
            |0  apple     APPLE|
            |1   None      None|
            └──────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.to_uppercase(),
            self._expr._metadata,
        )

    def to_lowercase(self: Self) -> ExprT:
        r"""Transform string to lowercase variant.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"fruits": ["APPLE", None]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(lower_col=nw.col("fruits").str.to_lowercase())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |  fruits lower_col|
            |0  APPLE     apple|
            |1   None      None|
            └──────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.to_lowercase(),
            self._expr._metadata,
        )
