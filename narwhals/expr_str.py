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
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.len_chars()
        )

    def replace(
        self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> ExprT:
        r"""Replace first matching regex/literal substring with a new string value.

        Arguments:
            pattern: A valid regular expression pattern.
            value: String that will replace the matched substring.
            literal: Treat `pattern` as a literal string.
            n: Number of matches to replace.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.replace(
                pattern, value, literal=literal, n=n
            )
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
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.replace_all(
                pattern, value, literal=literal
            )
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
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.strip_chars(characters)
        )

    def starts_with(self: Self, prefix: str) -> ExprT:
        r"""Check if string values start with a substring.

        Arguments:
            prefix: prefix substring

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.starts_with(prefix)
        )

    def ends_with(self: Self, suffix: str) -> ExprT:
        r"""Check if string values end with a substring.

        Arguments:
            suffix: suffix substring

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.ends_with(suffix)
        )

    def contains(self: Self, pattern: str, *, literal: bool = False) -> ExprT:
        r"""Check if string contains a substring that matches a pattern.

        Arguments:
            pattern: A Character sequence or valid regular expression pattern.
            literal: If True, treats the pattern as a literal string.
                     If False, assumes the pattern is a regular expression.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.contains(
                pattern, literal=literal
            )
        )

    def slice(self: Self, offset: int, length: int | None = None) -> ExprT:
        r"""Create subslices of the string values of an expression.

        Arguments:
            offset: Start index. Negative indexing is supported.
            length: Length of the slice. If set to `None` (default), the slice is taken to the
                end of the string.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.slice(
                offset=offset, length=length
            )
        )

    def head(self: Self, n: int = 5) -> ExprT:
        r"""Take the first n elements of each string.

        Arguments:
            n: Number of elements to take. Negative indexing is **not** supported.

        Returns:
            A new expression.

        Notes:
            If the length of the string has fewer than `n` characters, the full string is returned.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.slice(0, n)
        )

    def tail(self: Self, n: int = 5) -> ExprT:
        r"""Take the last n elements of each string.

        Arguments:
            n: Number of elements to take. Negative indexing is **not** supported.

        Returns:
            A new expression.

        Notes:
            If the length of the string has fewer than `n` characters, the full string is returned.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.slice(
                offset=-n, length=None
            )
        )

    def to_datetime(self: Self, format: str | None = None) -> ExprT:  # noqa: A002
        """Convert to Datetime dtype.

        Warning:
            As different backends auto-infer format in different ways, if `format=None`
            there is no guarantee that the result will be equal.

        Arguments:
            format: Format to use for conversion. If set to None (default), the format is
                inferred from the data.

        Returns:
            A new expression.

        Notes:
            pandas defaults to nanosecond time unit, Polars to microsecond.
            Prior to pandas 2.0, nanoseconds were the only time unit supported
            in pandas, with no ability to set any other one. The ability to
            set the time unit in pandas, if the version permits, will arrive.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.to_datetime(format=format)
        )

    def to_uppercase(self: Self) -> ExprT:
        r"""Transform string to uppercase variant.

        Returns:
            A new expression.

        Notes:
            The PyArrow backend will convert 'ß' to 'ẞ' instead of 'SS'.
            For more info see [the related issue](https://github.com/apache/arrow/issues/34599).
            There may be other unicode-edge-case-related variations across implementations.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.to_uppercase()
        )

    def to_lowercase(self: Self) -> ExprT:
        r"""Transform string to lowercase variant.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).str.to_lowercase()
        )
