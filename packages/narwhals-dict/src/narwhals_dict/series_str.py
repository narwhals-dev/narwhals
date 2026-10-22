from __future__ import annotations

import operator
import re
from datetime import datetime, time
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerSeriesNamespace
from narwhals._compliant.any_namespace import StringNamespace
from narwhals_dict.utils import parse_datetime_format, parse_time_format

if TYPE_CHECKING:
    from collections.abc import Callable

    from narwhals_dict.series import DictSeries


_RUST_GROUP_REF = re.compile(r"\$(?:\$|\{([^}]*)\}|([0-9A-Za-z_]+))")


@lru_cache(maxsize=256)
def _polars_replacement(template: str) -> Callable[[re.Match[str]], str]:
    """An `re.sub` replacement following Polars' (Rust regex) template syntax.

    `$1`, `$name` and `${name}` expand to that group (empty if it doesn't exist or
    didn't match) and `$$` is a literal `$`. Unlike Python templates, backslashes are
    literal: a callable replacement is returned as is, with no escape processing.
    """
    if "$" not in template:
        return lambda _: template

    def expand(match: re.Match[str]) -> str:
        def group(ref: re.Match[str]) -> str:
            if (name := ref[1] if ref[1] is not None else ref[2]) is None:
                return "$"
            try:
                return match.group(int(name) if name.isdigit() else name) or ""
            except IndexError:
                return ""

        return _RUST_GROUP_REF.sub(group, template)

    return expand


class DictSeriesStringNamespace(
    EagerSeriesNamespace["DictSeries", Any], StringNamespace["DictSeries"]
):
    def _unary(self, fn: Callable[[str], Any]) -> DictSeries:
        return self.with_native(
            [None if value is None else fn(value) for value in self.native]
        )

    def _with_other(self, fn: Callable[[str, str], Any], other: Any) -> DictSeries:
        """Apply `fn` elementwise, `other` being a scalar or a (broadcastable) series."""
        values, is_scalar = self.compliant._extract_comparand(other)
        if is_scalar:
            if values is None:
                return self.with_native([None] * len(self.native))
            return self._unary(lambda value: fn(value, values))
        return self.with_native(
            [
                None if (lhs is None or rhs is None) else fn(lhs, rhs)
                for lhs, rhs in zip(self.native, values, strict=True)
            ]
        )

    def len_chars(self) -> DictSeries:
        return self._unary(len)

    def replace(
        self, value: DictSeries, pattern: str, *, literal: bool, n: int
    ) -> DictSeries:
        if n > 1 and getattr(value, "_broadcast", True) is False:
            msg = "dict backend `.str.replace` with `n > 1` only supports str replacement values"
            raise TypeError(msg)

        if n == 0:
            return self.with_native(list(self.native))
        if literal:

            def fn(string: str, replacement: str) -> str:
                return string.replace(pattern, replacement, n)
        else:
            # `pattern` is a fixed regex, so compile it once rather than per row.
            sub = re.compile(pattern).sub
            count = max(n, 0)  # `re.sub` replaces all with `count=0`, i.e. `n=-1`

            def fn(string: str, replacement: str) -> str:
                return sub(_polars_replacement(replacement), string, count=count)

        # Unlike `_with_other`, a null replacement leaves the string as is (Polars).
        replacements, is_scalar = self.compliant._extract_comparand(value)
        if is_scalar:
            if replacements is None:
                return self.with_native(list(self.native))
            return self._unary(lambda string: fn(string, replacements))
        return self.with_native(
            [
                string
                if string is None or replacement is None
                else fn(string, replacement)
                for string, replacement in zip(self.native, replacements, strict=True)
            ]
        )

    def replace_all(
        self, value: DictSeries, pattern: str, *, literal: bool
    ) -> DictSeries:
        return self.replace(value, pattern, literal=literal, n=-1)

    def strip_chars(self, characters: str | None) -> DictSeries:
        return self._unary(lambda value: value.strip(characters))

    def strip_chars_start(self, characters: str | None) -> DictSeries:
        return self._unary(lambda value: value.lstrip(characters))

    def strip_chars_end(self, characters: str | None) -> DictSeries:
        return self._unary(lambda value: value.rstrip(characters))

    def starts_with(self, prefix: DictSeries) -> DictSeries:
        return self._with_other(str.startswith, prefix)

    def ends_with(self, suffix: DictSeries) -> DictSeries:
        return self._with_other(str.endswith, suffix)

    def contains(self, pattern: DictSeries, *, literal: bool) -> DictSeries:
        if literal:
            return self._with_other(lambda string, pat: pat in string, pattern)
        values, is_scalar = self.compliant._extract_comparand(pattern)
        if is_scalar:
            # The common `str.contains("regex")`: compile once, not per row.
            if values is None:
                return self.with_native([None] * len(self.native))
            search = re.compile(values).search
            return self._unary(lambda string: search(string) is not None)
        # Per-row patterns can't be precompiled; search each against its own.
        return self._with_other(
            lambda string, pat: re.search(pat, string) is not None, pattern
        )

    def slice(self, offset: int, length: int | None) -> DictSeries:
        stop = offset + length if length is not None else None
        if offset < 0 and stop is not None and stop >= 0:
            stop = None
        return self._unary(operator.itemgetter(slice(offset, stop)))

    def split(self, by: str) -> DictSeries:
        return self._unary(lambda value: value.split(by))

    def to_datetime(self, format: str | None) -> DictSeries:
        fmt = parse_datetime_format(self.native) if format is None else format
        return self._unary(lambda value: datetime.strptime(value, fmt))  # noqa: DTZ007

    def to_date(self, format: str | None) -> DictSeries:
        return self.to_datetime(format).dt.date()

    def to_time(self, format: str | None) -> DictSeries:
        fmt = parse_time_format(self.native) if format is None else format
        if not fmt:
            return self._unary(time.fromisoformat)
        return self._unary(lambda value: datetime.strptime(value, fmt).time())  # noqa: DTZ007

    def to_lowercase(self) -> DictSeries:
        return self._unary(str.lower)

    def to_titlecase(self) -> DictSeries:
        return self._unary(str.title)

    def to_uppercase(self) -> DictSeries:
        return self._unary(str.upper)

    def zfill(self, width: int) -> DictSeries:
        return self._unary(lambda value: value.zfill(width))

    def pad_start(self, length: int, fill_char: str) -> DictSeries:
        return self._unary(lambda value: value.rjust(length, fill_char))

    def pad_end(self, length: int, fill_char: str) -> DictSeries:
        return self._unary(lambda value: value.ljust(length, fill_char))
