from __future__ import annotations

import operator
import re
from datetime import datetime, time
from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerSeriesNamespace
from narwhals._compliant.any_namespace import StringNamespace
from narwhals_dict.utils import parse_datetime_format, parse_time_format

if TYPE_CHECKING:
    from collections.abc import Callable

    from narwhals_dict.series import DictSeries


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

        def fn(string: str, replacement: str) -> str:
            if literal:
                return string.replace(pattern, replacement, n)
            return re.sub(pattern, replacement, string, count=max(n, 0))

        return self._with_other(fn, value)

    def replace_all(
        self, value: DictSeries, pattern: str, *, literal: bool
    ) -> DictSeries:
        return self.replace(value, pattern, literal=literal, n=-1)

    def strip_chars(self, characters: str | None) -> DictSeries:
        return self._unary(lambda value: value.strip(characters))

    def starts_with(self, prefix: DictSeries) -> DictSeries:
        return self._with_other(str.startswith, prefix)

    def ends_with(self, suffix: DictSeries) -> DictSeries:
        return self._with_other(str.endswith, suffix)

    def contains(self, pattern: DictSeries, *, literal: bool) -> DictSeries:
        def fn(string: str, pat: str) -> bool:
            return pat in string if literal else re.search(pat, string) is not None

        return self._with_other(fn, pattern)

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
