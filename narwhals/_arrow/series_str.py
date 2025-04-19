from __future__ import annotations

import string
from typing import TYPE_CHECKING

import pyarrow.compute as pc

from narwhals._arrow.utils import ArrowSeriesNamespace
from narwhals._arrow.utils import lit
from narwhals._arrow.utils import parse_datetime_format

if TYPE_CHECKING:
    from narwhals._arrow.series import ArrowSeries


class ArrowSeriesStringNamespace(ArrowSeriesNamespace):
    def len_chars(self) -> ArrowSeries:
        return self.with_native(pc.utf8_length(self.native))

    def replace(self, pattern: str, value: str, *, literal: bool, n: int) -> ArrowSeries:
        fn = pc.replace_substring if literal else pc.replace_substring_regex
        arr = fn(self.native, pattern, replacement=value, max_replacements=n)
        return self.with_native(arr)

    def replace_all(self, pattern: str, value: str, *, literal: bool) -> ArrowSeries:
        return self.replace(pattern, value, literal=literal, n=-1)

    def strip_chars(self, characters: str | None) -> ArrowSeries:
        return self.with_native(
            pc.utf8_trim(self.native, characters or string.whitespace)
        )

    def starts_with(self, prefix: str) -> ArrowSeries:
        return self.with_native(pc.equal(self.slice(0, len(prefix)).native, lit(prefix)))

    def ends_with(self, suffix: str) -> ArrowSeries:
        return self.with_native(
            pc.equal(self.slice(-len(suffix), None).native, lit(suffix))
        )

    def contains(self, pattern: str, *, literal: bool) -> ArrowSeries:
        check_func = pc.match_substring if literal else pc.match_substring_regex
        return self.with_native(check_func(self.native, pattern))

    def slice(self, offset: int, length: int | None) -> ArrowSeries:
        stop = offset + length if length is not None else None
        return self.with_native(
            pc.utf8_slice_codeunits(self.native, start=offset, stop=stop)
        )

    def split(self, by: str) -> ArrowSeries:
        split_series = pc.split_pattern(self.native, by)  # type: ignore[call-overload]
        return self.with_native(split_series)

    def to_datetime(self, format: str | None) -> ArrowSeries:
        format = parse_datetime_format(self.native) if format is None else format
        timestamp_array = pc.strptime(self.native, format=format, unit="us")
        return self.with_native(timestamp_array)

    def to_uppercase(self) -> ArrowSeries:
        return self.with_native(pc.utf8_upper(self.native))

    def to_lowercase(self) -> ArrowSeries:
        return self.with_native(pc.utf8_lower(self.native))
