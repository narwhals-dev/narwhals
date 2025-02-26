from __future__ import annotations

import string
from typing import TYPE_CHECKING
from typing import Any

import pyarrow.compute as pc

from narwhals._arrow.utils import lit
from narwhals._arrow.utils import parse_datetime_format
from narwhals.utils import _StoresNative

if TYPE_CHECKING:
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import ArrowChunkedArray
    from narwhals._arrow.typing import Incomplete


class ArrowSeriesStringNamespace(_StoresNative["ArrowChunkedArray"]):
    def __init__(self: Self, series: ArrowSeries) -> None:
        self._compliant_series: ArrowSeries = series

    @property
    def native(self) -> ArrowChunkedArray:
        return self._compliant_series.native

    def len_chars(self: Self) -> ArrowSeries:
        return self._compliant_series._from_native_series(pc.utf8_length(self.native))

    def replace(
        self: Self, pattern: str, value: str, *, literal: bool, n: int
    ) -> ArrowSeries:
        fn = pc.replace_substring if literal else pc.replace_substring_regex
        arr = fn(self.native, pattern, replacement=value, max_replacements=n)
        return self._compliant_series._from_native_series(arr)

    def replace_all(
        self: Self, pattern: str, value: str, *, literal: bool
    ) -> ArrowSeries:
        return self.replace(pattern, value, literal=literal, n=-1)

    def strip_chars(self: Self, characters: str | None) -> ArrowSeries:
        whitespace = string.whitespace
        return self._compliant_series._from_native_series(
            pc.utf8_trim(self.native, characters or whitespace)
        )

    def starts_with(self: Self, prefix: str) -> ArrowSeries:
        return self._compliant_series._from_native_series(
            pc.equal(self.slice(0, len(prefix)).native, lit(prefix))
        )

    def ends_with(self: Self, suffix: str) -> ArrowSeries:
        return self._compliant_series._from_native_series(
            pc.equal(self.slice(-len(suffix), None).native, lit(suffix))
        )

    def contains(self: Self, pattern: str, *, literal: bool) -> ArrowSeries:
        check_func = pc.match_substring if literal else pc.match_substring_regex
        return self._compliant_series._from_native_series(
            check_func(self.native, pattern)
        )

    def slice(self: Self, offset: int, length: int | None) -> ArrowSeries:
        stop = offset + length if length is not None else None
        return self._compliant_series._from_native_series(
            pc.utf8_slice_codeunits(self.native, start=offset, stop=stop)
        )

    def to_datetime(self: Self, format: str | None) -> ArrowSeries:  # noqa: A002
        format = parse_datetime_format(self.native) if format is None else format
        strptime: Incomplete = pc.strptime
        timestamp_array: pa.Array[pa.TimestampScalar[Any, Any]] = strptime(
            self.native, format=format, unit="us"
        )
        return self._compliant_series._from_native_series(timestamp_array)

    def to_uppercase(self: Self) -> ArrowSeries:
        return self._compliant_series._from_native_series(pc.utf8_upper(self.native))

    def to_lowercase(self: Self) -> ArrowSeries:
        return self._compliant_series._from_native_series(pc.utf8_lower(self.native))
