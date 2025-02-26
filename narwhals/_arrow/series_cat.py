from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from narwhals.utils import _StoresNative

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import ArrowChunkedArray
    from narwhals._arrow.typing import Incomplete


class ArrowSeriesCatNamespace(_StoresNative["ArrowChunkedArray"]):
    def __init__(self: Self, series: ArrowSeries) -> None:
        self._compliant_series: ArrowSeries = series

    @property
    def native(self) -> ArrowChunkedArray:
        return self._compliant_series.native

    def get_categories(self: Self) -> ArrowSeries:
        # NOTE: Should be `list[pa.DictionaryArray]`, but `DictionaryArray` has no attributes
        chunks: Incomplete = self.native.chunks
        return self._compliant_series._from_native_series(
            pa.concat_arrays(x.dictionary for x in chunks).unique()
        )
