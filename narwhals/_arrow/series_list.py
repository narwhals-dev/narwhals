from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import ArrowChunkedArray


class ArrowSeriesListNamespace:
    def __init__(self: Self, series: ArrowSeries) -> None:
        self._arrow_series: ArrowSeries = series

    @property
    def native(self) -> ArrowChunkedArray:
        return self._arrow_series.native

    def len(self: Self) -> ArrowSeries:
        return self._arrow_series._from_native_series(
            pc.list_value_length(self.native).cast(pa.uint32())
        )
