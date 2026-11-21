from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.compute as pc

from narwhals._arrow.utils import ArrowSeriesNamespace, map_get
from narwhals._compliant.any_namespace import MapNamespace

if TYPE_CHECKING:
    from narwhals._arrow.series import ArrowSeries
    from narwhals.typing import NonNestedLiteral


class ArrowSeriesMapNamespace(ArrowSeriesNamespace, MapNamespace["ArrowSeries"]):
    def len(self) -> ArrowSeries:
        lengths_by_chunk = []
        for chunk in cast("list[pa.ListArray[Any]]", self.native.chunks):
            offsets = chunk.offsets
            lengths = pc.subtract(offsets.slice(1), offsets.slice(0, len(chunk)))
            # Offsets omit the validity bitmap, even when a null map spans entries.
            masked_lengths = pc.if_else(chunk.is_valid(), lengths, None)
            lengths_by_chunk.append(masked_lengths.cast(pa.uint32()))
        return self.with_native(pa.chunked_array(lengths_by_chunk, type=pa.uint32()))

    def get(self, key: NonNestedLiteral) -> ArrowSeries:
        return self.with_native(map_get(self.native, key))
