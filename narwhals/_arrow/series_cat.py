from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from narwhals._arrow.utils import ArrowSeriesNamespace

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import Incomplete


class ArrowSeriesCatNamespace(ArrowSeriesNamespace):
    def get_categories(self: Self) -> ArrowSeries:
        # NOTE: Should be `list[pa.DictionaryArray]`, but `DictionaryArray` has no attributes
        chunks: Incomplete = self.native.chunks
        return self.from_native(pa.concat_arrays(x.dictionary for x in chunks).unique())
