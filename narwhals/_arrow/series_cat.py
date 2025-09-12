from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pyarrow as pa

from narwhals._arrow.utils import ArrowSeriesNamespace

if TYPE_CHECKING:
    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import Incomplete
    from narwhals._compliant.typing import Accessor


class ArrowSeriesCatNamespace(ArrowSeriesNamespace):
    _accessor: ClassVar[Accessor] = "cat"

    def get_categories(self) -> ArrowSeries:
        # NOTE: Should be `list[pa.DictionaryArray]`, but `DictionaryArray` has no attributes
        chunks: Incomplete = self.native.chunks
        return self.with_native(pa.concat_arrays(x.dictionary for x in chunks).unique())
