from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from narwhals._arrow.utils import ArrowSeriesNamespace
from narwhals._compliant.any_namespace import StructNamespace

if TYPE_CHECKING:
    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.series import ArrowSeries


class ArrowSeriesStructNamespace(ArrowSeriesNamespace, StructNamespace["ArrowSeries"]):
    def field(self, name: str) -> ArrowSeries:
        return self.with_native(pc.struct_field(self.native, name)).alias(name)

    def unnest(self) -> ArrowDataFrame:
        from narwhals._arrow.dataframe import ArrowDataFrame

        native = self.native
        struct_type: pa.StructType = native.type
        table = pa.table({n: pc.struct_field(native, n) for n in struct_type.names})
        return ArrowDataFrame.from_native(table, context=self.compliant)
