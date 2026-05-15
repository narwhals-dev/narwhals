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

        # NOTE: struct_type.names is not available until pyarrow 18.0.0
        n_fields = struct_type.num_fields
        table = pa.table(
            {
                struct_type.field(idx).name: pc.struct_field(native, idx)
                for idx in range(n_fields)
            }
        )
        return ArrowDataFrame.from_native(table, context=self.compliant)
