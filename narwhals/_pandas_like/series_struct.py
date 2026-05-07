from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant.any_namespace import StructNamespace
from narwhals._pandas_like.utils import PandasLikeSeriesNamespace

if TYPE_CHECKING:
    import pyarrow as pa

    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.series import PandasLikeSeries


class PandasLikeSeriesStructNamespace(
    PandasLikeSeriesNamespace, StructNamespace["PandasLikeSeries"]
):
    def field(self, name: str) -> PandasLikeSeries:
        return self.with_native(self.native.struct.field(name)).alias(name)

    def unnest(self) -> PandasLikeDataFrame:
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        native = self.native
        struct_type: pa.StructType = native.dtype.pyarrow_dtype

        # NOTE: struct_type.names is not available until pyarrow 18.0.0
        n_fields = struct_type.num_fields
        ns = self.implementation.to_native_namespace()

        result = ns.DataFrame(
            {
                struct_type.field(idx).name: native.struct.field(idx)
                for idx in range(n_fields)
            }
        )
        return PandasLikeDataFrame.from_native(result, context=self.compliant)
