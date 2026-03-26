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
        pa_type: pa.StructType = native.dtype.pyarrow_dtype
        ns = self.implementation.to_native_namespace()
        result = ns.DataFrame({name: native.struct.field(name) for name in pa_type.names})
        return PandasLikeDataFrame.from_native(result, context=self.compliant)
