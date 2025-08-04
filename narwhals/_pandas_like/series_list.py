from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant.any_namespace import ListNamespace
from narwhals._pandas_like.utils import (
    PandasLikeSeriesNamespace,
    get_dtype_backend,
    narwhals_to_native_dtype,
)
from narwhals._pandas_like.utils import is_dtype_pyarrow

if TYPE_CHECKING:
    from narwhals._pandas_like.series import PandasLikeSeries


class PandasLikeSeriesListNamespace(
    PandasLikeSeriesNamespace, ListNamespace["PandasLikeSeries"]
):
    def len(self) -> PandasLikeSeries:
        result = self.native.list.len()
        implementation = self.implementation
        backend_version = self.backend_version
        if implementation.is_pandas() and backend_version < (3, 0):  # pragma: no cover
            # `result` is a new object so it's safe to do this inplace.
            result.index = self.native.index
        dtype = narwhals_to_native_dtype(
            self.version.dtypes.UInt32(),
            get_dtype_backend(result.dtype, implementation),
            implementation,
            self.version,
        )
        return self.with_native(result.astype(dtype)).alias(self.native.name)
    
    def unique(self) -> PandasLikeSeries:
        native = self.native
        if is_dtype_pyarrow(self.native.dtype):
            import pyarrow as pa  # ignore-banned-import

            compliant = self.compliant
            ca = pa.chunked_array([compliant.to_arrow()])  # type: ignore[arg-type]
            result = (
                compliant._version.namespace.from_backend("pyarrow")
                .compliant.from_native(ca)
                .list.unique()
                .native
            )
            return self.with_native(native.__class__(
                result, dtype=native.dtype, index=native.index, name=native.name
            ))
        else:
            msg = (
                f"The `unique` method is not implemented for {self.implementation} backend."
            )
            raise NotImplementedError(msg)