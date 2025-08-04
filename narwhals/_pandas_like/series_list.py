from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant.any_namespace import ListNamespace
from narwhals._pandas_like.utils import (
    PandasLikeSeriesNamespace,
    get_dtype_backend,
    narwhals_to_native_dtype,
)
from narwhals._utils import not_implemented

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

    unique = not_implemented()

    contains = not_implemented()

    def get(self, index: int) -> PandasLikeSeries:
        native = self.native
        native_cls = type(native)

        import pyarrow.compute as pc

        from narwhals._arrow.utils import native_to_narwhals_dtype

        ca = native.array._pa_array
        result_arr = pc.list_element(ca, index)
        nw_dtype = native_to_narwhals_dtype(result_arr.type, self.version)
        out_dtype = narwhals_to_native_dtype(
            nw_dtype, "pyarrow", self.implementation, self.version
        )
        result_native = native_cls(
            result_arr, dtype=out_dtype, index=native.index, name=native.name
        )
        return self.with_native(result_native)
