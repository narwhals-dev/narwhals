from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant.any_namespace import ListNamespace
from narwhals._pandas_like.utils import PandasLikeSeriesNamespace
from narwhals._pandas_like.utils import get_dtype_backend
from narwhals._pandas_like.utils import narwhals_to_native_dtype
from narwhals._pandas_like.utils import set_index
from narwhals.utils import import_dtypes_module

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
            result = set_index(
                result,
                self.native.index,
                implementation=implementation,
                backend_version=backend_version,
            )
        dtype = narwhals_to_native_dtype(
            import_dtypes_module(self.version).UInt32(),
            get_dtype_backend(result.dtype, implementation),
            implementation,
            backend_version,
            self.version,
        )
        return self.with_native(result.astype(dtype)).alias(self.native.name)
