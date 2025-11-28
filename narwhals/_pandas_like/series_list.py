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
    from typing import Literal

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
        result = self.native.list[index]
        result.name = self.native.name
        return self.with_native(result)

    def _agg(
        self, func: Literal["min", "max", "mean", "approximate_median", "sum"]
    ) -> PandasLikeSeries:
        dtype_backend = get_dtype_backend(
            self.native.dtype, self.compliant._implementation
        )
        if dtype_backend != "pyarrow":  # pragma: no cover
            msg = "Only pyarrow backend is currently supported."
            raise NotImplementedError(msg)

        from narwhals._arrow.utils import list_agg, native_to_narwhals_dtype

        ca = self.native.array._pa_array
        result_arr = list_agg(ca, func)
        nw_dtype = native_to_narwhals_dtype(result_arr.type, self.version)
        out_dtype = narwhals_to_native_dtype(
            nw_dtype, "pyarrow", self.implementation, self.version
        )
        result_native = type(self.native)(
            result_arr, dtype=out_dtype, index=self.native.index, name=self.native.name
        )
        return self.with_native(result_native)

    def min(self) -> PandasLikeSeries:
        return self._agg("min")

    def max(self) -> PandasLikeSeries:
        return self._agg("max")

    def mean(self) -> PandasLikeSeries:
        return self._agg("mean")

    def median(self) -> PandasLikeSeries:
        return self._agg("approximate_median")

    def sum(self) -> PandasLikeSeries:
        return self._agg("sum")
