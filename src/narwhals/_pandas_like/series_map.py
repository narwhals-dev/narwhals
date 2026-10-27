from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from narwhals._compliant.any_namespace import MapNamespace
from narwhals._pandas_like.utils import PandasLikeSeriesNamespace

if TYPE_CHECKING:
    from narwhals._pandas_like.series import PandasLikeSeries
    from narwhals.typing import NonNestedLiteral


class PandasLikeSeriesMapNamespace(
    PandasLikeSeriesNamespace, MapNamespace["PandasLikeSeries"]
):
    def get(self, key: NonNestedLiteral) -> PandasLikeSeries:
        from narwhals._arrow.utils import map_get

        result_native = self.compliant._apply_pyarrow_compute_func(
            self.native, partial(map_get, key=key)
        )
        return self.with_native(result_native)
