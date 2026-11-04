from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

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

        native = self.native
        result = map_get(native.array._pa_array, key)
        # A round trip through narwhals dtypes would garble dictionary value types.
        result_native = type(native)(
            result, dtype=pd.ArrowDtype(result.type), index=native.index, name=native.name
        )
        return self.with_native(result_native)
