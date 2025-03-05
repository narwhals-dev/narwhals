from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.series import PandasLikeSeries


class PandasLikeSeriesStructNamespace:
    def __init__(self: Self, series: PandasLikeSeries) -> None:
        self._compliant_series = series

    def field(self: Self, name: str) -> PandasLikeSeries:
        series = self._compliant_series._native_series

        if hasattr(series, "struct"):
            series = series.struct.field(name)
        else:
            series = series.apply(lambda x: x[name])

        return self._compliant_series._from_native_series(series.rename(name))
