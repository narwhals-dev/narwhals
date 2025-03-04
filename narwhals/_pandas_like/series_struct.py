from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.series import PandasLikeSeries


class PandasLikeSeriesStructNamespace:
    def __init__(self: Self, series: PandasLikeSeries) -> None:
        self._compliant_series = series

    def field(self: Self, name: str) -> PandasLikeSeries:
        return self._compliant_series._from_native_series(
            self._compliant_series._native_series.apply(lambda x: x[name]).rename(name),
        )
