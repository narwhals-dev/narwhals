from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from narwhals.typing import SeriesT

if TYPE_CHECKING:
    from narwhals.typing import NonNestedLiteral


class SeriesMapNamespace(Generic[SeriesT]):
    def __init__(self, series: SeriesT) -> None:
        self._narwhals_series = series

    def get(self, key: NonNestedLiteral) -> SeriesT:
        """Return the value for `key` in each map.

        The result is null where the map is null or does not contain `key`.

        Arguments:
            key: Key to look up.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s_native = pl.Series(
            ...     [{"x": 1, "y": 2}, {"y": 3}, None], dtype=pl.Map(pl.String, pl.Int64)
            ... )
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.map.get("y").to_list()
            [2, 3, None]
        """
        return self._narwhals_series._with_compliant(
            self._narwhals_series._compliant_series.map.get(key)
        )
