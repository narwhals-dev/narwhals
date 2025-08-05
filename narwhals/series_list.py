from __future__ import annotations

from typing import Generic

from narwhals.typing import SeriesT


class SeriesListNamespace(Generic[SeriesT]):
    def __init__(self, series: SeriesT) -> None:
        self._narwhals_series = series

    def len(self) -> SeriesT:
        """Return the number of elements in each list.

        Null values count towards the total.

        Returns:
            A new series.

        Examples:
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> s_native = pa.chunked_array([[[1, 2], [3, 4, None], None, []]])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.list.len().to_native()  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2,
                3,
                null,
                0
              ]
            ]
        """
        return self._narwhals_series._with_compliant(
            self._narwhals_series._compliant_series.list.len()
        )

    def unique(self) -> SeriesT:
        """Get the unique/distinct values in the list.

        Null values are included in the result. The order of unique values is not guaranteed.

        Returns:
            A new series.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s_native = pl.Series([[1, 1, 2], [3, 3, None], None, []])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.list.unique().to_native()
            shape: (4,)
            Series: '' [list[i64]]
            [
                    [1, 2]
                    [null, 3]
                    null
                    []
            ]
        """
        return self._narwhals_series._with_compliant(
            self._narwhals_series._compliant_series.list.unique()
        )
