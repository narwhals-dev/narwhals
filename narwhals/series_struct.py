from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.series import Series

SeriesT = TypeVar("SeriesT", bound="Series[Any]")


class SeriesStructNamespace(Generic[SeriesT]):
    def __init__(self: Self, series: SeriesT) -> None:
        self._narwhals_series = series

    def field(self: Self, *names: str) -> SeriesT:
        r"""Return the length of each string as the number of characters.

        Returns:
            A new Series containing the length of each string in characters.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s_native = pl.Series(["foo", "345", None])
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.str.len_chars().to_native()  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [u32]
            [
                    3
                    3
                    null
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.struct.field(*names)
        )
