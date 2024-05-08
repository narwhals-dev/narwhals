from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.dtypes import to_narwhals_dtype
from narwhals.dtypes import translate_dtype
from narwhals.translate import get_pandas
from narwhals.translate import get_polars

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class Series:
    def __init__(
        self,
        series: Any,
        *,
        is_polars: bool = False,
    ) -> None:
        from narwhals._pandas_like.series import PandasSeries

        self._is_polars = is_polars
        if hasattr(series, "__narwhals_series__"):
            self._series = series.__narwhals_series__()
            return
        if is_polars or (
            (pl := get_polars()) is not None and isinstance(series, pl.Series)
        ):
            self._series = series
            self._is_polars = True
            return
        if (pd := get_pandas()) is not None and isinstance(series, pd.Series):
            self._series = PandasSeries(series, implementation="pandas")
            return
        msg = f"Expected pandas or Polars Series, got: {type(series)}"  # pragma: no cover
        raise TypeError(msg)  # pragma: no cover

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._series.to_numpy(*args, **kwargs)

    def __getitem__(self, idx: int) -> Any:
        return self._series[idx]

    def __narwhals_namespace__(self) -> Any:
        if self._is_polars:
            import polars as pl

            return pl
        return self._series.__narwhals_namespace__()

    @property
    def shape(self) -> tuple[int]:
        return self._series.shape  # type: ignore[no-any-return]

    def _extract_native(self, arg: Any) -> Any:
        from narwhals.series import Series

        if isinstance(arg, Series):
            return arg._series
        return arg

    def _from_series(self, series: Any) -> Self:
        return self.__class__(series, is_polars=self._is_polars)

    def __repr__(self) -> str:  # pragma: no cover
        header = " Narwhals Series                                 "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Use `narwhals.to_native()` to see native output |\n"
            + "└"
            + "─" * length
            + "┘"
        )

    def __len__(self) -> int:
        return len(self._series)

    @property
    def dtype(self) -> Any:
        return to_narwhals_dtype(self._series.dtype, is_polars=self._is_polars)

    @property
    def name(self) -> str:
        return self._series.name  # type: ignore[no-any-return]

    def cast(
        self,
        dtype: Any,
    ) -> Self:
        return self._from_series(
            self._series.cast(translate_dtype(self.__narwhals_namespace__(), dtype))
        )

    def mean(self) -> Any:
        return self._series.mean()

    def any(self) -> Any:
        return self._series.any()

    def all(self) -> Any:
        return self._series.all()

    def min(self) -> Any:
        return self._series.min()

    def max(self) -> Any:
        return self._series.max()

    def sum(self) -> Any:
        return self._series.sum()

    def std(self, *, ddof: int = 1) -> Any:
        return self._series.std(ddof=ddof)

    def is_in(self, other: Any) -> Self:
        return self._from_series(self._series.is_in(self._extract_native(other)))

    def drop_nulls(self) -> Self:
        return self._from_series(self._series.drop_nulls())

    def cum_sum(self) -> Self:
        """
        Calculate the cumulative sum.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [2, 4, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a data-frame agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.cum_sum()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    2
            1    6
            2    9
            dtype: int64
            >>> func(s_pl)  # doctest:+SKIP
            shape: (3,)
            Series: '' [i64]
            [
               2
               6
               9
            ]
        """
        return self._from_series(self._series.cum_sum())

    def unique(self) -> Self:
        return self._from_series(self._series.unique())

    def diff(self) -> Self:
        """
        Calculate the difference with the previous element, for each element.

        Notes:
            pandas may change the dtype here, for example when introducing missing
            values in an integer column. To ensure, that the dtype doesn't change,
            you may want to use `fill_null` and `cast`. For example, to calculate
            the diff and fill missing values with `0` in a Int64 column, you could
            do:

            ```python
            s.diff().fill_null(0).cast(nw.Int64)
            ```

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [2, 4, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a data-frame agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.diff()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    NaN
            1    2.0
            2   -1.0
            dtype: float64
            >>> func(s_pl)  # doctest:+SKIP
            shape: (3,)
            Series: '' [i64]
            [
               null
               2
               -1
            ]
        """
        return self._from_series(self._series.diff())

    def sample(
        self,
        n: int | None = None,
        fraction: float | None = None,
        *,
        with_replacement: bool = False,
    ) -> Self:
        return self._from_series(
            self._series.sample(n=n, fraction=fraction, with_replacement=with_replacement)
        )

    def alias(self, name: str) -> Self:
        return self._from_series(self._series.alias(name=name))

    def sort(self, *, descending: bool = False) -> Self:
        return self._from_series(self._series.sort(descending=descending))

    def is_null(self) -> Self:
        return self._from_series(self._series.is_null())

    def fill_null(self, value: Any) -> Self:
        return self._from_series(self._series.fill_null(value))

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Self:
        return self._from_series(
            self._series.is_between(lower_bound, upper_bound, closed=closed)
        )

    def n_unique(self) -> int:
        return self._series.n_unique()  # type: ignore[no-any-return]

    def to_numpy(self) -> Any:
        return self._series.to_numpy()

    def to_pandas(self) -> Any:
        return self._series.to_pandas()

    def __gt__(self, other: Any) -> Series:
        return self._from_series(self._series.__gt__(self._extract_native(other)))

    def __ge__(self, other: Any) -> Series:  # pragma: no cover (todo)
        return self._from_series(self._series.__ge__(self._extract_native(other)))

    def __lt__(self, other: Any) -> Series:  # pragma: no cover (todo)
        return self._from_series(self._series.__lt__(self._extract_native(other)))

    def __le__(self, other: Any) -> Series:  # pragma: no cover (todo)
        return self._from_series(self._series.__le__(self._extract_native(other)))

    def __and__(self, other: Any) -> Series:  # pragma: no cover (todo)
        return self._from_series(self._series.__and__(self._extract_native(other)))

    def __or__(self, other: Any) -> Series:  # pragma: no cover (todo)
        return self._from_series(self._series.__or__(self._extract_native(other)))

    # unary
    def __invert__(self) -> Series:
        return self._from_series(self._series.__invert__())

    def filter(self, other: Any) -> Series:
        return self._from_series(self._series.filter(self._extract_native(other)))

    @property
    def str(self) -> SeriesStringNamespace:
        return SeriesStringNamespace(self)

    @property
    def dt(self) -> SeriesDateTimeNamespace:
        return SeriesDateTimeNamespace(self)


class SeriesStringNamespace:
    def __init__(self, series: Series) -> None:
        self._series = series

    def ends_with(self, suffix: str) -> Series:
        return self._series.__class__(self._series._series.str.ends_with(suffix))

    def head(self, n: int = 5) -> Series:
        """
        Take the first n elements of each string.

        Arguments:
            n: Number of elements to take.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> lyrics = ['Atatata', 'taata', 'taatatata', 'zukkyun']
            >>> s_pd = pd.Series(lyrics)
            >>> s_pl = pl.Series(lyrics)

            We define a data-frame agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.str.head()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    Atata
            1    taata
            2    taata
            3    zukky
            dtype: object
            >>> func(s_pl)  # doctest: +SKIP
            shape: (2,)
            Series: '' [str]
            [
               "Atata"
               "taata"
               "taata"
               "zukky"
            ]
        """
        if self._series._is_polars:
            return self._series.__class__(self._series._series.str.slice(0, n))
        return self._series.__class__(self._series._series.str.head(n))


class SeriesDateTimeNamespace:
    def __init__(self, series: Series) -> None:
        self._series = series

    def year(self) -> Series:
        return self._series.__class__(self._series._series.dt.year())
