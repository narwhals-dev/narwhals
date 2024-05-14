from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.dtypes import to_narwhals_dtype
from narwhals.dtypes import translate_dtype
from narwhals.translate import get_cudf
from narwhals.translate import get_modin
from narwhals.translate import get_pandas
from narwhals.translate import get_polars

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

    from narwhals.dataframe import DataFrame


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
        if (pd := get_modin()) is not None and isinstance(
            series, pd.Series
        ):  # pragma: no cover
            self._series = PandasSeries(series, implementation="modin")
            return
        if (pd := get_cudf()) is not None and isinstance(
            series, pd.Series
        ):  # pragma: no cover
            self._series = PandasSeries(series, implementation="cudf")
            return
        msg = (  # pragma: no cover
            f"Expected pandas, Polars, modin, or cuDF Series, got: {type(series)}. "
            "If passing something which is not already a Series, but is convertible "
            "to one, you must specify `implementation=` "
            "(e.g. `nw.Series([1,2,3], implementation='polars')`)"
        )
        raise TypeError(msg)  # pragma: no cover

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._series.to_numpy(*args, **kwargs)

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, int):
            return self._series[idx]
        return self._from_series(self._series[idx])

    def __narwhals_namespace__(self) -> Any:
        if self._is_polars:
            import polars as pl

            return pl
        return self._series.__narwhals_namespace__()

    @property
    def shape(self) -> tuple[int]:
        """
        Get the shape of the Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     return s.shape

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            (3,)
            >>> func(s_pl)
            (3,)
        """
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
        """
        Get the data type of the Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     return s.dtype

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            Int64
            >>> func(s_pl)
            Int64
        """
        return to_narwhals_dtype(self._series.dtype, is_polars=self._is_polars)

    @property
    def name(self) -> str:
        """
        Get the name of the Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name="foo")
            >>> s_pl = pl.Series("foo", s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     return s.name

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            'foo'
            >>> func(s_pl)
            'foo'
        """
        return self._series.name  # type: ignore[no-any-return]

    def cast(
        self,
        dtype: Any,
    ) -> Self:
        return self._from_series(
            self._series.cast(translate_dtype(self.__narwhals_namespace__(), dtype))
        )

    def to_frame(self) -> DataFrame:
        """
        Convert to dataframe.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name='a')
            >>> s_pl = pl.Series('a', s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     df = s.to_frame()
            ...     return nw.to_native(df)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
               a
            0  1
            1  2
            2  3
            >>> func(s_pl)
            shape: (3, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            │ 2   │
            │ 3   │
            └─────┘
        """
        from narwhals.dataframe import DataFrame

        return DataFrame(self._series.to_frame())

    def mean(self) -> Any:
        """
        Reduce this Series to the mean value.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     return s.mean()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            2.0
            >>> func(s_pl)
            2.0
        """
        return self._series.mean()

    def any(self) -> Any:
        return self._series.any()

    def all(self) -> Any:
        return self._series.all()

    def min(self) -> Any:
        """
        Get the minimal value in this Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     return s.min()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            1
            >>> func(s_pl)
            1
        """
        return self._series.min()

    def max(self) -> Any:
        """
        Get the maximum value in this Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     return s.max()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            3
            >>> func(s_pl)
            3
        """
        return self._series.max()

    def sum(self) -> Any:
        """
        Reduce this Series to the sum value.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     return s.sum()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            6
            >>> func(s_pl)
            6
        """
        return self._series.sum()

    def std(self, *, ddof: int = 1) -> Any:
        return self._series.std(ddof=ddof)

    def is_in(self, other: Any) -> Self:
        return self._from_series(self._series.is_in(self._extract_native(other)))

    def drop_nulls(self) -> Self:
        """
        Drop all null values.

        See Also:
          drop_nans

        Notes:
          A null value is not the same as a NaN value.
          To drop NaN values, use :func:`drop_nans`.

        Examples:
          >>> import pandas as pd
          >>> import polars as pl
          >>> import numpy as np
          >>> import narwhals as nw
          >>> s_pd = pd.Series([2, 4, None, 3, 5])
          >>> s_pl = pl.Series('a', [2, 4, None, 3, 5])

          Now define a dataframe-agnostic function with a `column` argument for the column to evaluate :

          >>> def func(s_any):
          ...   s = nw.from_native(s_any, series_only=True)
          ...   s = s.drop_nulls()
          ...   return nw.to_native(s)

          Then we can pass either Series (polars or pandas) to `func`:

          >>> func(s_pd)
          0    2.0
          1    4.0
          3    3.0
          4    5.0
          dtype: float64
          >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
          shape: (4,)
          Series: 'a' [i64]
          [
             2
             4
             3
             5
          ]
        """
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

            We define a dataframe-agnostic function:

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
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
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
        """
        Returns unique values

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [2, 4, 4, 6]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            Let's define a dataframe-agnostic function:

            >>> def func(s_any):
            ...    s = nw.from_native(s_any, series_only=True)
            ...    s = s.unique()
            ...    return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    2
            1    4
            2    6
            dtype: int64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               2
               4
               6
            ]
        """
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

            We define a dataframe-agnostic function:

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
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               null
               2
               -1
            ]
        """
        return self._from_series(self._series.diff())

    def shift(self, n: int) -> Self:
        """
        Shift values by `n` positions.

        Notes:
            pandas may change the dtype here, for example when introducing missing
            values in an integer column. To ensure, that the dtype doesn't change,
            you may want to use `fill_null` and `cast`. For example, to shift
            and fill missing values with `0` in a Int64 column, you could
            do:

            ```python
            s.shift(1).fill_null(0).cast(nw.Int64)
            ```

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [2, 4, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a dataframe-agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.shift(1)
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    NaN
            1    2.0
            2    4.0
            dtype: float64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               null
               2
               4
            ]
        """
        return self._from_series(self._series.shift(n))

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
        """
        Rename the Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name="foo")
            >>> s_pl = pl.Series("foo", s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.alias("bar")
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    1
            1    2
            2    3
            Name: bar, dtype: int64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: 'bar' [i64]
            [
               1
               2
               3
            ]
        """
        return self._from_series(self._series.alias(name=name))

    def sort(self, *, descending: bool = False) -> Self:
        return self._from_series(self._series.sort(descending=descending))

    def is_null(self) -> Self:
        """
        Returns a boolean Series indicating which values are null.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, None]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a dataframe-agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.is_null()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    False
            1    False
            2     True
            dtype: bool
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [bool]
            [
               false
               false
               true
            ]
        """
        return self._from_series(self._series.is_null())

    def fill_null(self, value: Any) -> Self:
        """
        Fill null values using the specified value.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, None]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a dataframe-agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.fill_null(5)
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    1.0
            1    2.0
            2    5.0
            dtype: float64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               1
               2
               5
            ]
        """
        return self._from_series(self._series.fill_null(value))

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Self:
        return self._from_series(
            self._series.is_between(lower_bound, upper_bound, closed=closed)
        )

    def n_unique(self) -> int:
        """
        Count the number of unique values.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     return s.n_unique()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            3
            >>> func(s_pl)
            3
        """
        return self._series.n_unique()  # type: ignore[no-any-return]

    def to_numpy(self) -> Any:
        return self._series.to_numpy()

    def to_pandas(self) -> Any:
        return self._series.to_pandas()

    def __eq__(self, other: object) -> Series:  # type: ignore[override]
        return self._from_series(self._series.__eq__(self._extract_native(other)))

    def __ne__(self, other: object) -> Series:  # type: ignore[override]
        return self._from_series(self._series.__ne__(self._extract_native(other)))

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

            We define a dataframe-agnostic function:

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
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
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
        """
        Get the year in a date series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = [datetime(2012, 1, 7), datetime(2023, 3, 10)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.year()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    2012
            1    2023
            dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i32]
            [
               2012
               2023
            ]
        """
        return self._series.__class__(self._series._series.dt.year())

    def month(self) -> Series:
        """
        Gets the month in a date series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = [datetime(2023, 2, 1), datetime(2023, 8, 3)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.month()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    2
            1    8
            dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i8]
            [
               2
               8
            ]
        """
        return self._series.__class__(self._series._series.dt.month())

    def day(self) -> Series:
        """
        Extracts the day in a date series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = [datetime(2022, 1, 1), datetime(2022, 1, 5)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.day()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    1
            1    5
            dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i8]
            [
               1
               5
            ]
        """
        return self._series.__class__(self._series._series.dt.day())

    def hour(self) -> Series:
        """
         Extracts the hour in a date series.

        Examples:
             >>> import pandas as pd
             >>> import polars as pl
             >>> from datetime import datetime
             >>> import narwhals as nw
             >>> data = [datetime(2022, 1, 1, 5, 3), datetime(2022, 1, 5, 9, 12)]
             >>> s_pd = pd.Series(data)
             >>> s_pl = pl.Series(data)

             We define a library agnostic function:

             >>> def func(s_any):
             ...     s = nw.from_native(s_any, series_only=True)
             ...     s = s.dt.hour()
             ...     return nw.to_native(s)

             We can then pass either pandas or Polars to `func`:

             >>> func(s_pd)
             0    5
             1    9
             dtype: int...
             >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
             shape: (2,)
             Series: '' [i8]
             [
                5
                9
             ]
        """
        return self._series.__class__(self._series._series.dt.hour())

    def minute(self) -> Series:
        return self._series.__class__(self._series._series.dt.minute())

    def second(self) -> Series:
        return self._series.__class__(self._series._series.dt.second())

    def ordinal_day(self) -> Series:
        """
        Get ordinal day.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = [datetime(2020, 1, 1), datetime(2020, 8, 3)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.ordinal_day()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0      1
            1    216
            dtype: int32
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i16]
            [
               1
               216
            ]
        """
        return self._series.__class__(self._series._series.dt.ordinal_day())

    def total_minutes(self) -> Series:
        """
        Get total minutes.

        Notes:
            The function outputs the total minutes in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = [timedelta(minutes=10), timedelta(minutes=20, seconds=40)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.total_minutes()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    10
            1    20
            dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
                    10
                    20
            ]
        """
        return self._series.__class__(self._series._series.dt.total_minutes())

    def total_seconds(self) -> Series:
        """
        Get total seconds.

        Notes:
            The function outputs the total seconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = [timedelta(seconds=10), timedelta(seconds=20, milliseconds=40)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.total_seconds()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    10
            1    20
            dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
                    10
                    20
            ]
        """
        return self._series.__class__(self._series._series.dt.total_seconds())

    def total_milliseconds(self) -> Series:
        """
        Get total milliseconds.

        Notes:
            The function outputs the total milliseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = [timedelta(milliseconds=10),
            ...     timedelta(milliseconds=20, microseconds=40)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.total_milliseconds()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    10
            1    20
            dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
                    10
                    20
            ]
        """
        return self._series.__class__(self._series._series.dt.total_milliseconds())

    def total_microseconds(self) -> Series:
        """
        Get total microseconds.

        Notes:
            The function outputs the total microseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = [timedelta(microseconds=10),
            ...     timedelta(milliseconds=1, microseconds=200)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.total_microseconds()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0      10
            1    1200
            dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
                    10
                    1200
            ]
        """
        return self._series.__class__(self._series._series.dt.total_microseconds())

    def total_nanoseconds(self) -> Series:
        """
        Get total nanoseconds.

        Notes:
            The function outputs the total nanoseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = ['2024-01-01 00:00:00.000000001',
            ...     '2024-01-01 00:00:00.000000002']
            >>> s_pd = pd.to_datetime(pd.Series(data))
            >>> s_pl = pl.Series(data).str.to_datetime(time_unit='ns')

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.diff().dt.total_nanoseconds()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    NaN
            1    1.0
            dtype: float64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
                    null
                    1
            ]
        """
        return self._series.__class__(self._series._series.dt.total_nanoseconds())
