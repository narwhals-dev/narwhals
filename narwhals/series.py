from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

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
            return get_polars()
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
        """
        Cast between data types.

        Arguments:
            dtype: Data type that the object will be cast into.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [True, False, True]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a dataframe-agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.cast(nw.Int64)
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    1
            1    0
            2    1
            dtype: int64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               1
               0
               1
            ]
        """
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
        """
        Return whether any of the values in the Series are True.

        Notes:
          Only works on Series of data type Boolean.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [False, True, False]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     return s.any()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            True
            >>> func(s_pl)
            True
        """
        return self._series.any()

    def all(self) -> Any:
        """
        Return whether all values in the Series are True.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [True, False, True]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     return s.all()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            False
            >>> func(s_pl)
            False

        """
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
        """
        Get the standard deviation of this Series.

        Arguments:
            ddof: “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
                     where N represents the number of elements.

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
            ...     return s.std()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            1.0
            >>> func(s_pl)
            1.0
        """
        return self._series.std(ddof=ddof)

    def is_in(self, other: Any) -> Self:
        """
        Check if the elements of this Series are in the other sequence.

        Arguments:
            other: Sequence of primitive type.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s_pd = pd.Series([1, 2, 3])
            >>> s_pl = pl.Series([1, 2, 3])

            We define a library agnostic function:

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.is_in([3, 2, 8])

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    False
            1     True
            2     True
            dtype: bool
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [bool]
            [
               false
               true
               true
            ]
        """
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

          >>> @nw.narwhalify(series_only=True)
          ... def func(s):
          ...   return s.drop_nulls()

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

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.cum_sum()

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

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...    return s.unique()

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

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.diff()

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

        Arguments:
            n: Number of indices to shift forward. If a negative value is passed,
                values are shifted in the opposite direction instead.

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

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.shift(1)

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
        """
        Sample randomly from this Series.

        Arguments:
            n: Number of items to return. Cannot be used with fraction.

            fraction: Fraction of items to return. Cannot be used with n.

            with_replacement: Allow values to be sampled more than once.

        Notes:
            The `sample` method returns a Series with a specified number of
            randomly selected items chosen from this Series.
            The results are not consistent across libraries.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl

            >>> s_pd = pd.Series([1, 2, 3, 4])
            >>> s_pl = pl.Series([1, 2, 3, 4])

            We define a library agnostic function:

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.sample(fraction=1.0, with_replacement=True)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest:+SKIP
               a
            2  3
            1  2
            3  4
            3  4
            >>> func(s_pl)  # doctest:+SKIP
            shape: (4,)
            Series: '' [i64]
            [
               1
               4
               3
               4
            ]
        """
        return self._from_series(
            self._series.sample(n=n, fraction=fraction, with_replacement=with_replacement)
        )

    def alias(self, name: str) -> Self:
        """
        Rename the Series.

        Arguments:
            name: The new name.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name="foo")
            >>> s_pl = pl.Series("foo", s)

            We define a library agnostic function:

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.alias("bar")

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
        """
        Sort this Series. Place null values first.

        Arguments:
            descending: Sort in descending order.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [5, None, 1, 2]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define library agnostic functions:

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.sort()

            >>> @nw.narwhalify(series_only=True)
            ... def func_descend(s):
            ...     return s.sort(descending=True)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            1    NaN
            2    1.0
            3    2.0
            0    5.0
            dtype: float64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
               null
               1
               2
               5
            ]
            >>> func_descend(s_pd)
            1    NaN
            0    5.0
            3    2.0
            2    1.0
            dtype: float64
            >>> func_descend(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
               null
               5
               2
               1
            ]
        """
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

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.is_null()

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

        Arguments:
            value: Value used to fill null values.

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

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.fill_null(5)

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
        """
        Get a boolean mask of the values that are between the given lower/upper bounds.

        Arguments:
            lower_bound: Lower bound value.

            upper_bound: Upper bound value.

            closed: Define which sides of the interval are closed (inclusive).

        Notes:
            If the value of the `lower_bound` is greater than that of the `upper_bound`,
            then the values will be False, as no value can satisfy the condition.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s_pd = pd.Series([1, 2, 3, 4, 5])
            >>> s_pl = pl.Series([1, 2, 3, 4, 5])

            We define a library agnostic function:

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.is_between(2, 4, 'right')

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    False
            1    False
            2     True
            3     True
            4    False
            dtype: bool
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [bool]
            [
               false
               false
               true
               true
               false
            ]
        """
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
        """
        Convert to numpy.

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
            ...     df = s.to_numpy()
            ...     return df

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            array([1, 2, 3]...)
            >>> func(s_pl)
            array([1, 2, 3]...)
        """
        return self._series.to_numpy()

    def to_pandas(self) -> Any:
        """
        Convert to pandas.

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
            ...     df = s.to_pandas()
            ...     return df

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    1
            1    2
            2    3
            Name: a, dtype: int64
            >>> func(s_pl)
            0    1
            1    2
            2    3
            Name: a, dtype: int64
        """
        return self._series.to_pandas()

    def __add__(self, other: object) -> Series:
        return self._from_series(self._series.__add__(self._extract_native(other)))

    def __radd__(self, other: object) -> Series:
        return self._from_series(self._series.__radd__(self._extract_native(other)))

    def __sub__(self, other: object) -> Series:
        return self._from_series(self._series.__sub__(self._extract_native(other)))

    def __rsub__(self, other: object) -> Series:
        return self._from_series(self._series.__rsub__(self._extract_native(other)))

    def __mul__(self, other: object) -> Series:
        return self._from_series(self._series.__mul__(self._extract_native(other)))

    def __rmul__(self, other: object) -> Series:
        return self._from_series(self._series.__rmul__(self._extract_native(other)))

    def __truediv__(self, other: object) -> Series:
        return self._from_series(self._series.__truediv__(self._extract_native(other)))

    def __floordiv__(self, other: object) -> Series:
        return self._from_series(self._series.__floordiv__(self._extract_native(other)))

    def __rfloordiv__(self, other: object) -> Series:
        return self._from_series(self._series.__rfloordiv__(self._extract_native(other)))

    def __pow__(self, other: object) -> Series:
        return self._from_series(self._series.__pow__(self._extract_native(other)))

    def __rpow__(self, other: object) -> Series:
        return self._from_series(self._series.__rpow__(self._extract_native(other)))

    def __mod__(self, other: object) -> Series:
        return self._from_series(self._series.__mod__(self._extract_native(other)))

    def __rmod__(self, other: object) -> Series:
        return self._from_series(self._series.__rmod__(self._extract_native(other)))

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
        """
        Filter elements in the Series based on a condition.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [4, 10, 15, 34, 50]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.filter(s > 10)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            2    15
            3    34
            4    50
            dtype: int64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               15
               34
               50
            ]
        """
        return self._from_series(self._series.filter(self._extract_native(other)))

    # --- descriptive ---
    def is_duplicated(self: Self) -> Series:
        r"""
        Get a mask of all duplicated rows in the Series.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, 2, 3, 1])
            >>> s_pl = pl.Series([1, 2, 3, 1])

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.is_duplicated()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            0     True
            1    False
            2    False
            3     True
            dtype: bool
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
                true
                false
                false
                true
            ]
        """
        return Series(self._series.is_duplicated())

    def is_empty(self: Self) -> bool:
        r"""
        Check if the series is empty.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl

            Let's define a dataframe-agnostic function that filters rows in which "foo"
            values are greater than 10, and then checks if the result is empty or not:

            >>> def func(s_any):
            ...     series = nw.from_native(s_any, allow_series=True)
            ...     return series.filter(series > 10).is_empty()

            We can then pass either pandas or Polars to `func`:

            >>> s_pd = pd.Series([1, 2, 3])
            >>> s_pl = pl.Series([1, 2, 3])
            >>> func(s_pd), func(s_pl)
            (True, True)

            >>> s_pd = pd.Series([100, 2, 3])
            >>> s_pl = pl.Series([100, 2, 3])
            >>> func(s_pd), func(s_pl)
            (False, False)
        """
        return self._series.is_empty()  # type: ignore[no-any-return]

    def is_unique(self: Self) -> Series:
        r"""
        Get a mask of all unique rows in the Series.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, 2, 3, 1])
            >>> s_pl = pl.Series([1, 2, 3, 1])

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.is_unique()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            0    False
            1     True
            2     True
            3    False
            dtype: bool

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
                false
                 true
                 true
                false
            ]
        """
        return Series(self._series.is_unique())

    def null_count(self: Self) -> int:
        r"""
        Create a new Series that shows the null counts per column.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, None, 3])
            >>> s_pl = pl.Series([1, None, None])

            Let's define a dataframe-agnostic function that returns the null count of
            the series:

            >>> def func(s_any):
            ...     series = nw.from_native(s_any, allow_series=True)
            ...     return series.null_count()

            We can then pass either pandas or Polars to `func`:
            >>> func(s_pd)
            1
            >>> func(s_pl)
            2
        """

        return self._series.null_count()  # type: ignore[no-any-return]

    def is_first_distinct(self: Self) -> Series:
        r"""
        Return a boolean mask indicating the first occurrence of each distinct value.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, 1, 2, 3, 2])
            >>> s_pl = pl.Series([1, 1, 2, 3, 2])

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.is_first_distinct()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            0     True
            1    False
            2     True
            3     True
            4    False
            dtype: bool

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [bool]
            [
                true
                false
                true
                true
                false
            ]
        """
        return Series(self._series.is_first_distinct())

    def is_last_distinct(self: Self) -> Series:
        r"""
        Return a boolean mask indicating the last occurrence of each distinct value.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, 1, 2, 3, 2])
            >>> s_pl = pl.Series([1, 1, 2, 3, 2])

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.is_last_distinct()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            0    False
            1     True
            2    False
            3     True
            4     True
            dtype: bool

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [bool]
            [
                false
                true
                false
                true
                true
            ]
        """
        return Series(self._series.is_last_distinct())

    def is_sorted(self: Self, *, descending: bool = False) -> bool:
        r"""
        Check if the Series is sorted.

        Arguments:
            descending: Check if the Series is sorted in descending order.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> unsorted_data = [1, 3, 2]
            >>> sorted_data = [3, 2, 1]

            Let's define a dataframe-agnostic function:

            >>> def func(s_any, descending=False):
            ...     series = nw.from_native(s_any, allow_series=True)
            ...     return series.is_sorted(descending=descending)

            We can then pass either pandas or Polars to `func`:

            >>> func(pl.Series(unsorted_data))
            False
            >>> func(pl.Series(sorted_data), descending=True)
            True
            >>> func(pd.Series(unsorted_data))
            False
            >>> func(pd.Series(sorted_data), descending=True)
            True
        """
        return self._series.is_sorted(descending=descending)  # type: ignore[no-any-return]

    def value_counts(
        self: Self, *, sort: bool = False, parallel: bool = False
    ) -> DataFrame:
        r"""
        Count the occurrences of unique values.

        Arguments:
            sort: Sort the output by count in descending order. If set to False (default),
                the order of the output is random.
            parallel: Execute the computation in parallel. Unused for pandas-like APIs.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, 1, 2, 3, 2], name="s")
            >>> s_pl = pl.Series(values=[1, 1, 2, 3, 2], name="s")

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.value_counts(sort=True)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
               s  count
            0  1      2
            1  2      2
            2  3      1

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3, 2)
            ┌─────┬───────┐
            │ s   ┆ count │
            │ --- ┆ ---   │
            │ i64 ┆ u32   │
            ╞═════╪═══════╡
            │ 1   ┆ 2     │
            │ 2   ┆ 2     │
            │ 3   ┆ 1     │
            └─────┴───────┘
        """
        from narwhals.dataframe import DataFrame

        return DataFrame(self._series.value_counts(sort=sort, parallel=parallel))

    def quantile(
        self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Any:
        """
        Get quantile value of the series.

        Note:
            pandas and Polars may have implementation differences for a given interpolation method.

        Arguments:
            quantile : float
                Quantile between 0.0 and 1.0.
            interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
                Interpolation method.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = list(range(50))
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            Let's define a dataframe-agnostic function:

            >>> def func(s_any):
            ...     series = nw.from_native(s_any, allow_series=True)
            ...     return [
            ...         series.quantile(quantile=q, interpolation='nearest')
            ...         for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            ...         ]

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            [5, 12, 24, 37, 44]

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            [5.0, 12.0, 25.0, 37.0, 44.0]
        """
        return self._series.quantile(quantile=quantile, interpolation=interpolation)

    def zip_with(self, mask: Any, other: Any) -> Self:
        """
        Take values from self or other based on the given mask. Where mask evaluates true, take values from self. Where mask evaluates false, take values from other.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s1_pl = pl.Series([1, 2, 3, 4, 5])
            >>> s2_pl = pl.Series([5, 4, 3, 2, 1])
            >>> mask_pl = pl.Series([True, False, True, False, True])
            >>> s1_pd = pd.Series([1, 2, 3, 4, 5])
            >>> s2_pd = pd.Series([5, 4, 3, 2, 1])
            >>> mask_pd = pd.Series([True, False, True, False, True])

            Let's define a dataframe-agnostic function:

            >>> def func(s1_any, mask_any, s2_any):
            ...     s1 = nw.from_native(s1_any, allow_series=True)
            ...     mask = nw.from_native(mask_any, series_only=True)
            ...     s2 = nw.from_native(s2_any, series_only=True)
            ...     s = s1.zip_with(mask, s2)
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s1_pl, mask_pl, s2_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [i64]
            [
               1
               4
               3
               2
               5
            ]
            >>> func(s1_pd, mask_pd, s2_pd)
            0    1
            1    4
            2    3
            3    2
            4    5
            dtype: int64
        """

        return self._from_series(
            self._series.zip_with(self._extract_native(mask), self._extract_native(other))
        )

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

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.str.head()

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
        Get the year in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2012, 1, 7), datetime(2023, 3, 10)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

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
        Gets the month in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2023, 2, 1), datetime(2023, 8, 3)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

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
        Extracts the day in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2022, 1, 1), datetime(2022, 1, 5)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

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
         Extracts the hour in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2022, 1, 1, 5, 3), datetime(2022, 1, 5, 9, 12)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

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
        """
        Extracts the minute in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2022, 1, 1, 5, 3), datetime(2022, 1, 5, 9, 12)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.minute()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0     3
            1    12
            dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i8]
            [
               3
               12
            ]
        """
        return self._series.__class__(self._series._series.dt.minute())

    def second(self) -> Series:
        """
        Extracts the second(s) in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2022, 1, 1, 5, 3, 10), datetime(2022, 1, 5, 9, 12, 4)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.second()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    10
            1     4
            dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i8]
            [
               10
                4
            ]
        """
        return self._series.__class__(self._series._series.dt.second())

    def millisecond(self) -> Series:
        """
        Extracts the milliseconds in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [
            ...     datetime(2023, 5, 21, 12, 55, 10, 400000),
            ...     datetime(2023, 5, 21, 12, 55, 10, 600000),
            ...     datetime(2023, 5, 21, 12, 55, 10, 800000),
            ...     datetime(2023, 5, 21, 12, 55, 11, 0),
            ...     datetime(2023, 5, 21, 12, 55, 11, 200000)
            ... ]

            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.millisecond().alias("datetime")
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    400
            1    600
            2    800
            3      0
            4    200
            Name: datetime, dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: 'datetime' [i32]
            [
                400
                600
                800
                0
                200
            ]
        """
        return self._series.__class__(self._series._series.dt.millisecond())

    def microsecond(self) -> Series:
        """
        Extracts the microseconds in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [
            ...     datetime(2023, 5, 21, 12, 55, 10, 400000),
            ...     datetime(2023, 5, 21, 12, 55, 10, 600000),
            ...     datetime(2023, 5, 21, 12, 55, 10, 800000),
            ...     datetime(2023, 5, 21, 12, 55, 11, 0),
            ...     datetime(2023, 5, 21, 12, 55, 11, 200000)
            ... ]

            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.microsecond().alias("datetime")
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    400000
            1    600000
            2    800000
            3         0
            4    200000
            Name: datetime, dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: 'datetime' [i32]
            [
               400000
               600000
               800000
               0
               200000
            ]
        """
        return self._series.__class__(self._series._series.dt.microsecond())

    def nanosecond(self) -> Series:
        """
        Extracts the nanosecond(s) in a date series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2022, 1, 1, 5, 3, 10, 500000), datetime(2022, 1, 5, 9, 12, 4, 60000)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> def func(s_any):
            ...     s = nw.from_native(s_any, series_only=True)
            ...     s = s.dt.nanosecond()
            ...     return nw.to_native(s)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    500000000
            1     60000000
            dtype: int...
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i32]
            [
               500000000
               60000000
            ]
        """
        return self._series.__class__(self._series._series.dt.nanosecond())

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
