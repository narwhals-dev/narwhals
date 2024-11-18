from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterator
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import TypeVar
from typing import overload

from narwhals.utils import parse_version

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals.dataframe import DataFrame
    from narwhals.dtypes import DType


class Series:
    """Narwhals Series, backed by a native series.

    The native series might be pandas.Series, polars.Series, ...

    This class is not meant to be instantiated directly - instead, use
    `narwhals.from_native`, making sure to pass `allow_series=True` or
    `series_only=True`.
    """

    @property
    def _dataframe(self) -> type[DataFrame[Any]]:
        from narwhals.dataframe import DataFrame

        return DataFrame

    def __init__(
        self: Self,
        series: Any,
        *,
        level: Literal["full", "interchange"],
    ) -> None:
        self._level = level
        if hasattr(series, "__narwhals_series__"):
            self._compliant_series = series.__narwhals_series__()
        else:  # pragma: no cover
            msg = f"Expected Polars Series or an object which implements `__narwhals_series__`, got: {type(series)}."
            raise AssertionError(msg)

    def __array__(self: Self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        return self._compliant_series.__array__(dtype=dtype, copy=copy)

    @overload
    def __getitem__(self: Self, idx: int) -> Any: ...

    @overload
    def __getitem__(self: Self, idx: slice | Sequence[int]) -> Self: ...

    def __getitem__(self: Self, idx: int | slice | Sequence[int]) -> Any | Self:
        if isinstance(idx, int):
            return self._compliant_series[idx]
        return self._from_compliant_series(self._compliant_series[idx])

    def __native_namespace__(self: Self) -> ModuleType:
        return self._compliant_series.__native_namespace__()  # type: ignore[no-any-return]

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        """Export a Series via the Arrow PyCapsule Interface.

        Narwhals doesn't implement anything itself here:

        - if the underlying series implements the interface, it'll return that
        - else, it'll call `to_arrow` and then defer to PyArrow's implementation

        See [PyCapsule Interface](https://arrow.apache.org/docs/dev/format/CDataInterface/PyCapsuleInterface.html)
        for more.
        """
        native_series = self._compliant_series._native_series
        if hasattr(native_series, "__arrow_c_stream__"):
            return native_series.__arrow_c_stream__(requested_schema=requested_schema)
        try:
            import pyarrow as pa  # ignore-banned-import
        except ModuleNotFoundError as exc:  # pragma: no cover
            msg = f"PyArrow>=16.0.0 is required for `Series.__arrow_c_stream__` for object of type {type(native_series)}"
            raise ModuleNotFoundError(msg) from exc
        if parse_version(pa.__version__) < (16, 0):  # pragma: no cover
            msg = f"PyArrow>=16.0.0 is required for `Series.__arrow_c_stream__` for object of type {type(native_series)}"
            raise ModuleNotFoundError(msg)
        ca = pa.chunked_array([self.to_arrow()])
        return ca.__arrow_c_stream__(requested_schema=requested_schema)

    def to_native(self) -> Any:
        """Convert Narwhals series to native series.

        Returns:
            Series of class that user started with.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.to_native()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    1
            1    2
            2    3
            dtype: int64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
                1
                2
                3
            ]
        """
        return self._compliant_series._native_series

    def scatter(self, indices: int | Sequence[int], values: Any) -> Self:
        """Set value(s) at given position(s).

        Arguments:
            indices: Position(s) to set items at.
            values: Values to set.

        Note:
            This method always returns a new Series, without modifying the original one.
            Using this function in a for-loop is an anti-pattern, we recommend building
            up your positions and values beforehand and doing an update in one go.

            For example, instead of

            ```python
            for i in [1, 3, 2]:
                value = some_function(i)
                s = s.scatter(i, value)
            ```

            prefer

            ```python
            positions = [1, 3, 2]
            values = [some_function(x) for x in positions]
            s = s.scatter(positions, values)
            ```

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(df["a"].scatter([0, 1], [999, 888]))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                 a  b
            0  999  4
            1  888  5
            2    3  6
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 999 ┆ 4   │
            │ 888 ┆ 5   │
            │ 3   ┆ 6   │
            └─────┴─────┘
        """
        return self._from_compliant_series(
            self._compliant_series.scatter(indices, self._extract_native(values))
        )

    @property
    def shape(self) -> tuple[int]:
        """Get the shape of the Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.shape

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            (3,)
            >>> func(s_pl)
            (3,)
        """
        return self._compliant_series.shape  # type: ignore[no-any-return]

    def _extract_native(self, arg: Any) -> Any:
        from narwhals.series import Series

        if isinstance(arg, Series):
            return arg._compliant_series
        return arg

    def _from_compliant_series(self, series: Any) -> Self:
        return self.__class__(
            series,
            level=self._level,
        )

    def pipe(self, function: Callable[[Any], Self], *args: Any, **kwargs: Any) -> Self:
        """Pipe function call.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> s_pd = pd.Series([1, 2, 3, 4])
            >>> s_pl = pl.Series([1, 2, 3, 4])

            Lets define a function to pipe into
            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.pipe(lambda x: x + 2)

            Now apply it to the series

            >>> func(s_pd)
            0    3
            1    4
            2    5
            3    6
            dtype: int64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
               3
               4
               5
               6
            ]


        """
        return function(self, *args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        header = " Narwhals Series                         "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Use `.to_native()` to see native output |\n"
            + "└"
            + "─" * length
            + "┘"
        )

    def __len__(self) -> int:
        return len(self._compliant_series)

    def len(self) -> int:
        r"""Return the number of elements in the Series.

        Null values count towards the total.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = [1, 2, None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            Let's define a dataframe-agnostic function that computes the len of the series:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.len()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            3
            >>> func(s_pl)
            3
        """
        return len(self._compliant_series)

    @property
    def dtype(self: Self) -> DType:
        """Get the data type of the Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dtype

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            Int64
            >>> func(s_pl)
            Int64
        """
        return self._compliant_series.dtype  # type: ignore[no-any-return]

    @property
    def name(self) -> str:
        """Get the name of the Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name="foo")
            >>> s_pl = pl.Series("foo", s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.name

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            'foo'
            >>> func(s_pl)
            'foo'
        """
        return self._compliant_series.name  # type: ignore[no-any-return]

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        """Cast between data types.

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

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.cast(nw.Int64)

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
        return self._from_compliant_series(self._compliant_series.cast(dtype))

    def to_frame(self) -> DataFrame[Any]:
        """Convert to dataframe.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name="a")
            >>> s_pl = pl.Series("a", s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.to_frame()

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
        return self._dataframe(
            self._compliant_series.to_frame(),
            level=self._level,
        )

    def to_list(self) -> list[Any]:
        """Convert to list.

        Notes:
            This function converts to Python scalars. It's typically
            more efficient to keep your data in the format native to
            your original dataframe, so we recommend only calling this
            when you absolutely need to.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name="a")
            >>> s_pl = pl.Series("a", s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.to_list()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            [1, 2, 3]
            >>> func(s_pl)
            [1, 2, 3]
        """
        return self._compliant_series.to_list()  # type: ignore[no-any-return]

    def mean(self) -> Any:
        """Reduce this Series to the mean value.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.mean()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            np.float64(2.0)
            >>> func(s_pl)
            2.0
        """
        return self._compliant_series.mean()

    def median(self) -> Any:
        """Reduce this Series to the median value.

        Notes:
            Results might slightly differ across backends due to differences in the underlying algorithms used to compute the median.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> s = [5, 3, 8]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)
            >>> s_pa = pa.chunked_array([s])

            Let's define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.median()

            We can then pass any supported library such as pandas, Polars, or PyArrow to `func`:

            >>> func(s_pd)
            np.float64(5.0)
            >>> func(s_pl)
            5.0
            >>> func(s_pa)
            <pyarrow.DoubleScalar: 5.0>
        """
        return self._compliant_series.median()

    def count(self) -> Any:
        """Returns the number of non-null elements in the Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.count()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            np.int64(3)
            >>> func(s_pl)
            3

        """
        return self._compliant_series.count()

    def any(self) -> Any:
        """Return whether any of the values in the Series are True.

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

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.any()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            np.True_
            >>> func(s_pl)
            True
        """
        return self._compliant_series.any()

    def all(self) -> Any:
        """Return whether all values in the Series are True.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [True, False, True]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.all()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            np.False_
            >>> func(s_pl)
            False

        """
        return self._compliant_series.all()

    def min(self) -> Any:
        """Get the minimal value in this Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.min()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            np.int64(1)
            >>> func(s_pl)
            1
        """
        return self._compliant_series.min()

    def max(self) -> Any:
        """Get the maximum value in this Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.max()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            np.int64(3)
            >>> func(s_pl)
            3
        """
        return self._compliant_series.max()

    def sum(self) -> Any:
        """Reduce this Series to the sum value.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.sum()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            np.int64(6)
            >>> func(s_pl)
            6
        """
        return self._compliant_series.sum()

    def std(self, *, ddof: int = 1) -> Any:
        """Get the standard deviation of this Series.

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

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.std()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            np.float64(1.0)
            >>> func(s_pl)
            1.0
        """
        return self._compliant_series.std(ddof=ddof)

    def clip(
        self, lower_bound: Any | None = None, upper_bound: Any | None = None
    ) -> Self:
        r"""Clip values in the Series.

        Arguments:
            lower_bound: Lower bound value.
            upper_bound: Upper bound value.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>>
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func_lower(s):
            ...     return s.clip(2)

            We can then pass either pandas or Polars to `func_lower`:

            >>> func_lower(s_pd)
            0    2
            1    2
            2    3
            dtype: int64
            >>> func_lower(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               2
               2
               3
            ]

            We define another library agnostic function:

            >>> @nw.narwhalify
            ... def func_upper(s):
            ...     return s.clip(upper_bound=2)

            We can then pass either pandas or Polars to `func_upper`:

            >>> func_upper(s_pd)
            0    1
            1    2
            2    2
            dtype: int64
            >>> func_upper(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               1
               2
               2
            ]

            We can have both at the same time

            >>> s = [-1, 1, -3, 3, -5, 5]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.clip(-1, 3)

            We can pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0   -1
            1    1
            2   -1
            3    3
            4   -1
            5    3
            dtype: int64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (6,)
            Series: '' [i64]
            [
               -1
                1
               -1
                3
               -1
                3
            ]
        """
        return self._from_compliant_series(
            self._compliant_series.clip(lower_bound=lower_bound, upper_bound=upper_bound)
        )

    def is_in(self, other: Any) -> Self:
        """Check if the elements of this Series are in the other sequence.

        Arguments:
            other: Sequence of primitive type.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s_pd = pd.Series([1, 2, 3])
            >>> s_pl = pl.Series([1, 2, 3])

            We define a library agnostic function:

            >>> @nw.narwhalify
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
        return self._from_compliant_series(
            self._compliant_series.is_in(self._extract_native(other))
        )

    def arg_true(self) -> Self:
        """Find elements where boolean Series is True.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = [1, None, None, 2]
            >>> s_pd = pd.Series(data, name="a")
            >>> s_pl = pl.Series("a", data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.is_null().arg_true()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            1    1
            2    2
            Name: a, dtype: int64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: 'a' [u32]
            [
               1
               2
            ]
        """
        return self._from_compliant_series(self._compliant_series.arg_true())

    def drop_nulls(self) -> Self:
        """Drop all null values.

        Notes:
          pandas and Polars handle null values differently. Polars distinguishes
          between NaN and Null, whereas pandas doesn't.

        Examples:
          >>> import pandas as pd
          >>> import polars as pl
          >>> import numpy as np
          >>> import narwhals as nw
          >>> s_pd = pd.Series([2, 4, None, 3, 5])
          >>> s_pl = pl.Series("a", [2, 4, None, 3, 5])

          Now define a dataframe-agnostic function with a `column` argument for the column to evaluate :

          >>> @nw.narwhalify
          ... def func(s):
          ...     return s.drop_nulls()

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
        return self._from_compliant_series(self._compliant_series.drop_nulls())

    def abs(self) -> Self:
        """Calculate the absolute value of each element.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [2, -4, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.abs()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    2
            1    4
            2    3
            dtype: int64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               2
               4
               3
            ]
        """
        return self._from_compliant_series(self._compliant_series.abs())

    def cum_sum(self: Self, *, reverse: bool = False) -> Self:
        """Calculate the cumulative sum.

        Arguments:
            reverse: reverse the operation

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [2, 4, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
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
        return self._from_compliant_series(
            self._compliant_series.cum_sum(reverse=reverse)
        )

    def unique(self, *, maintain_order: bool = False) -> Self:
        """Returns unique values of the series.

        Arguments:
            maintain_order: Keep the same order as the original series. This may be more
                expensive to compute. Settings this to `True` blocks the possibility
                to run on the streaming engine for Polars.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [2, 4, 4, 6]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.unique(maintain_order=True)

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
        return self._from_compliant_series(
            self._compliant_series.unique(maintain_order=maintain_order)
        )

    def diff(self) -> Self:
        """Calculate the difference with the previous element, for each element.

        Notes:
            pandas may change the dtype here, for example when introducing missing
            values in an integer column. To ensure, that the dtype doesn't change,
            you may want to use `fill_null` and `cast`. For example, to calculate
            the diff and fill missing values with `0` in a Int64 column, you could
            do:

                s.diff().fill_null(0).cast(nw.Int64)

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [2, 4, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
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
        return self._from_compliant_series(self._compliant_series.diff())

    def shift(self, n: int) -> Self:
        """Shift values by `n` positions.

        Arguments:
            n: Number of indices to shift forward. If a negative value is passed,
                values are shifted in the opposite direction instead.

        Notes:
            pandas may change the dtype here, for example when introducing missing
            values in an integer column. To ensure, that the dtype doesn't change,
            you may want to use `fill_null` and `cast`. For example, to shift
            and fill missing values with `0` in a Int64 column, you could
            do:

                s.shift(1).fill_null(0).cast(nw.Int64)

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [2, 4, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
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
        return self._from_compliant_series(self._compliant_series.shift(n))

    def sample(
        self: Self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        seed: int | None = None,
    ) -> Self:
        """Sample randomly from this Series.

        Arguments:
            n: Number of items to return. Cannot be used with fraction.
            fraction: Fraction of items to return. Cannot be used with n.
            with_replacement: Allow values to be sampled more than once.
            seed: Seed for the random number generator. If set to None (default), a random
                seed is generated for each sample operation.

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

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.sample(fraction=1.0, with_replacement=True)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +SKIP
               a
            2  3
            1  2
            3  4
            3  4
            >>> func(s_pl)  # doctest: +SKIP
            shape: (4,)
            Series: '' [i64]
            [
               1
               4
               3
               4
            ]
        """
        return self._from_compliant_series(
            self._compliant_series.sample(
                n=n, fraction=fraction, with_replacement=with_replacement, seed=seed
            )
        )

    def alias(self, name: str) -> Self:
        """Rename the Series.

        Notes:
            This method is very cheap, but does not guarantee that data
            will be copied. For example:

            ```python
            s1: nw.Series
            s2 = s1.alias("foo")
            arr = s2.to_numpy()
            arr[0] = 999
            ```

            may (depending on the backend, and on the version) result in
            `s1`'s data being modified. We recommend:

                - if you need to alias an object and don't need the original
                  one around any more, just use `alias` without worrying about it.
                - if you were expecting `alias` to copy data, then explicily call
                  `.clone` before calling `alias`.

        Arguments:
            name: The new name.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name="foo")
            >>> s_pl = pl.Series("foo", s)
            >>> s_pa = pa.chunked_array([s])

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.alias("bar")

            We can then pass any supported library such as pandas, Polars, or PyArrow:

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
            >>> func(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at 0x...>
            [
              [
                1,
                2,
                3
              ]
            ]
        """
        return self._from_compliant_series(self._compliant_series.alias(name=name))

    def rename(self, name: str) -> Self:
        """Rename the Series.

        Alias for `Series.alias()`.

        Notes:
            This method is very cheap, but does not guarantee that data
            will be copied. For example:

            ```python
            s1: nw.Series
            s2 = s1.rename("foo")
            arr = s2.to_numpy()
            arr[0] = 999
            ```

            may (depending on the backend, and on the version) result in
            `s1`'s data being modified. We recommend:

                - if you need to rename an object and don't need the original
                  one around any more, just use `rename` without worrying about it.
                - if you were expecting `rename` to copy data, then explicily call
                  `.clone` before calling `rename`.

        Arguments:
            name: The new name.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name="foo")
            >>> s_pl = pl.Series("foo", s)
            >>> s_pa = pa.chunked_array([s])

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.rename("bar")

            We can then pass any supported library such as pandas, Polars, or PyArrow:

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
            >>> func(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at 0x...>
            [
              [
                1,
                2,
                3
              ]
            ]
        """
        return self.alias(name=name)

    def replace_strict(
        self: Self,
        old: Sequence[Any] | Mapping[Any, Any],
        new: Sequence[Any] | None = None,
        *,
        return_dtype: DType | type[DType] | None = None,
    ) -> Self:
        """Replace all values by different values.

        This function must replace all non-null input values (else it raises an error).

        Arguments:
            old: Sequence of values to replace. It also accepts a mapping of values to
                their replacement as syntactic sugar for
                `replace_all(old=list(mapping.keys()), new=list(mapping.values()))`.
            new: Sequence of values to replace by. Length must match the length of `old`.
            return_dtype: The data type of the resulting expression. If set to `None`
                (default), the data type is determined automatically based on the other
                inputs.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> df_pd = pd.DataFrame({"a": [3, 0, 1, 2]})
            >>> df_pl = pl.DataFrame({"a": [3, 0, 1, 2]})
            >>> df_pa = pa.table({"a": [3, 0, 1, 2]})

            Let's define dataframe-agnostic functions:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.replace_strict(
            ...         [0, 1, 2, 3], ["zero", "one", "two", "three"], return_dtype=nw.String
            ...     )

            We can then pass any supported library such as Pandas, Polars, or PyArrow to `func`:

            >>> func(df_pd["a"])
            0    three
            1     zero
            2      one
            3      two
            Name: a, dtype: object
            >>> func(df_pl["a"])  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: 'a' [str]
            [
                "three"
                "zero"
                "one"
                "two"
            ]
            >>> func(df_pa["a"])
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "three",
                "zero",
                "one",
                "two"
              ]
            ]
        """
        if new is None:
            if not isinstance(old, Mapping):
                msg = "`new` argument is required if `old` argument is not a Mapping type"
                raise TypeError(msg)

            new = list(old.values())
            old = list(old.keys())

        return self._from_compliant_series(
            self._compliant_series.replace_strict(old, new, return_dtype=return_dtype)
        )

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        """Sort this Series. Place null values first.

        Arguments:
            descending: Sort in descending order.
            nulls_last: Place null values last instead of first.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [5, None, 1, 2]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define library agnostic functions:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.sort()

            >>> @nw.narwhalify
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
        return self._from_compliant_series(
            self._compliant_series.sort(descending=descending, nulls_last=nulls_last)
        )

    def is_null(self) -> Self:
        """Returns a boolean Series indicating which values are null.

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

            >>> @nw.narwhalify
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
        return self._from_compliant_series(self._compliant_series.is_null())

    def fill_null(
        self,
        value: Any | None = None,
        strategy: Literal["forward", "backward"] | None = None,
        limit: int | None = None,
    ) -> Self:
        """Fill null values using the specified value.

        Arguments:
            value: Value used to fill null values.

            strategy: Strategy used to fill null values.

            limit: Number of consecutive null values to fill when using the 'forward' or 'backward' strategy.

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

            >>> @nw.narwhalify
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

            Using a strategy:

            >>> @nw.narwhalify
            ... def func_strategies(s):
            ...     return s.fill_null(strategy="forward", limit=1)

            >>> func_strategies(s_pd)
            0    1.0
            1    2.0
            2    2.0
            dtype: float64

            >>> func_strategies(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               1
               2
               2
            ]
        """
        if value is not None and strategy is not None:
            msg = "cannot specify both `value` and `strategy`"
            raise ValueError(msg)
        if value is None and strategy is None:
            msg = "must specify either a fill `value` or `strategy`"
            raise ValueError(msg)
        if strategy is not None and strategy not in {"forward", "backward"}:
            msg = f"strategy not supported: {strategy}"
            raise ValueError(msg)
        return self._from_compliant_series(
            self._compliant_series.fill_null(value=value, strategy=strategy, limit=limit)
        )

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Self:
        """Get a boolean mask of the values that are between the given lower/upper bounds.

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

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.is_between(2, 4, "right")

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
        return self._from_compliant_series(
            self._compliant_series.is_between(lower_bound, upper_bound, closed=closed)
        )

    def n_unique(self) -> int:
        """Count the number of unique values.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 2, 3]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.n_unique()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            3
            >>> func(s_pl)
            3
        """
        return self._compliant_series.n_unique()  # type: ignore[no-any-return]

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name="a")
            >>> s_pl = pl.Series("a", s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.to_numpy()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            array([1, 2, 3]...)
            >>> func(s_pl)
            array([1, 2, 3]...)
        """
        return self._compliant_series.to_numpy()

    def to_pandas(self) -> pd.Series:
        """Convert to pandas.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name="a")
            >>> s_pl = pl.Series("a", s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.to_pandas()

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
        return self._compliant_series.to_pandas()

    def __add__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__add__(self._extract_native(other))
        )

    def __radd__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__radd__(self._extract_native(other))
        )

    def __sub__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__sub__(self._extract_native(other))
        )

    def __rsub__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__rsub__(self._extract_native(other))
        )

    def __mul__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__mul__(self._extract_native(other))
        )

    def __rmul__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__rmul__(self._extract_native(other))
        )

    def __truediv__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__truediv__(self._extract_native(other))
        )

    def __rtruediv__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__rtruediv__(self._extract_native(other))
        )

    def __floordiv__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__floordiv__(self._extract_native(other))
        )

    def __rfloordiv__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__rfloordiv__(self._extract_native(other))
        )

    def __pow__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__pow__(self._extract_native(other))
        )

    def __rpow__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__rpow__(self._extract_native(other))
        )

    def __mod__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__mod__(self._extract_native(other))
        )

    def __rmod__(self, other: object) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__rmod__(self._extract_native(other))
        )

    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        return self._from_compliant_series(
            self._compliant_series.__eq__(self._extract_native(other))
        )

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        return self._from_compliant_series(
            self._compliant_series.__ne__(self._extract_native(other))
        )

    def __gt__(self, other: Any) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__gt__(self._extract_native(other))
        )

    def __ge__(self, other: Any) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__ge__(self._extract_native(other))
        )

    def __lt__(self, other: Any) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__lt__(self._extract_native(other))
        )

    def __le__(self, other: Any) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__le__(self._extract_native(other))
        )

    def __and__(self, other: Any) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__and__(self._extract_native(other))
        )

    def __or__(self, other: Any) -> Self:
        return self._from_compliant_series(
            self._compliant_series.__or__(self._extract_native(other))
        )

    # unary
    def __invert__(self) -> Self:
        return self._from_compliant_series(self._compliant_series.__invert__())

    def filter(self, other: Any) -> Self:
        """Filter elements in the Series based on a condition.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> s = [4, 10, 15, 34, 50]
            >>> s_pd = pd.Series(s)
            >>> s_pl = pl.Series(s)

            We define a library agnostic function:

            >>> @nw.narwhalify
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
        return self._from_compliant_series(
            self._compliant_series.filter(self._extract_native(other))
        )

    # --- descriptive ---
    def is_duplicated(self: Self) -> Self:
        r"""Get a mask of all duplicated rows in the Series.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, 2, 3, 1])
            >>> s_pl = pl.Series([1, 2, 3, 1])

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
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
        return self._from_compliant_series(self._compliant_series.is_duplicated())

    def is_empty(self: Self) -> bool:
        r"""Check if the series is empty.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl

            Let's define a dataframe-agnostic function that filters rows in which "foo"
            values are greater than 10, and then checks if the result is empty or not:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.filter(s > 10).is_empty()

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
        return self._compliant_series.is_empty()  # type: ignore[no-any-return]

    def is_unique(self: Self) -> Self:
        r"""Get a mask of all unique rows in the Series.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, 2, 3, 1])
            >>> s_pl = pl.Series([1, 2, 3, 1])

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
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
        return self._from_compliant_series(self._compliant_series.is_unique())

    def null_count(self: Self) -> int:
        r"""Create a new Series that shows the null counts per column.

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

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.null_count()

            We can then pass either pandas or Polars to `func`:
            >>> func(s_pd)
            np.int64(1)
            >>> func(s_pl)
            2
        """
        return self._compliant_series.null_count()  # type: ignore[no-any-return]

    def is_first_distinct(self: Self) -> Self:
        r"""Return a boolean mask indicating the first occurrence of each distinct value.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, 1, 2, 3, 2])
            >>> s_pl = pl.Series([1, 1, 2, 3, 2])

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
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
        return self._from_compliant_series(self._compliant_series.is_first_distinct())

    def is_last_distinct(self: Self) -> Self:
        r"""Return a boolean mask indicating the last occurrence of each distinct value.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, 1, 2, 3, 2])
            >>> s_pl = pl.Series([1, 1, 2, 3, 2])

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
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
        return self._from_compliant_series(self._compliant_series.is_last_distinct())

    def is_sorted(self: Self, *, descending: bool = False) -> bool:
        r"""Check if the Series is sorted.

        Arguments:
            descending: Check if the Series is sorted in descending order.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> unsorted_data = [1, 3, 2]
            >>> sorted_data = [3, 2, 1]

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s, descending=False):
            ...     return s.is_sorted(descending=descending)

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
        return self._compliant_series.is_sorted(descending=descending)  # type: ignore[no-any-return]

    def value_counts(
        self: Self,
        *,
        sort: bool = False,
        parallel: bool = False,
        name: str | None = None,
        normalize: bool = False,
    ) -> DataFrame[Any]:
        r"""Count the occurrences of unique values.

        Arguments:
            sort: Sort the output by count in descending order. If set to False (default),
                the order of the output is random.
            parallel: Execute the computation in parallel. Used for Polars only.
            name: Give the resulting count column a specific name; if `normalize` is True
                defaults to "proportion", otherwise defaults to "count".
            normalize: If true gives relative frequencies of the unique values

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, 1, 2, 3, 2], name="s")
            >>> s_pl = pl.Series(values=[1, 1, 2, 3, 2], name="s")

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
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
        return self._dataframe(
            self._compliant_series.value_counts(
                sort=sort, parallel=parallel, name=name, normalize=normalize
            ),
            level=self._level,
        )

    def quantile(
        self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Any:
        """Get quantile value of the series.

        Note:
            pandas and Polars may have implementation differences for a given interpolation method.

        Arguments:
            quantile: Quantile between 0.0 and 1.0.
            interpolation: Interpolation method.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = list(range(50))
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return [
            ...         s.quantile(quantile=q, interpolation="nearest")
            ...         for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            ...     ]

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            [np.int64(5), np.int64(12), np.int64(24), np.int64(37), np.int64(44)]

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            [5.0, 12.0, 25.0, 37.0, 44.0]
        """
        return self._compliant_series.quantile(
            quantile=quantile, interpolation=interpolation
        )

    def zip_with(self: Self, mask: Self, other: Self) -> Self:
        """Take values from self or other based on the given mask.

        Where mask evaluates true, take values from self. Where mask evaluates false,
        take values from other.

        Arguments:
            mask: Boolean Series
            other: Series of same type.

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

            >>> @nw.narwhalify
            ... def func(s1_any, mask_any, s2_any):
            ...     return s1_any.zip_with(mask_any, s2_any)

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
        return self._from_compliant_series(
            self._compliant_series.zip_with(
                self._extract_native(mask), self._extract_native(other)
            )
        )

    def item(self: Self, index: int | None = None) -> Any:
        r"""Return the Series as a scalar, or return the element at the given index.

        If no index is provided, this is equivalent to `s[0]`, with a check
        that the shape is (1,). With an index, this is equivalent to `s[index]`.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl

            Let's define a dataframe-agnostic function that returns item at given index

            >>> @nw.narwhalify
            ... def func(s, index=None):
            ...     return s.item(index)

            We can then pass either pandas or Polars to `func`:

            >>> func(pl.Series("a", [1]), None), func(pd.Series([1]), None)
            (1, np.int64(1))

            >>> func(pl.Series("a", [9, 8, 7]), -1), func(pl.Series([9, 8, 7]), -2)
            (7, 8)
        """
        return self._compliant_series.item(index=index)

    def head(self: Self, n: int = 10) -> Self:
        r"""Get the first `n` rows.

        Arguments:
            n: Number of rows to return.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = list(range(10))
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            Let's define a dataframe-agnostic function that returns the first 3 rows:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.head(3)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            0    0
            1    1
            2    2
            dtype: int64

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               0
               1
               2
            ]
        """
        return self._from_compliant_series(self._compliant_series.head(n))

    def tail(self: Self, n: int = 10) -> Self:
        r"""Get the last `n` rows.

        Arguments:
            n: Number of rows to return.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = list(range(10))
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            Let's define a dataframe-agnostic function that returns the last 3 rows:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.tail(3)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            7    7
            8    8
            9    9
            dtype: int64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               7
               8
               9
            ]
        """
        return self._from_compliant_series(self._compliant_series.tail(n))

    def round(self: Self, decimals: int = 0) -> Self:
        r"""Round underlying floating point data by `decimals` digits.

        Arguments:
            decimals: Number of decimals to round by.

        Notes:
            For values exactly halfway between rounded decimal values pandas behaves differently than Polars and Arrow.

            pandas rounds to the nearest even value (e.g. -0.5 and 0.5 round to 0.0, 1.5 and 2.5 round to 2.0, 3.5 and
            4.5 to 4.0, etc..).

            Polars and Arrow round away from 0 (e.g. -0.5 to -1.0, 0.5 to 1.0, 1.5 to 2.0, 2.5 to 3.0, etc..).

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = [1.12345, 2.56789, 3.901234]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            Let's define a dataframe-agnostic function that rounds to the first decimal:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.round(1)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            0    1.1
            1    2.6
            2    3.9
            dtype: float64

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [f64]
            [
               1.1
               2.6
               3.9
            ]
        """
        return self._from_compliant_series(self._compliant_series.round(decimals))

    def to_dummies(
        self: Self, *, separator: str = "_", drop_first: bool = False
    ) -> DataFrame[Any]:
        r"""Get dummy/indicator variables.

        Arguments:
            separator: Separator/delimiter used when generating column names.
            drop_first: Remove the first category from the variable being encoded.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data, name="a")
            >>> s_pl = pl.Series("a", data)

            Let's define a dataframe-agnostic function that rounds to the first decimal:

            >>> @nw.narwhalify
            ... def func(s, drop_first: bool = False):
            ...     return s.to_dummies(drop_first=drop_first)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
               a_1  a_2  a_3
            0    1    0    0
            1    0    1    0
            2    0    0    1

            >>> func(s_pd, drop_first=True)
               a_2  a_3
            0    0    0
            1    1    0
            2    0    1

            >>> func(s_pl)
            shape: (3, 3)
            ┌─────┬─────┬─────┐
            │ a_1 ┆ a_2 ┆ a_3 │
            │ --- ┆ --- ┆ --- │
            │ i8  ┆ i8  ┆ i8  │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 0   ┆ 0   │
            │ 0   ┆ 1   ┆ 0   │
            │ 0   ┆ 0   ┆ 1   │
            └─────┴─────┴─────┘
            >>> func(s_pl, drop_first=True)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a_2 ┆ a_3 │
            │ --- ┆ --- │
            │ i8  ┆ i8  │
            ╞═════╪═════╡
            │ 0   ┆ 0   │
            │ 1   ┆ 0   │
            │ 0   ┆ 1   │
            └─────┴─────┘
        """
        return self._dataframe(
            self._compliant_series.to_dummies(separator=separator, drop_first=drop_first),
            level=self._level,
        )

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        r"""Take every nth value in the Series and return as new Series.

        Arguments:
            n: Gather every *n*-th row.
            offset: Starting index.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = [1, 2, 3, 4]
            >>> s_pd = pd.Series(name="a", data=data)
            >>> s_pl = pl.Series(name="a", values=data)

            Let's define a dataframe-agnostic function in which gather every 2 rows,
            starting from a offset of 1:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.gather_every(n=2, offset=1)

            >>> func(s_pd)
            1    2
            3    4
            Name: a, dtype: int64

            >>> func(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (2,)
            Series: 'a' [i64]
            [
               2
               4
            ]
        """
        return self._from_compliant_series(
            self._compliant_series.gather_every(n=n, offset=offset)
        )

    def to_arrow(self: Self) -> pa.Array:
        r"""Convert to arrow.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = [1, 2, 3, 4]
            >>> s_pd = pd.Series(name="a", data=data)
            >>> s_pl = pl.Series(name="a", values=data)

            Let's define a dataframe-agnostic function that converts to arrow:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.to_arrow()

            >>> func(s_pd)  # doctest:+NORMALIZE_WHITESPACE
            <pyarrow.lib.Int64Array object at ...>
            [
                1,
                2,
                3,
                4
            ]

            >>> func(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            <pyarrow.lib.Int64Array object at ...>
            [
                1,
                2,
                3,
                4
            ]
        """
        return self._compliant_series.to_arrow()

    def mode(self: Self) -> Self:
        r"""Compute the most occurring value(s).

        Can return multiple values.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw

            >>> data = [1, 1, 2, 2, 3]
            >>> s_pd = pd.Series(name="a", data=data)
            >>> s_pl = pl.Series(name="a", values=data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.mode().sort()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    1
            1    2
            Name: a, dtype: int64

            >>> func(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (2,)
            Series: 'a' [i64]
            [
               1
               2
            ]
        """
        return self._from_compliant_series(self._compliant_series.mode())

    def is_finite(self: Self) -> Self:
        """Returns a boolean Series indicating which values are finite.

        Warning:
            Different backend handle null values differently. `is_finite` will return
            False for NaN and Null's in the Dask and pandas non-nullable backend, while
            for Polars, PyArrow and pandas nullable backends null values are kept as such.

        Returns:
            Expression of `Boolean` data type.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> data = [float("nan"), float("inf"), 2.0, None]

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.is_finite()

            We can then pass any supported library such as Pandas, Polars, or PyArrow to `func`:

            >>> func(pd.Series(data))
            0    False
            1    False
            2     True
            3    False
            dtype: bool

            >>> func(pl.Series(data))  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
               false
               false
               true
               null
            ]

            >>> func(pa.chunked_array([data]))  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                false,
                false,
                true,
                null
              ]
            ]
        """
        return self._from_compliant_series(self._compliant_series.is_finite())

    def cum_count(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative count of the non-null values in the series.

        Arguments:
            reverse: reverse the operation

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> data = ["x", "k", None, "d"]

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.cum_count(reverse=True)

            We can then pass any supported library such as Pandas, Polars, or PyArrow to `func`:

            >>> func(pd.Series(data))
            0    3
            1    2
            2    1
            3    1
            dtype: int64
            >>> func(pl.Series(data))  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [u32]
            [
                3
                2
                1
                1
            ]
            >>> func(pa.chunked_array([data]))  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                3,
                2,
                1,
                1
              ]
            ]

        """
        return self._from_compliant_series(
            self._compliant_series.cum_count(reverse=reverse)
        )

    def cum_min(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative min of the non-null values in the series.

        Arguments:
            reverse: reverse the operation

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> data = [3, 1, None, 2]

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.cum_min()

            We can then pass any supported library such as Pandas, Polars, or PyArrow to `func`:

            >>> func(pd.Series(data))
            0    3.0
            1    1.0
            2    NaN
            3    1.0
            dtype: float64
            >>> func(pl.Series(data))  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
               3
               1
               null
               1
            ]
            >>> func(pa.chunked_array([data]))  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                3,
                1,
                null,
                1
              ]
            ]

        """
        return self._from_compliant_series(
            self._compliant_series.cum_min(reverse=reverse)
        )

    def cum_max(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative max of the non-null values in the series.

        Arguments:
            reverse: reverse the operation

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> data = [1, 3, None, 2]

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.cum_max()

            We can then pass any supported library such as Pandas, Polars, or PyArrow to `func`:

            >>> func(pd.Series(data))
            0    1.0
            1    3.0
            2    NaN
            3    3.0
            dtype: float64
            >>> func(pl.Series(data))  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
               1
               3
               null
               3
            ]
            >>> func(pa.chunked_array([data]))  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                3,
                null,
                3
              ]
            ]

        """
        return self._from_compliant_series(
            self._compliant_series.cum_max(reverse=reverse)
        )

    def cum_prod(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative product of the non-null values in the series.

        Arguments:
            reverse: reverse the operation

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> data = [1, 3, None, 2]

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.cum_prod()

            We can then pass any supported library such as Pandas, Polars, or PyArrow to `func`:

            >>> func(pd.Series(data))
            0    1.0
            1    3.0
            2    NaN
            3    6.0
            dtype: float64
            >>> func(pl.Series(data))  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
               1
               3
               null
               6
            ]
            >>> func(pa.chunked_array([data]))  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                3,
                null,
                6
              ]
            ]

        """
        return self._from_compliant_series(
            self._compliant_series.cum_prod(reverse=reverse)
        )

    def __iter__(self: Self) -> Iterator[Any]:
        yield from self._compliant_series.__iter__()

    @property
    def str(self: Self) -> SeriesStringNamespace[Self]:
        return SeriesStringNamespace(self)

    @property
    def dt(self: Self) -> SeriesDateTimeNamespace[Self]:
        return SeriesDateTimeNamespace(self)

    @property
    def cat(self: Self) -> SeriesCatNamespace[Self]:
        return SeriesCatNamespace(self)


SeriesT = TypeVar("SeriesT", bound=Series)


class SeriesCatNamespace(Generic[SeriesT]):
    def __init__(self: Self, series: SeriesT) -> None:
        self._narwhals_series = series

    def get_categories(self: Self) -> SeriesT:
        """Get unique categories from column.

        Examples:
            Let's create some series:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = ["apple", "mango", "mango"]
            >>> s_pd = pd.Series(data, dtype="category")
            >>> s_pl = pl.Series(data, dtype=pl.Categorical)

            We define a dataframe-agnostic function to get unique categories
            from column 'fruits':

            >>> @nw.narwhalify(series_only=True)
            ... def func(s):
            ...     return s.cat.get_categories()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    apple
            1    mango
            dtype: object
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [str]
            [
               "apple"
               "mango"
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.cat.get_categories()
        )


class SeriesStringNamespace(Generic[SeriesT]):
    def __init__(self: Self, series: SeriesT) -> None:
        self._narwhals_series = series

    def len_chars(self: Self) -> SeriesT:
        r"""Return the length of each string as the number of characters.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = ["foo", "Café", "345", "東京", None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.str.len_chars()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    3.0
            1    4.0
            2    3.0
            3    2.0
            4    NaN
            dtype: float64

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [u32]
            [
               3
               4
               3
               2
               null
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.len_chars()
        )

    def replace(
        self: Self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> SeriesT:
        r"""Replace first matching regex/literal substring with a new string value.

        Arguments:
            pattern: A valid regular expression pattern.
            value: String that will replace the matched substring.
            literal: Treat `pattern` as a literal string.
            n: Number of matches to replace.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = ["123abc", "abc abc123"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     s = s.str.replace("abc", "")
            ...     return s.to_list()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            ['123', ' abc123']

            >>> func(s_pl)
            ['123', ' abc123']
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.replace(
                pattern, value, literal=literal, n=n
            )
        )

    def replace_all(
        self: Self, pattern: str, value: str, *, literal: bool = False
    ) -> SeriesT:
        r"""Replace all matching regex/literal substring with a new string value.

        Arguments:
            pattern: A valid regular expression pattern.
            value: String that will replace the matched substring.
            literal: Treat `pattern` as a literal string.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = ["123abc", "abc abc123"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     s = s.str.replace_all("abc", "")
            ...     return s.to_list()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            ['123', ' 123']

            >>> func(s_pl)
            ['123', ' 123']
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.replace_all(
                pattern, value, literal=literal
            )
        )

    def strip_chars(self: Self, characters: str | None = None) -> SeriesT:
        r"""Remove leading and trailing characters.

        Arguments:
            characters: The set of characters to be removed. All combinations of this set of characters will be stripped from the start and end of the string. If set to None (default), all leading and trailing whitespace is removed instead.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = ["apple", "\nmango"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     s = s.str.strip_chars()
            ...     return s.to_list()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            ['apple', 'mango']

            >>> func(s_pl)
            ['apple', 'mango']
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.strip_chars(characters)
        )

    def starts_with(self: Self, prefix: str) -> SeriesT:
        r"""Check if string values start with a substring.

        Arguments:
            prefix: prefix substring

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = ["apple", "mango", None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.str.starts_with("app")

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0     True
            1    False
            2     None
            dtype: object

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [bool]
            [
               true
               false
               null
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.starts_with(prefix)
        )

    def ends_with(self: Self, suffix: str) -> SeriesT:
        r"""Check if string values end with a substring.

        Arguments:
            suffix: suffix substring

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = ["apple", "mango", None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.str.ends_with("ngo")

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    False
            1     True
            2     None
            dtype: object

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [bool]
            [
               false
               true
               null
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.ends_with(suffix)
        )

    def contains(self: Self, pattern: str, *, literal: bool = False) -> SeriesT:
        r"""Check if string contains a substring that matches a pattern.

        Arguments:
            pattern: A Character sequence or valid regular expression pattern.
            literal: If True, treats the pattern as a literal string.
                     If False, assumes the pattern is a regular expression.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> pets = ["cat", "dog", "rabbit and parrot", "dove", None]
            >>> s_pd = pd.Series(pets)
            >>> s_pl = pl.Series(pets)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.str.contains("parrot|dove")

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    False
            1    False
            2     True
            3     True
            4     None
            dtype: object

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [bool]
            [
               false
               false
               true
               true
               null
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.contains(pattern, literal=literal)
        )

    def slice(self: Self, offset: int, length: int | None = None) -> SeriesT:
        r"""Create subslices of the string values of a Series.

        Arguments:
            offset: Start index. Negative indexing is supported.
            length: Length of the slice. If set to `None` (default), the slice is taken to the
                end of the string.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = ["pear", None, "papaya", "dragonfruit"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.str.slice(4, length=3)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            0
            1    None
            2      ya
            3     onf
            dtype: object

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [str]
            [
               ""
               null
               "ya"
               "onf"
            ]

            Using negative indexes:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.str.slice(-3)

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            0     ear
            1    None
            2     aya
            3     uit
            dtype: object

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [str]
            [
                "ear"
                null
                "aya"
                "uit"
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.slice(
                offset=offset, length=length
            )
        )

    def head(self: Self, n: int = 5) -> SeriesT:
        r"""Take the first n elements of each string.

        Arguments:
            n: Number of elements to take. Negative indexing is supported (see note (1.))

        Notes:
            1. When the `n` input is negative, `head` returns characters up to the n-th from the end of the string.
                For example, if `n = -3`, then all characters except the last three are returned.
            2. If the length of the string has fewer than `n` characters, the full string is returned.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> lyrics = ["Atatata", "taata", "taatatata", "zukkyun"]
            >>> s_pd = pd.Series(lyrics)
            >>> s_pl = pl.Series(lyrics)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.slice(0, n)
        )

    def tail(self: Self, n: int = 5) -> SeriesT:
        r"""Take the last n elements of each string.

        Arguments:
            n: Number of elements to take. Negative indexing is supported (see note (1.))

        Notes:
            1. When the `n` input is negative, `tail` returns characters starting from the n-th from the beginning of
                the string. For example, if `n = -3`, then all characters except the first three are returned.
            2. If the length of the string has fewer than `n` characters, the full string is returned.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> lyrics = ["Atatata", "taata", "taatatata", "zukkyun"]
            >>> s_pd = pd.Series(lyrics)
            >>> s_pl = pl.Series(lyrics)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.str.tail()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    atata
            1    taata
            2    atata
            3    kkyun
            dtype: object
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [str]
            [
               "atata"
               "taata"
               "atata"
               "kkyun"
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.slice(-n)
        )

    def to_uppercase(self) -> SeriesT:
        r"""Transform string to uppercase variant.

        Notes:
            The PyArrow backend will convert 'ß' to 'ẞ' instead of 'SS'.
            For more info see: https://github.com/apache/arrow/issues/34599
            There may be other unicode-edge-case-related variations across implementations.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {"fruits": ["apple", "mango", None]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(upper_col=nw.col("fruits").str.to_uppercase())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest: +NORMALIZE_WHITESPACE
             fruits  upper_col
            0  apple      APPLE
            1  mango      MANGO
            2   None       None

            >>> func(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3, 2)
            ┌────────┬───────────┐
            │ fruits ┆ upper_col │
            │ ---    ┆ ---       │
            │ str    ┆ str       │
            ╞════════╪═══════════╡
            │ apple  ┆ APPLE     │
            │ mango  ┆ MANGO     │
            │ null   ┆ null      │
            └────────┴───────────┘

        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.to_uppercase()
        )

    def to_lowercase(self) -> SeriesT:
        r"""Transform string to lowercase variant.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {"fruits": ["APPLE", "MANGO", None]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(lower_col=nw.col("fruits").str.to_lowercase())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest: +NORMALIZE_WHITESPACE
              fruits lower_col
            0  APPLE     apple
            1  MANGO     mango
            2   None      None


            >>> func(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3, 2)
            ┌────────┬───────────┐
            │ fruits ┆ lower_col │
            │ ---    ┆ ---       │
            │ str    ┆ str       │
            ╞════════╪═══════════╡
            │ APPLE  ┆ apple     │
            │ MANGO  ┆ mango     │
            │ null   ┆ null      │
            └────────┴───────────┘
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.to_lowercase()
        )

    def to_datetime(self: Self, format: str | None = None) -> SeriesT:  # noqa: A002
        """Parse Series with strings to a Series with Datetime dtype.

        Notes:
            pandas defaults to nanosecond time unit, Polars to microsecond.
            Prior to pandas 2.0, nanoseconds were the only time unit supported
            in pandas, with no ability to set any other one. The ability to
            set the time unit in pandas, if the version permits, will arrive.

        Warning:
            As different backends auto-infer format in different ways, if `format=None`
            there is no guarantee that the result will be equal.

        Arguments:
            format: Format to use for conversion. If set to None (default), the format is
                inferred from the data.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> data = ["2020-01-01", "2020-01-02"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.str.to_datetime(format="%Y-%m-%d")

            We can then pass any supported library such as pandas, Polars, or PyArrow::

            >>> func(s_pd)
            0   2020-01-01
            1   2020-01-02
            dtype: datetime64[ns]
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [datetime[μs]]
            [
               2020-01-01 00:00:00
               2020-01-02 00:00:00
            ]
            >>> func(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at 0x...>
            [
              [
                2020-01-01 00:00:00.000000,
                2020-01-02 00:00:00.000000
              ]
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.str.to_datetime(format=format)
        )


class SeriesDateTimeNamespace(Generic[SeriesT]):
    def __init__(self: Self, series: SeriesT) -> None:
        self._narwhals_series = series

    def date(self: Self) -> SeriesT:
        """Get the date in a datetime series.

        Raises:
            NotImplementedError: If pandas default backend is being used.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2012, 1, 7, 10, 20), datetime(2023, 3, 10, 11, 32)]
            >>> s_pd = pd.Series(dates).convert_dtypes(dtype_backend="pyarrow")
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.date()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    2012-01-07
            1    2023-03-10
            dtype: date32[day][pyarrow]

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [date]
            [
               2012-01-07
               2023-03-10
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.date()
        )

    def year(self: Self) -> SeriesT:
        """Get the year in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2012, 1, 7), datetime(2023, 3, 10)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.year()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.year()
        )

    def month(self: Self) -> SeriesT:
        """Gets the month in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2023, 2, 1), datetime(2023, 8, 3)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.month()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.month()
        )

    def day(self: Self) -> SeriesT:
        """Extracts the day in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2022, 1, 1), datetime(2022, 1, 5)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.day()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.day()
        )

    def hour(self: Self) -> SeriesT:
        """Extracts the hour in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2022, 1, 1, 5, 3), datetime(2022, 1, 5, 9, 12)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.hour()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.hour()
        )

    def minute(self: Self) -> SeriesT:
        """Extracts the minute in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2022, 1, 1, 5, 3), datetime(2022, 1, 5, 9, 12)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.minute()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.minute()
        )

    def second(self: Self) -> SeriesT:
        """Extracts the seconds in a datetime series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [datetime(2022, 1, 1, 5, 3, 10), datetime(2022, 1, 5, 9, 12, 4)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.second()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.second()
        )

    def millisecond(self: Self) -> SeriesT:
        """Extracts the milliseconds in a datetime series.

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
            ...     datetime(2023, 5, 21, 12, 55, 11, 200000),
            ... ]

            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.millisecond().alias("datetime")

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.millisecond()
        )

    def microsecond(self: Self) -> SeriesT:
        """Extracts the microseconds in a datetime series.

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
            ...     datetime(2023, 5, 21, 12, 55, 11, 200000),
            ... ]

            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.microsecond().alias("datetime")

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.microsecond()
        )

    def nanosecond(self: Self) -> SeriesT:
        """Extract the nanoseconds in a date series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> dates = [
            ...     datetime(2022, 1, 1, 5, 3, 10, 500000),
            ...     datetime(2022, 1, 5, 9, 12, 4, 60000),
            ... ]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.nanosecond()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.nanosecond()
        )

    def ordinal_day(self: Self) -> SeriesT:
        """Get ordinal day.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = [datetime(2020, 1, 1), datetime(2020, 8, 3)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.ordinal_day()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.ordinal_day()
        )

    def total_minutes(self: Self) -> SeriesT:
        """Get total minutes.

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

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.total_minutes()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.total_minutes()
        )

    def total_seconds(self: Self) -> SeriesT:
        """Get total seconds.

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

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.total_seconds()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.total_seconds()
        )

    def total_milliseconds(self: Self) -> SeriesT:
        """Get total milliseconds.

        Notes:
            The function outputs the total milliseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = [
            ...     timedelta(milliseconds=10),
            ...     timedelta(milliseconds=20, microseconds=40),
            ... ]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.total_milliseconds()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.total_milliseconds()
        )

    def total_microseconds(self: Self) -> SeriesT:
        """Get total microseconds.

        Notes:
            The function outputs the total microseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = [
            ...     timedelta(microseconds=10),
            ...     timedelta(milliseconds=1, microseconds=200),
            ... ]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.total_microseconds()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.total_microseconds()
        )

    def total_nanoseconds(self: Self) -> SeriesT:
        """Get total nanoseconds.

        Notes:
            The function outputs the total nanoseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = ["2024-01-01 00:00:00.000000001", "2024-01-01 00:00:00.000000002"]
            >>> s_pd = pd.to_datetime(pd.Series(data))
            >>> s_pl = pl.Series(data).str.to_datetime(time_unit="ns")

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.diff().dt.total_nanoseconds()

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
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.total_nanoseconds()
        )

    def to_string(self: Self, format: str) -> SeriesT:  # noqa: A002
        """Convert a Date/Time/Datetime series into a String series with the given format.

        Notes:
            Unfortunately, different libraries interpret format directives a bit
            differently.

            - Chrono, the library used by Polars, uses `"%.f"` for fractional seconds,
              whereas pandas and Python stdlib use `".%f"`.
            - PyArrow interprets `"%S"` as "seconds, including fractional seconds"
              whereas most other tools interpret it as "just seconds, as 2 digits".

            Therefore, we make the following adjustments:

            - for pandas-like libraries, we replace `"%S.%f"` with `"%S%.f"`.
            - for PyArrow, we replace `"%S.%f"` with `"%S"`.

            Workarounds like these don't make us happy, and we try to avoid them as
            much as possible, but here we feel like it's the best compromise.

            If you just want to format a date/datetime Series as a local datetime
            string, and have it work as consistently as possible across libraries,
            we suggest using:

            - `"%Y-%m-%dT%H:%M:%S%.f"` for datetimes
            - `"%Y-%m-%d"` for dates

            though note that, even then, different tools may return a different number
            of trailing zeros. Nonetheless, this is probably consistent enough for
            most applications.

            If you have an application where this is not enough, please open an issue
            and let us know.

        Examples:
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = [
            ...     datetime(2020, 3, 1),
            ...     datetime(2020, 4, 1),
            ...     datetime(2020, 5, 1),
            ... ]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.to_string("%Y/%m/%d")

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
            0    2020/03/01
            1    2020/04/01
            2    2020/05/01
            dtype: object

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [str]
            [
               "2020/03/01"
               "2020/04/01"
               "2020/05/01"
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.to_string(format)
        )

    def replace_time_zone(self: Self, time_zone: str | None) -> SeriesT:
        """Replace time zone.

        Arguments:
            time_zone: Target time zone.

        Examples:
            >>> from datetime import datetime, timezone
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> data = [
            ...     datetime(2024, 1, 1, tzinfo=timezone.utc),
            ...     datetime(2024, 1, 2, tzinfo=timezone.utc),
            ... ]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.replace_time_zone("Asia/Kathmandu")

            We can then pass pandas / PyArrow / Polars / any other supported library:

            >>> func(s_pd)
            0   2024-01-01 00:00:00+05:45
            1   2024-01-02 00:00:00+05:45
            dtype: datetime64[ns, Asia/Kathmandu]
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [datetime[μs, Asia/Kathmandu]]
            [
                2024-01-01 00:00:00 +0545
                2024-01-02 00:00:00 +0545
            ]
            >>> func(s_pa)
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2023-12-31 18:15:00.000000Z,
                2024-01-01 18:15:00.000000Z
              ]
            ]
        """
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.replace_time_zone(time_zone)
        )

    def convert_time_zone(self: Self, time_zone: str) -> SeriesT:
        """Convert time zone.

        If converting from a time-zone-naive column, then conversion happens
        as if converting from UTC.

        Arguments:
            time_zone: Target time zone.

        Examples:
            >>> from datetime import datetime, timezone
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> data = [
            ...     datetime(2024, 1, 1, tzinfo=timezone.utc),
            ...     datetime(2024, 1, 2, tzinfo=timezone.utc),
            ... ]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.convert_time_zone("Asia/Kathmandu")

            We can then pass pandas / PyArrow / Polars / any other supported library:

            >>> func(s_pd)
            0   2024-01-01 05:45:00+05:45
            1   2024-01-02 05:45:00+05:45
            dtype: datetime64[ns, Asia/Kathmandu]
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [datetime[μs, Asia/Kathmandu]]
            [
                2024-01-01 05:45:00 +0545
                2024-01-02 05:45:00 +0545
            ]
            >>> func(s_pa)
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2024-01-01 00:00:00.000000Z,
                2024-01-02 00:00:00.000000Z
              ]
            ]
        """
        if time_zone is None:
            msg = "Target `time_zone` cannot be `None` in `convert_time_zone`. Please use `replace_time_zone(None)` if you want to remove the time zone."
            raise TypeError(msg)
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.convert_time_zone(time_zone)
        )

    def timestamp(self: Self, time_unit: Literal["ns", "us", "ms"] = "us") -> SeriesT:
        """Return a timestamp in the given time unit.

        Arguments:
            time_unit: {'ns', 'us', 'ms'}
                Time unit.

        Examples:
            >>> from datetime import date
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> data = [date(2001, 1, 1), None, date(2001, 1, 3)]
            >>> s_pd = pd.Series(data, dtype="datetime64[ns]")
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.dt.timestamp("ms")

            We can then pass pandas / PyArrow / Polars / any other supported library:

            >>> func(s_pd)
            0    9.783072e+11
            1             NaN
            2    9.784800e+11
            dtype: float64
            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
                    978307200000
                    null
                    978480000000
            ]
            >>> func(s_pa)
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                978307200000,
                null,
                978480000000
              ]
            ]
        """
        if time_unit not in {"ns", "us", "ms"}:
            msg = (
                "invalid `time_unit`"
                f"\n\nExpected one of {{'ns', 'us', 'ms'}}, got {time_unit!r}."
            )
            raise ValueError(msg)
        return self._narwhals_series._from_compliant_series(
            self._narwhals_series._compliant_series.dt.timestamp(time_unit)
        )
