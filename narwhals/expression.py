from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal

from narwhals.dependencies import get_polars
from narwhals.dtypes import translate_dtype
from narwhals.utils import flatten
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from narwhals.typing import IntoExpr


def extract_native(expr: Expr, other: Any) -> Any:
    from narwhals.series import Series

    if isinstance(other, Expr):
        return other._call(expr)
    if isinstance(other, Series):
        return other._series
    return other


class Expr:
    def __init__(self, call: Callable[[Any], Any]) -> None:
        # callable from namespace to expr
        self._call = call

    # --- convert ---
    def alias(self, name: str) -> Expr:
        """
        Rename the expression.

        Arguments:
            name: The new name.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [1, 2], 'b': [4, 5]})
            >>> df_pl = pl.DataFrame({'a': [1, 2], 'b': [4, 5]})

            Let's define a dataframe-agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.select((nw.col('b')+10).alias('c'))
            ...     return nw.to_native(df)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                c
            0  14
            1  15
            >>> func(df_pl)
            shape: (2, 1)
            ┌─────┐
            │ c   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 14  │
            │ 15  │
            └─────┘
        """
        return self.__class__(lambda plx: self._call(plx).alias(name))

    def cast(
        self,
        dtype: Any,
    ) -> Expr:
        """
        Redefine an object's data type.

        Arguments:
            dtype: Data type that the object will be cast into.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> from datetime import date
            >>> df_pd = pd.DataFrame({"foo": [1, 2, 3],"bar": [6.0, 7.0, 8.0]})
            >>> df_pl = pl.DataFrame({"foo": [1, 2, 3],"bar": [6.0, 7.0, 8.0]})

            Let's define a dataframe-agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.select(nw.col('foo').cast(nw.Float32), nw.col('bar').cast(nw.UInt8))
            ...     native_df = nw.to_native(df)
            ...     return native_df

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo  bar
            0  1.0    6
            1  2.0    7
            2  3.0    8
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ f32 ┆ u8  │
            ╞═════╪═════╡
            │ 1.0 ┆ 6   │
            │ 2.0 ┆ 7   │
            │ 3.0 ┆ 8   │
            └─────┴─────┘
        """

        return self.__class__(
            lambda plx: self._call(plx).cast(translate_dtype(plx, dtype)),
        )

    # --- binary ---
    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        return self.__class__(
            lambda plx: self._call(plx).__eq__(extract_native(plx, other))
        )

    def __ne__(self, other: object) -> Expr:  # type: ignore[override]
        return self.__class__(
            lambda plx: self._call(plx).__ne__(extract_native(plx, other))
        )

    def __and__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__and__(extract_native(plx, other))
        )

    def __rand__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__rand__(extract_native(plx, other))
        )

    def __or__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__or__(extract_native(plx, other))
        )

    def __ror__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__ror__(extract_native(plx, other))
        )

    def __add__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__add__(extract_native(plx, other))
        )

    def __radd__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__radd__(extract_native(plx, other))
        )

    def __sub__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__sub__(extract_native(plx, other))
        )

    def __rsub__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__rsub__(extract_native(plx, other))
        )

    def __truediv__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__truediv__(extract_native(plx, other))
        )

    def __rtruediv__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__rtruediv__(extract_native(plx, other))
        )

    def __mul__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__mul__(extract_native(plx, other))
        )

    def __rmul__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__rmul__(extract_native(plx, other))
        )

    def __le__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__le__(extract_native(plx, other))
        )

    def __lt__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__lt__(extract_native(plx, other))
        )

    def __gt__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__gt__(extract_native(plx, other))
        )

    def __ge__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__ge__(extract_native(plx, other))
        )

    def __pow__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__pow__(extract_native(plx, other))
        )

    def __rpow__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__rpow__(extract_native(plx, other))
        )

    def __floordiv__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__floordiv__(extract_native(plx, other))
        )

    def __rfloordiv__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__rfloordiv__(extract_native(plx, other))
        )

    def __mod__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__mod__(extract_native(plx, other))
        )

    def __rmod__(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).__rmod__(extract_native(plx, other))
        )

    # --- unary ---
    def __invert__(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).__invert__())

    def any(self) -> Expr:
        """
        Return whether any of the values in the column are `True`

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [True, False], 'b': [True, True]})
            >>> df_pl = pl.DataFrame({'a': [True, False], 'b': [True, True]})

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.col('a', 'b').any())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                  a     b
            0  True  True
            >>> func(df_pl)
            shape: (1, 2)
            ┌──────┬──────┐
            │ a    ┆ b    │
            │ ---  ┆ ---  │
            │ bool ┆ bool │
            ╞══════╪══════╡
            │ true ┆ true │
            └──────┴──────┘
        """
        return self.__class__(lambda plx: self._call(plx).any())

    def all(self) -> Expr:
        """
        Return whether all values in the column are `True`.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [True, False], 'b': [True, True]})
            >>> df_pl = pl.DataFrame({'a': [True, False], 'b': [True, True]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.col('a', 'b').all())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                   a     b
            0  False  True
            >>> func(df_pl)
            shape: (1, 2)
            ┌───────┬──────┐
            │ a     ┆ b    │
            │ ---   ┆ ---  │
            │ bool  ┆ bool │
            ╞═══════╪══════╡
            │ false ┆ true │
            └───────┴──────┘
        """
        return self.__class__(lambda plx: self._call(plx).all())

    def mean(self) -> Expr:
        """
        Get mean value.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [-1, 0, 1], 'b': [2, 4, 6]})
            >>> df_pl = pl.DataFrame({'a': [-1, 0, 1], 'b': [2, 4, 6]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...    return df.select(nw.col('a', 'b').mean())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                 a    b
            0  0.0  4.0
            >>> func(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ f64 ┆ f64 │
            ╞═════╪═════╡
            │ 0.0 ┆ 4.0 │
            └─────┴─────┘
        """
        return self.__class__(lambda plx: self._call(plx).mean())

    def std(self, *, ddof: int = 1) -> Expr:
        """
        Get standard deviation.

        Arguments:
            ddof: “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
                     where N represents the number of elements. By default ddof is 1.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [20, 25, 60], 'b': [1.5, 1, -1.4]})
            >>> df_pl = pl.DataFrame({'a': [20, 25, 60], 'b': [1.5, 1, -1.4]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...    return df.select(nw.col('a', 'b').std(ddof=0))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                      a         b
            0  17.79513  1.265789
            >>> func(df_pl)
            shape: (1, 2)
            ┌──────────┬──────────┐
            │ a        ┆ b        │
            │ ---      ┆ ---      │
            │ f64      ┆ f64      │
            ╞══════════╪══════════╡
            │ 17.79513 ┆ 1.265789 │
            └──────────┴──────────┘

        """
        return self.__class__(lambda plx: self._call(plx).std(ddof=ddof))

    def sum(self) -> Expr:
        """
        Return the sum value.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [5, 10], 'b': [50, 100]})
            >>> df_pl = pl.DataFrame({'a': [5, 10], 'b': [50, 100]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.col('a', 'b').sum())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                a    b
            0  15  150
            >>> func(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 15  ┆ 150 │
            └─────┴─────┘
        """
        return self.__class__(lambda plx: self._call(plx).sum())

    def min(self) -> Expr:
        """
        Returns the minimum value(s) from a column(s).

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [1, 2], 'b': [4, 3]})
            >>> df_pl = pl.DataFrame({'a': [1, 2], 'b': [4, 3]})

            Let's define a dataframe-agnostic function:
            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.min('a','b'))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a  b
            0  1  3
            >>> func(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 3   │
            └─────┴─────┘
        """

        return self.__class__(lambda plx: self._call(plx).min())

    def max(self) -> Expr:
        """
        Returns the maximum value(s) from a column(s).

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [10, 20], 'b': [50, 100]})
            >>> df_pl = pl.DataFrame({'a': [10, 20], 'b': [50, 100]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...    return df.select(nw.max('a', 'b'))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                a    b
            0  20  100
            >>> func(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 20  ┆ 100 │
            └─────┴─────┘
        """
        return self.__class__(lambda plx: self._call(plx).max())

    def n_unique(self) -> Expr:
        """
         Returns count of unique values

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 1, 3, 3, 5]})
            >>> df_pl = pl.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 1, 3, 3, 5]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...    return df.select(nw.col('a', 'b').n_unique())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a  b
            0  5  3
            >>> func(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ u32 ┆ u32 │
            ╞═════╪═════╡
            │ 5   ┆ 3   │
            └─────┴─────┘
        """
        return self.__class__(lambda plx: self._call(plx).n_unique())

    def unique(self) -> Expr:
        """
        Return unique values

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [1, 1, 3, 5, 5], 'b': [2, 4, 4, 6, 6]})
            >>> df_pl = pl.DataFrame({'a': [1, 1, 3, 5, 5], 'b': [2, 4, 4, 6, 6]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...    return df.select(nw.col('a', 'b').unique())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a  b
            0  1  2
            1  3  4
            2  5  6
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 2   │
            │ 3   ┆ 4   │
            │ 5   ┆ 6   │
            └─────┴─────┘
        """
        return self.__class__(lambda plx: self._call(plx).unique())

    def cum_sum(self) -> Expr:
        """
        Return cumulative sum.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [1, 1, 3, 5, 5], 'b': [2, 4, 4, 6, 6]})
            >>> df_pl = pl.DataFrame({'a': [1, 1, 3, 5, 5], 'b': [2, 4, 4, 6, 6]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...    return df.select(nw.col('a', 'b').cum_sum())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                a   b
            0   1   2
            1   2   6
            2   5  10
            3  10  16
            4  15  22
            >>> func(df_pl)
            shape: (5, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 2   │
            │ 2   ┆ 6   │
            │ 5   ┆ 10  │
            │ 10  ┆ 16  │
            │ 15  ┆ 22  │
            └─────┴─────┘
        """
        return self.__class__(lambda plx: self._call(plx).cum_sum())

    def diff(self) -> Expr:
        """
        Returns the difference between each element and the previous one.

        Notes:
            pandas may change the dtype here, for example when introducing missing
            values in an integer column. To ensure, that the dtype doesn't change,
            you may want to use `fill_null` and `cast`. For example, to calculate
            the diff and fill missing values with `0` in a Int64 column, you could
            do:

            ```python
            nw.col('a').diff().fill_null(0).cast(nw.Int64)
            ```

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [1, 1, 3, 5, 5]})
            >>> df_pl = pl.DataFrame({'a': [1, 1, 3, 5, 5]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...    return df.select(a_diff=nw.col('a').diff())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a_diff
            0     NaN
            1     0.0
            2     2.0
            3     2.0
            4     0.0
            >>> func(df_pl)
            shape: (5, 1)
            ┌────────┐
            │ a_diff │
            │ ---    │
            │ i64    │
            ╞════════╡
            │ null   │
            │ 0      │
            │ 2      │
            │ 2      │
            │ 0      │
            └────────┘
        """
        return self.__class__(lambda plx: self._call(plx).diff())

    def shift(self, n: int) -> Expr:
        """
        Shift values by `n` positions.

        Notes:
            pandas may change the dtype here, for example when introducing missing
            values in an integer column. To ensure, that the dtype doesn't change,
            you may want to use `fill_null` and `cast`. For example, to shift
            and fill missing values with `0` in a Int64 column, you could
            do:

            ```python
            nw.col('a').shift(1).fill_null(0).cast(nw.Int64)
            ```

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [1, 1, 3, 5, 5]})
            >>> df_pl = pl.DataFrame({'a': [1, 1, 3, 5, 5]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...    return df.select(a_shift=nw.col('a').shift(n=1))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a_shift
            0      NaN
            1      1.0
            2      1.0
            3      3.0
            4      5.0
            >>> func(df_pl)
            shape: (5, 1)
            ┌─────────┐
            │ a_shift │
            │ ---     │
            │ i64     │
            ╞═════════╡
            │ null    │
            │ 1       │
            │ 1       │
            │ 3       │
            │ 5       │
            └─────────┘
        """
        return self.__class__(lambda plx: self._call(plx).shift(n))

    def sort(self, *, descending: bool = False) -> Expr:
        """
        Sort this column. Place null values first.

        Arguments:
            descending: Sort in descending order.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl

            >>> df_pd = pd.DataFrame({"a": [5, None, 1, 2]})
            >>> df_pl = pl.DataFrame({"a": [5, None, 1, 2]})

            Let's define dataframe-agnostic functions:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.col('a').sort())

            >>> def func_descend(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.select(nw.col('a').sort(descending=True))
            ...     return nw.to_native(df)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                 a
            1  NaN
            2  1.0
            3  2.0
            0  5.0
            >>> func(df_pl)
            shape: (4, 1)
            ┌──────┐
            │ a    │
            │ ---  │
            │ i64  │
            ╞══════╡
            │ null │
            │ 1    │
            │ 2    │
            │ 5    │
            └──────┘

            >>> func_descend(df_pd)
                 a
            1  NaN
            0  5.0
            3  2.0
            2  1.0
            >>> func_descend(df_pl)
            shape: (4, 1)
            ┌──────┐
            │ a    │
            │ ---  │
            │ i64  │
            ╞══════╡
            │ null │
            │ 5    │
            │ 2    │
            │ 1    │
            └──────┘
        """
        return self.__class__(lambda plx: self._call(plx).sort(descending=descending))

    # --- transform ---
    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Expr:
        """
        Check if this expression is between the given lower and upper bounds.

        Arguments:
            lower_bound: Lower bound value.

            upper_bound: Upper bound value.

            closed: Define which sides of the interval are closed (inclusive).

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
            >>> df_pl = pl.DataFrame({'a': [1, 2, 3, 4, 5]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.col('a').is_between(2,4,'right'))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                   a
            0  False
            1  False
            2   True
            3   True
            4  False
            >>> func(df_pl)
            shape: (5, 1)
            ┌───────┐
            │ a     │
            │ ---   │
            │ bool  │
            ╞═══════╡
            │ false │
            │ false │
            │ true  │
            │ true  │
            │ false │
            └───────┘
        """
        return self.__class__(
            lambda plx: self._call(plx).is_between(lower_bound, upper_bound, closed)
        )

    def is_in(self, other: Any) -> Expr:
        """
        Check if elements of this expression are present in the other iterable.

        Arguments:
            other: iterable

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [1, 2, 9, 10]})
            >>> df_pl = pl.DataFrame({'a': [1, 2, 9, 10]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...    return df.with_columns(b = nw.col('a').is_in([1, 2]))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                a      b
            0   1   True
            1   2   True
            2   9  False
            3  10  False

            >>> func(df_pl)
            shape: (4, 2)
            ┌─────┬───────┐
            │ a   ┆ b     │
            │ --- ┆ ---   │
            │ i64 ┆ bool  │
            ╞═════╪═══════╡
            │ 1   ┆ true  │
            │ 2   ┆ true  │
            │ 9   ┆ false │
            │ 10  ┆ false │
            └─────┴───────┘
        """
        if isinstance(other, Iterable) and not isinstance(other, (str, bytes)):
            return self.__class__(lambda plx: self._call(plx).is_in(other))
        else:
            raise NotImplementedError(
                "Narwhals `is_in` doesn't accept expressions as an argument, as opposed to Polars. You should provide an iterable instead."
            )

    def filter(self, *predicates: Any) -> Expr:
        """
        Filters elements based on a condition, returning a new expression.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [2, 3, 4, 5, 6, 7], 'b': [10, 11, 12, 13, 14, 15]})
            >>> df_pl = pl.DataFrame({'a': [2, 3, 4, 5, 6, 7], 'b': [10, 11, 12, 13, 14, 15]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(
            ...             nw.col("a").filter(nw.col("a") > 4),
            ...             nw.col("b").filter(nw.col("b") < 13)
            ...             )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a   b
            3  5  10
            4  6  11
            5  7  12
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 5   ┆ 10  │
            │ 6   ┆ 11  │
            │ 7   ┆ 12  │
            └─────┴─────┘
        """
        return self.__class__(
            lambda plx: self._call(plx).filter(
                *[extract_native(plx, pred) for pred in flatten(predicates)]
            )
        )

    def is_null(self) -> Expr:
        """
        Returns a boolean Series indicating which values are null.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame(
            ...         {
            ...             'a': [2, 4, None, 3, 5],
            ...             'b': [2.0, 4.0, float("nan"), 3.0, 5.0]
            ...         }
            ... )
            >>> df_pl = pl.DataFrame(
            ...         {
            ...             'a': [2, 4, None, 3, 5],
            ...             'b': [2.0, 4.0, float("nan"), 3.0, 5.0]
            ...         }
            ... )

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...         a_is_null = nw.col('a').is_null(),
            ...         b_is_null = nw.col('b').is_null()
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                 a    b  a_is_null  b_is_null
            0  2.0  2.0      False      False
            1  4.0  4.0      False      False
            2  NaN  NaN       True       True
            3  3.0  3.0      False      False
            4  5.0  5.0      False      False

            >>> func(df_pl)  # nan != null for polars
            shape: (5, 4)
            ┌──────┬─────┬───────────┬───────────┐
            │ a    ┆ b   ┆ a_is_null ┆ b_is_null │
            │ ---  ┆ --- ┆ ---       ┆ ---       │
            │ i64  ┆ f64 ┆ bool      ┆ bool      │
            ╞══════╪═════╪═══════════╪═══════════╡
            │ 2    ┆ 2.0 ┆ false     ┆ false     │
            │ 4    ┆ 4.0 ┆ false     ┆ false     │
            │ null ┆ NaN ┆ true      ┆ false     │
            │ 3    ┆ 3.0 ┆ false     ┆ false     │
            │ 5    ┆ 5.0 ┆ false     ┆ false     │
            └──────┴─────┴───────────┴───────────┘
        """
        return self.__class__(lambda plx: self._call(plx).is_null())

    def fill_null(self, value: Any) -> Expr:
        """
        Fill null values with given value.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame(
            ...         {
            ...             'a': [2, 4, None, 3, 5],
            ...             'b': [2.0, 4.0, float("nan"), 3.0, 5.0]
            ...         }
            ... )
            >>> df_pl = pl.DataFrame(
            ...         {
            ...             'a': [2, 4, None, 3, 5],
            ...             'b': [2.0, 4.0, float("nan"), 3.0, 5.0]
            ...         }
            ... )

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(nw.col('a', 'b').fill_null(0))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                 a    b
            0  2.0  2.0
            1  4.0  4.0
            2  0.0  0.0
            3  3.0  3.0
            4  5.0  5.0

            >>> func(df_pl)  # nan != null for polars
            shape: (5, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ f64 │
            ╞═════╪═════╡
            │ 2   ┆ 2.0 │
            │ 4   ┆ 4.0 │
            │ 0   ┆ NaN │
            │ 3   ┆ 3.0 │
            │ 5   ┆ 5.0 │
            └─────┴─────┘
        """
        return self.__class__(lambda plx: self._call(plx).fill_null(value))

    # --- partial reduction ---
    def drop_nulls(self) -> Expr:
        """
        Remove missing values.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl

            >>> df_pd = pd.DataFrame({"a": [2.0, 4.0, float("nan"), 3.0, None, 5.0]})
            >>> df_pl = pl.DataFrame({"a": [2.0, 4.0, float("nan"), 3.0, None, 5.0]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.col("a").drop_nulls())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                 a
            0  2.0
            1  4.0
            3  3.0
            5  5.0
            >>> func(df_pl)  # nan != null for polars
            shape: (5, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ f64 │
            ╞═════╡
            │ 2.0 │
            │ 4.0 │
            │ NaN │
            │ 3.0 │
            │ 5.0 │
            └─────┘
        """
        return self.__class__(lambda plx: self._call(plx).drop_nulls())

    def sample(
        self,
        n: int | None = None,
        fraction: float | None = None,
        *,
        with_replacement: bool = False,
    ) -> Expr:
        """
        Sample randomly from this expression.

        Arguments:
            n: Number of items to return. Cannot be used with fraction.

            fraction: Fraction of items to return. Cannot be used with n.

            with_replacement: Allow values to be sampled more than once.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl

            >>> df_pd = pd.DataFrame({"a": [1, 2, 3]})
            >>> df_pl = pl.DataFrame({"a": [1, 2, 3]})

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.col('a').sample(fraction=1.0, with_replacement=True))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest:+SKIP
               a
            2  3
            0  1
            2  3
            >>> func(df_pl)  # doctest:+SKIP
            shape: (3, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ f64 │
            ╞═════╡
            │ 2   │
            │ 3   │
            │ 3   │
            └─────┘
        """
        return self.__class__(
            lambda plx: self._call(plx).sample(
                n, fraction=fraction, with_replacement=with_replacement
            )
        )

    def over(self, *keys: str | Iterable[str]) -> Expr:
        """
        Compute expressions over the given groups.

        Arguments:
            keys: Names of columns to compute window expression over.
                  Must be names of columns, as opposed to expressions -
                  so, this is a bit less flexible than Polars' `Expr.over`.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {'a': [1, 2, 3], 'b': [1, 1, 2]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...         a_min_per_group = nw.col('a').min().over('b')
            ...     )

            We can then pass either pandas or Polars:

            >>> func(df_pd)
               a  b  a_min_per_group
            0  1  1                1
            1  2  1                1
            2  3  2                3
            >>> func(df_pl)
            shape: (3, 3)
            ┌─────┬─────┬─────────────────┐
            │ a   ┆ b   ┆ a_min_per_group │
            │ --- ┆ --- ┆ ---             │
            │ i64 ┆ i64 ┆ i64             │
            ╞═════╪═════╪═════════════════╡
            │ 1   ┆ 1   ┆ 1               │
            │ 2   ┆ 1   ┆ 1               │
            │ 3   ┆ 2   ┆ 3               │
            └─────┴─────┴─────────────────┘
        """
        return self.__class__(lambda plx: self._call(plx).over(flatten(keys)))

    def is_duplicated(self) -> Expr:
        r"""
        Return a boolean mask indicating duplicated values.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.all().is_duplicated())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest: +NORMALIZE_WHITESPACE
                   a      b
            0   True   True
            1  False   True
            2  False  False
            3   True  False
            >>> func(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4, 2)
            ┌───────┬───────┐
            │ a     ┆ b     │
            │ ---   ┆ ---   │
            │ bool  ┆ bool  │
            ╞═══════╪═══════╡
            │ true  ┆ true  │
            │ false ┆ true  │
            │ false ┆ false │
            │ true  ┆ false │
            └───────┴───────┘
        """
        return self.__class__(lambda plx: self._call(plx).is_duplicated())

    def is_unique(self) -> Expr:
        r"""
        Return a boolean mask indicating unique values.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.all().is_unique())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest: +NORMALIZE_WHITESPACE
                   a      b
            0  False  False
            1   True  False
            2   True   True
            3  False   True
            >>> func(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4, 2)
            ┌───────┬───────┐
            │ a     ┆ b     │
            │ ---   ┆ ---   │
            │ bool  ┆ bool  │
            ╞═══════╪═══════╡
            │ false ┆ false │
            │ true  ┆ false │
            │ true  ┆ true  │
            │ false ┆ true  │
            └───────┴───────┘
        """

        return self.__class__(lambda plx: self._call(plx).is_unique())

    def null_count(self) -> Expr:
        r"""
        Count null values.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {"a": [1, 2, None, 1], "b": ["a", None, "b", None]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.all().null_count())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a  b
            0  1  2
            >>> func(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ u32 ┆ u32 │
            ╞═════╪═════╡
            │ 1   ┆ 2   │
            └─────┴─────┘
        """
        return self.__class__(lambda plx: self._call(plx).null_count())

    def is_first_distinct(self) -> Expr:
        r"""
        Return a boolean mask indicating the first occurrence of each distinct value.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.all().is_first_distinct())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest: +NORMALIZE_WHITESPACE
                   a      b
            0   True   True
            1   True  False
            2   True   True
            3  False   True
            >>> func(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4, 2)
            ┌───────┬───────┐
            │ a     ┆ b     │
            │ ---   ┆ ---   │
            │ bool  ┆ bool  │
            ╞═══════╪═══════╡
            │ true  ┆ true  │
            │ true  ┆ false │
            │ true  ┆ true  │
            │ false ┆ true  │
            └───────┴───────┘
        """
        return self.__class__(lambda plx: self._call(plx).is_first_distinct())

    def is_last_distinct(self) -> Expr:
        r"""Return a boolean mask indicating the last occurrence of each distinct value.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.all().is_last_distinct())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest: +NORMALIZE_WHITESPACE
                   a      b
            0  False  False
            1   True   True
            2   True   True
            3   True   True
            >>> func(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4, 2)
            ┌───────┬───────┐
            │ a     ┆ b     │
            │ ---   ┆ ---   │
            │ bool  ┆ bool  │
            ╞═══════╪═══════╡
            │ false ┆ false │
            │ true  ┆ true  │
            │ true  ┆ true  │
            │ true  ┆ true  │
            └───────┴───────┘
        """
        return self.__class__(lambda plx: self._call(plx).is_last_distinct())

    def quantile(
        self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Expr:
        r"""Get quantile value.

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
            >>> data = {'a': list(range(50)), 'b': list(range(50, 100))}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.col('a', 'b').quantile(0.5, interpolation='linear'))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest: +NORMALIZE_WHITESPACE
                a   b
            0  24.5  74.5

            >>> func(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (1, 2)
            ┌──────┬──────┐
            │ a    ┆ b    │
            │ ---  ┆ ---  │
            │ f64  ┆ f64  │
            ╞══════╪══════╡
            │ 24.5 ┆ 74.5 │
            └──────┴──────┘
        """
        return self.__class__(
            lambda plx: self._call(plx).quantile(quantile, interpolation)
        )

    @property
    def str(self) -> ExprStringNamespace:
        return ExprStringNamespace(self)

    @property
    def dt(self) -> ExprDateTimeNamespace:
        return ExprDateTimeNamespace(self)


class ExprStringNamespace:
    def __init__(self, expr: Expr) -> None:
        self._expr = expr

    def ends_with(self, suffix: str) -> Expr:
        return self._expr.__class__(
            lambda plx: self._expr._call(plx).str.ends_with(suffix)
        )

    def head(self, n: int = 5) -> Expr:
        """
        Take the first n elements of each string.

        Arguments:
            n: Number of elements to take.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {'lyrics': ['Atatata', 'taata', 'taatatata', 'zukkyun']}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(lyrics_head = nw.col('lyrics').str.head())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                  lyrics lyrics_head
            0    Atatata       Atata
            1      taata       taata
            2  taatatata       taata
            3    zukkyun       zukky
            >>> func(df_pl)
            shape: (4, 2)
            ┌───────────┬─────────────┐
            │ lyrics    ┆ lyrics_head │
            │ ---       ┆ ---         │
            │ str       ┆ str         │
            ╞═══════════╪═════════════╡
            │ Atatata   ┆ Atata       │
            │ taata     ┆ taata       │
            │ taatatata ┆ taata       │
            │ zukkyun   ┆ zukky       │
            └───────────┴─────────────┘
        """

        def func(plx: Any) -> Any:
            if plx is get_polars():
                return self._expr._call(plx).str.slice(0, n)
            return self._expr._call(plx).str.head(n)

        return self._expr.__class__(func)

    def to_datetime(self, format: str) -> Expr:  # noqa: A002
        """
        Convert to Datetime dtype.

        Notes:
            pandas defaults to nanosecond time unit, Polars to microsecond.
            Prior to pandas 2.0, nanoseconds were the only time unit supported
            in pandas, with no ability to set any other one. The ability to
            set the time unit in pandas, if the version permits, will arrive.

        Arguments:
            format: Format to parse strings with. Must be passed, as different
                    dataframe libraries have different ways of auto-inferring
                    formats.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': ['2020-01-01', '2020-01-02']})
            >>> df_pl = pl.DataFrame({'a': ['2020-01-01', '2020-01-02']})

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.col('a').str.to_datetime(format='%Y-%m-%d'))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                       a
            0 2020-01-01
            1 2020-01-02
            >>> func(df_pl)
            shape: (2, 1)
            ┌─────────────────────┐
            │ a                   │
            │ ---                 │
            │ datetime[μs]        │
            ╞═════════════════════╡
            │ 2020-01-01 00:00:00 │
            │ 2020-01-02 00:00:00 │
            └─────────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._call(plx).str.to_datetime(format=format)
        )


class ExprDateTimeNamespace:
    def __init__(self, expr: Expr) -> None:
        self._expr = expr

    def year(self) -> Expr:
        """
        Extract year from underlying DateTime representation.

        Returns the year number in the calendar date.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = {
            ...        "datetime": [
            ...         datetime(1978, 6, 1),
            ...         datetime(2024, 12, 13),
            ...         datetime(2065, 1, 1),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.year().alias("year")
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                datetime  year
            0 1978-06-01  1978
            1 2024-12-13  2024
            2 2065-01-01  2065
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────────────────────┬──────┐
            │ datetime            ┆ year │
            │ ---                 ┆ ---  │
            │ datetime[μs]        ┆ i32  │
            ╞═════════════════════╪══════╡
            │ 1978-06-01 00:00:00 ┆ 1978 │
            │ 2024-12-13 00:00:00 ┆ 2024 │
            │ 2065-01-01 00:00:00 ┆ 2065 │
            └─────────────────────┴──────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.year())

    def month(self) -> Expr:
        """
        Extract month from underlying DateTime representation.

        Returns the month number starting from 1. The return value ranges from 1 to 12.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = {
            ...        "datetime": [
            ...         datetime(1978, 6, 1),
            ...         datetime(2024, 12, 13),
            ...         datetime(2065, 1, 1),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.year().alias("year"),
            ...         nw.col("datetime").dt.month().alias("month")
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                datetime  year  month
            0 1978-06-01  1978      6
            1 2024-12-13  2024     12
            2 2065-01-01  2065      1
            >>> func(df_pl)
            shape: (3, 3)
            ┌─────────────────────┬──────┬───────┐
            │ datetime            ┆ year ┆ month │
            │ ---                 ┆ ---  ┆ ---   │
            │ datetime[μs]        ┆ i32  ┆ i8    │
            ╞═════════════════════╪══════╪═══════╡
            │ 1978-06-01 00:00:00 ┆ 1978 ┆ 6     │
            │ 2024-12-13 00:00:00 ┆ 2024 ┆ 12    │
            │ 2065-01-01 00:00:00 ┆ 2065 ┆ 1     │
            └─────────────────────┴──────┴───────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.month())

    def day(self) -> Expr:
        """
        Extract day from underlying DateTime representation.

        Returns the day of month starting from 1. The return value ranges from 1 to 31. (The last day of month differs by months.)

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = {
            ...        "datetime": [
            ...         datetime(1978, 6, 1),
            ...         datetime(2024, 12, 13),
            ...         datetime(2065, 1, 1),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.year().alias("year"),
            ...         nw.col("datetime").dt.month().alias("month"),
            ...         nw.col("datetime").dt.day().alias("day")
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                datetime  year  month  day
            0 1978-06-01  1978      6    1
            1 2024-12-13  2024     12   13
            2 2065-01-01  2065      1    1
            >>> func(df_pl)
            shape: (3, 4)
            ┌─────────────────────┬──────┬───────┬─────┐
            │ datetime            ┆ year ┆ month ┆ day │
            │ ---                 ┆ ---  ┆ ---   ┆ --- │
            │ datetime[μs]        ┆ i32  ┆ i8    ┆ i8  │
            ╞═════════════════════╪══════╪═══════╪═════╡
            │ 1978-06-01 00:00:00 ┆ 1978 ┆ 6     ┆ 1   │
            │ 2024-12-13 00:00:00 ┆ 2024 ┆ 12    ┆ 13  │
            │ 2065-01-01 00:00:00 ┆ 2065 ┆ 1     ┆ 1   │
            └─────────────────────┴──────┴───────┴─────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.day())

    def hour(self) -> Expr:
        """
        Extract hour from underlying DateTime representation.

        Returns the hour number from 0 to 23.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1),
            ...         datetime(2024, 10, 13, 5),
            ...         datetime(2065, 1, 1, 10),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.hour().alias("hour")
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                         datetime  hour
            0 1978-01-01 01:00:00     1
            1 2024-10-13 05:00:00     5
            2 2065-01-01 10:00:00    10
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────────────────────┬──────┐
            │ datetime            ┆ hour │
            │ ---                 ┆ ---  │
            │ datetime[μs]        ┆ i8   │
            ╞═════════════════════╪══════╡
            │ 1978-01-01 01:00:00 ┆ 1    │
            │ 2024-10-13 05:00:00 ┆ 5    │
            │ 2065-01-01 10:00:00 ┆ 10   │
            └─────────────────────┴──────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.hour())

    def minute(self) -> Expr:
        """
        Extract minutes from underlying DateTime representation.

        Returns the minute number from 0 to 59.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1, 1),
            ...         datetime(2024, 10, 13, 5, 30),
            ...         datetime(2065, 1, 1, 10, 20),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.hour().alias("hour"),
            ...         nw.col("datetime").dt.minute().alias("minute"),
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                         datetime  hour  minute
            0 1978-01-01 01:01:00     1       1
            1 2024-10-13 05:30:00     5      30
            2 2065-01-01 10:20:00    10      20
            >>> func(df_pl)
            shape: (3, 3)
            ┌─────────────────────┬──────┬────────┐
            │ datetime            ┆ hour ┆ minute │
            │ ---                 ┆ ---  ┆ ---    │
            │ datetime[μs]        ┆ i8   ┆ i8     │
            ╞═════════════════════╪══════╪════════╡
            │ 1978-01-01 01:01:00 ┆ 1    ┆ 1      │
            │ 2024-10-13 05:30:00 ┆ 5    ┆ 30     │
            │ 2065-01-01 10:20:00 ┆ 10   ┆ 20     │
            └─────────────────────┴──────┴────────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.minute())

    def second(self) -> Expr:
        """
        Extract seconds from underlying DateTime representation.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1, 1, 1),
            ...         datetime(2024, 10, 13, 5, 30, 14),
            ...         datetime(2065, 1, 1, 10, 20, 30),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.hour().alias("hour"),
            ...         nw.col("datetime").dt.minute().alias("minute"),
            ...         nw.col("datetime").dt.second().alias("second"),
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                         datetime  hour  minute  second
            0 1978-01-01 01:01:01     1       1       1
            1 2024-10-13 05:30:14     5      30      14
            2 2065-01-01 10:20:30    10      20      30
            >>> func(df_pl)
            shape: (3, 4)
            ┌─────────────────────┬──────┬────────┬────────┐
            │ datetime            ┆ hour ┆ minute ┆ second │
            │ ---                 ┆ ---  ┆ ---    ┆ ---    │
            │ datetime[μs]        ┆ i8   ┆ i8     ┆ i8     │
            ╞═════════════════════╪══════╪════════╪════════╡
            │ 1978-01-01 01:01:01 ┆ 1    ┆ 1      ┆ 1      │
            │ 2024-10-13 05:30:14 ┆ 5    ┆ 30     ┆ 14     │
            │ 2065-01-01 10:20:30 ┆ 10   ┆ 20     ┆ 30     │
            └─────────────────────┴──────┴────────┴────────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.second())

    def millisecond(self) -> Expr:
        """
        Extract milliseconds from underlying DateTime representation.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1, 1, 1, 0),
            ...         datetime(2024, 10, 13, 5, 30, 14, 505000),
            ...         datetime(2065, 1, 1, 10, 20, 30, 67000),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.hour().alias("hour"),
            ...         nw.col("datetime").dt.minute().alias("minute"),
            ...         nw.col("datetime").dt.second().alias("second"),
            ...         nw.col("datetime").dt.millisecond().alias("millisecond")
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                             datetime  hour  minute  second  millisecond
            0 1978-01-01 01:01:01.000     1       1       1            0
            1 2024-10-13 05:30:14.505     5      30      14          505
            2 2065-01-01 10:20:30.067    10      20      30           67
            >>> func(df_pl)
            shape: (3, 5)
            ┌─────────────────────────┬──────┬────────┬────────┬─────────────┐
            │ datetime                ┆ hour ┆ minute ┆ second ┆ millisecond │
            │ ---                     ┆ ---  ┆ ---    ┆ ---    ┆ ---         │
            │ datetime[μs]            ┆ i8   ┆ i8     ┆ i8     ┆ i32         │
            ╞═════════════════════════╪══════╪════════╪════════╪═════════════╡
            │ 1978-01-01 01:01:01     ┆ 1    ┆ 1      ┆ 1      ┆ 0           │
            │ 2024-10-13 05:30:14.505 ┆ 5    ┆ 30     ┆ 14     ┆ 505         │
            │ 2065-01-01 10:20:30.067 ┆ 10   ┆ 20     ┆ 30     ┆ 67          │
            └─────────────────────────┴──────┴────────┴────────┴─────────────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.millisecond())

    def microsecond(self) -> Expr:
        """
        Extract microseconds from underlying DateTime representation.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1, 1, 1, 0),
            ...         datetime(2024, 10, 13, 5, 30, 14, 505000),
            ...         datetime(2065, 1, 1, 10, 20, 30, 67000),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.hour().alias("hour"),
            ...         nw.col("datetime").dt.minute().alias("minute"),
            ...         nw.col("datetime").dt.second().alias("second"),
            ...         nw.col("datetime").dt.microsecond().alias("microsecond")
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                             datetime  hour  minute  second  microsecond
            0 1978-01-01 01:01:01.000     1       1       1            0
            1 2024-10-13 05:30:14.505     5      30      14       505000
            2 2065-01-01 10:20:30.067    10      20      30        67000
            >>> func(df_pl)
            shape: (3, 5)
            ┌─────────────────────────┬──────┬────────┬────────┬─────────────┐
            │ datetime                ┆ hour ┆ minute ┆ second ┆ microsecond │
            │ ---                     ┆ ---  ┆ ---    ┆ ---    ┆ ---         │
            │ datetime[μs]            ┆ i8   ┆ i8     ┆ i8     ┆ i32         │
            ╞═════════════════════════╪══════╪════════╪════════╪═════════════╡
            │ 1978-01-01 01:01:01     ┆ 1    ┆ 1      ┆ 1      ┆ 0           │
            │ 2024-10-13 05:30:14.505 ┆ 5    ┆ 30     ┆ 14     ┆ 505000      │
            │ 2065-01-01 10:20:30.067 ┆ 10   ┆ 20     ┆ 30     ┆ 67000       │
            └─────────────────────────┴──────┴────────┴────────┴─────────────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.microsecond())

    def nanosecond(self) -> Expr:
        """
        Extract Nanoseconds from underlying DateTime representation

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1, 1, 1, 0),
            ...         datetime(2024, 10, 13, 5, 30, 14, 500000),
            ...         datetime(2065, 1, 1, 10, 20, 30, 60000)
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.hour().alias("hour"),
            ...         nw.col("datetime").dt.minute().alias("minute"),
            ...         nw.col("datetime").dt.second().alias("second"),
            ...         nw.col("datetime").dt.nanosecond().alias("nanosecond")
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                             datetime  hour  minute  second  nanosecond
            0 1978-01-01 01:01:01.000     1       1       1           0
            1 2024-10-13 05:30:14.500     5      30      14   500000000
            2 2065-01-01 10:20:30.060    10      20      30    60000000
            >>> func(df_pl)
            shape: (3, 5)
            ┌─────────────────────────┬──────┬────────┬────────┬────────────┐
            │ datetime                ┆ hour ┆ minute ┆ second ┆ nanosecond │
            │ ---                     ┆ ---  ┆ ---    ┆ ---    ┆ ---        │
            │ datetime[μs]            ┆ i8   ┆ i8     ┆ i8     ┆ i32        │
            ╞═════════════════════════╪══════╪════════╪════════╪════════════╡
            │ 1978-01-01 01:01:01     ┆ 1    ┆ 1      ┆ 1      ┆ 0          │
            │ 2024-10-13 05:30:14.500 ┆ 5    ┆ 30     ┆ 14     ┆ 500000000  │
            │ 2065-01-01 10:20:30.060 ┆ 10   ┆ 20     ┆ 30     ┆ 60000000   │
            └─────────────────────────┴──────┴────────┴────────┴────────────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.nanosecond())

    def ordinal_day(self) -> Expr:
        """
        Get ordinal day.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> data = {'a': [datetime(2020, 1, 1), datetime(2020, 8, 3)]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(a_ordinal_day=nw.col('a').dt.ordinal_day())

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                       a  a_ordinal_day
            0 2020-01-01              1
            1 2020-08-03            216
            >>> func(df_pl)
            shape: (2, 2)
            ┌─────────────────────┬───────────────┐
            │ a                   ┆ a_ordinal_day │
            │ ---                 ┆ ---           │
            │ datetime[μs]        ┆ i16           │
            ╞═════════════════════╪═══════════════╡
            │ 2020-01-01 00:00:00 ┆ 1             │
            │ 2020-08-03 00:00:00 ┆ 216           │
            └─────────────────────┴───────────────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.ordinal_day())

    def total_minutes(self) -> Expr:
        """
        Get total minutes.

        Notes:
            The function outputs the total minutes in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` and `cast` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = {'a': [timedelta(minutes=10), timedelta(minutes=20, seconds=40)]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...       a_total_minutes = nw.col('a').dt.total_minutes()
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                            a  a_total_minutes
            0 0 days 00:10:00               10
            1 0 days 00:20:40               20
            >>> func(df_pl)
            shape: (2, 2)
            ┌──────────────┬─────────────────┐
            │ a            ┆ a_total_minutes │
            │ ---          ┆ ---             │
            │ duration[μs] ┆ i64             │
            ╞══════════════╪═════════════════╡
            │ 10m          ┆ 10              │
            │ 20m 40s      ┆ 20              │
            └──────────────┴─────────────────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.total_minutes())

    def total_seconds(self) -> Expr:
        """
        Get total seconds.

        Notes:
            The function outputs the total seconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` and `cast` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = {'a': [timedelta(seconds=10),
            ...     timedelta(seconds=20, milliseconds=40)]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...       a_total_seconds = nw.col('a').dt.total_seconds()
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                                   a  a_total_seconds
            0        0 days 00:00:10               10
            1 0 days 00:00:20.040000               20
            >>> func(df_pl)
            shape: (2, 2)
            ┌──────────────┬─────────────────┐
            │ a            ┆ a_total_seconds │
            │ ---          ┆ ---             │
            │ duration[μs] ┆ i64             │
            ╞══════════════╪═════════════════╡
            │ 10s          ┆ 10              │
            │ 20s 40ms     ┆ 20              │
            └──────────────┴─────────────────┘
        """
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.total_seconds())

    def total_milliseconds(self) -> Expr:
        """
        Get total milliseconds.

        Notes:
            The function outputs the total milliseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` and `cast` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = {'a': [timedelta(milliseconds=10),
            ...     timedelta(milliseconds=20, microseconds=40)]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...       a_total_milliseconds = nw.col('a').dt.total_milliseconds()
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                                   a  a_total_milliseconds
            0 0 days 00:00:00.010000                    10
            1 0 days 00:00:00.020040                    20
            >>> func(df_pl)
            shape: (2, 2)
            ┌──────────────┬──────────────────────┐
            │ a            ┆ a_total_milliseconds │
            │ ---          ┆ ---                  │
            │ duration[μs] ┆ i64                  │
            ╞══════════════╪══════════════════════╡
            │ 10ms         ┆ 10                   │
            │ 20040µs      ┆ 20                   │
            └──────────────┴──────────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._call(plx).dt.total_milliseconds()
        )

    def total_microseconds(self) -> Expr:
        """
        Get total microseconds.

        Notes:
            The function outputs the total microseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` and `cast` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = {'a': [timedelta(microseconds=10),
            ...     timedelta(milliseconds=1, microseconds=200)]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...       a_total_microseconds = nw.col('a').dt.total_microseconds()
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                                   a  a_total_microseconds
            0 0 days 00:00:00.000010                    10
            1 0 days 00:00:00.001200                  1200
            >>> func(df_pl)
            shape: (2, 2)
            ┌──────────────┬──────────────────────┐
            │ a            ┆ a_total_microseconds │
            │ ---          ┆ ---                  │
            │ duration[μs] ┆ i64                  │
            ╞══════════════╪══════════════════════╡
            │ 10µs         ┆ 10                   │
            │ 1200µs       ┆ 1200                 │
            └──────────────┴──────────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._call(plx).dt.total_microseconds()
        )

    def total_nanoseconds(self) -> Expr:
        """
        Get total nanoseconds.

        Notes:
            The function outputs the total nanoseconds in the int dtype by default,
            however, pandas may change the dtype to float when there are missing values,
            consider using `fill_null()` and `cast` in this case.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> from datetime import timedelta
            >>> import narwhals as nw
            >>> data = ['2024-01-01 00:00:00.000000001',
            ...     '2024-01-01 00:00:00.000000002']
            >>> df_pd = pd.DataFrame({'a': pd.to_datetime(data)})
            >>> df_pl = (pl.DataFrame({'a': data}).with_columns(
            ...     pl.col('a').str.to_datetime(time_unit='ns')))

            We define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns(
            ...       a_diff_total_nanoseconds = nw.col('a').diff().dt.total_nanoseconds()
            ...     )

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                                          a  a_diff_total_nanoseconds
            0 2024-01-01 00:00:00.000000001                       NaN
            1 2024-01-01 00:00:00.000000002                       1.0
            >>> func(df_pl)
            shape: (2, 2)
            ┌───────────────────────────────┬──────────────────────────┐
            │ a                             ┆ a_diff_total_nanoseconds │
            │ ---                           ┆ ---                      │
            │ datetime[ns]                  ┆ i64                      │
            ╞═══════════════════════════════╪══════════════════════════╡
            │ 2024-01-01 00:00:00.000000001 ┆ null                     │
            │ 2024-01-01 00:00:00.000000002 ┆ 1                        │
            └───────────────────────────────┴──────────────────────────┘
        """
        return self._expr.__class__(
            lambda plx: self._expr._call(plx).dt.total_nanoseconds()
        )


def col(*names: str | Iterable[str]) -> Expr:
    """
    Creates an expression that references one or more columns by their name(s).

    Arguments:
        names: Name(s) of the columns to use in the aggregation function.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pl = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        We define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.col('a') * nw.col('b'))

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
           a
        0  3
        1  8
        >>> func(df_pl)
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 3   │
        │ 8   │
        └─────┘
    """
    return Expr(lambda plx: plx.col(*names))


def all() -> Expr:
    """
    Instantiate an expression representing all columns.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> df_pl = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...    return df.select(nw.all() * 2)

        We can then pass either pandas or Polars to `func`:

        >>> func(df_pd)
           a   b
        0  2   8
        1  4  10
        2  6  12
        >>> func(df_pl)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 8   │
        │ 4   ┆ 10  │
        │ 6   ┆ 12  │
        └─────┴─────┘
    """
    return Expr(lambda plx: plx.all())


def len() -> Expr:
    """
    Return the number of rows.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({'a': [1, 2], 'b': [5, 10]})
        >>> df_pl = pl.DataFrame({'a': [1, 2], 'b': [5, 10]})

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...    return df.select(nw.len())

        We can then pass either pandas or Polars to `func`:

        >>> func(df_pd)
           len
        0    2
        >>> func(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ len │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 2   │
        └─────┘
    """

    def func(plx: Any) -> Any:
        if (
            (pl := get_polars()) is not None
            and plx is pl
            and parse_version(pl.__version__) < parse_version("0.20.4")
        ):  # pragma: no cover
            return plx.count().alias("len")
        return plx.len()

    return Expr(func)


def sum(*columns: str) -> Expr:
    """
    Sum all values.

    Note:
        Syntactic sugar for ``nw.col(columns).sum()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pl = pl.DataFrame({"a": [1, 2]})
        >>> df_pd = pd.DataFrame({"a": [1, 2]})

        We define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.sum('a'))

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
           a
        0  3
        >>> func(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 3   │
        └─────┘
    """

    return Expr(lambda plx: plx.sum(*columns))


def mean(*columns: str) -> Expr:
    """
    Get the mean value.

    Note:
        Syntactic sugar for ``nw.col(columns).mean()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pl = pl.DataFrame({"a": [1, 8, 3]})
        >>> df_pd = pd.DataFrame({"a": [1, 8, 3]})

        We define a dataframe agnostic function:

        >>> def func(df_any):
        ...     df = nw.from_native(df_any)
        ...     df = df.select(nw.mean('a'))
        ...     return nw.to_native(df)

        We can then pass either pandas or Polars to `func`:

        >>> func(df_pd)
             a
        0  4.0
        >>> func(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 4.0 │
        └─────┘
    """

    return Expr(lambda plx: plx.mean(*columns))


def min(*columns: str) -> Expr:
    """
    Return the minimum value.

    Note:
       Syntactic sugar for ``nw.col(columns).min()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({'a': [1, 2], 'b': [5, 10]})
        >>> df_pl = pl.DataFrame({'a': [1, 2], 'b': [5, 10]})

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...    return df.select(nw.min('b'))

        We can then pass either pandas or Polars to `func`:

        >>> func(df_pd)
           b
        0  5
        >>> func(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 5   │
        └─────┘
    """
    return Expr(lambda plx: plx.min(*columns))


def max(*columns: str) -> Expr:
    """
    Return the maximum value.

    Note:
       Syntactic sugar for ``nw.col(columns).max()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({'a': [1, 2], 'b': [5, 10]})
        >>> df_pl = pl.DataFrame({'a': [1, 2], 'b': [5, 10]})

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...    return df.select(nw.max('a'))

        We can then pass either pandas or Polars to `func`:

        >>> func(df_pd)
           a
        0  2
        >>> func(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        └─────┘
    """
    return Expr(lambda plx: plx.max(*columns))


def sum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Sum all values horizontally across columns

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts expression input.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pl = pl.DataFrame({"a": [1, 2, 3], "b": [5, 10, 15]})
        >>> df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [5, 10, 15]})

        We define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.sum_horizontal('a', 'b'))

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
            a
        0   6
        1  12
        2  18
        >>> func(df_pl)
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 6   │
        │ 12  │
        │ 18  │
        └─────┘

    """
    return Expr(
        lambda plx: plx.sum_horizontal([extract_native(plx, v) for v in flatten(exprs)])
    )


__all__ = [
    "Expr",
]
