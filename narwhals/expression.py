from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable

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

            We define a data-frame agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.select(nw.col('a', 'b').any())
            ...     return nw.to_native(df)

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

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.select(nw.col('a', 'b').all())
            ...     return nw.to_native(df)

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

            >>> def func(df_any):
            ...    df = nw.from_native(df_any)
            ...    df = df.select(nw.col('a', 'b').mean())
            ...    return nw.to_native(df)

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

            >>> def func(df_any):
            ...    df = nw.from_native(df_any)
            ...    df = df.select(nw.col('a', 'b').std(ddof=0))
            ...    return nw.to_native(df)

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

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.select(nw.col('a', 'b').sum())
            ...     return nw.to_native(df)

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
            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.select(nw.min('a','b'))
            ...     return nw.to_native(df)

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

            >>> def func(df_any):
            ...    df = nw.from_native(df_any)
            ...    df = df.select(nw.max('a', 'b'))
            ...    return nw.to_native(df)

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

            >>> def func(df_any):
            ...    df = nw.from_native(df_any)
            ...    df = df.select(nw.col('a', 'b').n_unique())
            ...    return nw.to_native(df)

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
         Returns unique values

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [1, 1, 3, 5, 5], 'b': [2, 4, 4, 6, 6]})
            >>> df_pl = pl.DataFrame({'a': [1, 1, 3, 5, 5], 'b': [2, 4, 4, 6, 6]})

            Let's define a dataframe-agnostic function:

            >>> def func(df_any):
            ...    df = nw.from_native(df_any)
            ...    df = df.select(nw.col('a', 'b').unique())
            ...    return nw.to_native(df)

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

    def sort(self, *, descending: bool = False) -> Expr:
        return self.__class__(lambda plx: self._call(plx).sort(descending=descending))

    # --- transform ---
    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).is_between(lower_bound, upper_bound, closed)
        )

    def is_in(self, other: Any) -> Expr:
        return self.__class__(lambda plx: self._call(plx).is_in(other))

    def filter(self, other: Any) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).filter(extract_native(plx, other))
        )

    def is_null(self) -> Expr:
        """
        Returns a boolean Series indicating which values are null.

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

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.with_columns(
            ...         a_is_null = nw.col('a').is_null(),
            ...         b_is_null = nw.col('b').is_null()
            ...     )
            ...     return nw.to_native(df)

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

    # --- partial reduction ---
    def drop_nulls(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).drop_nulls())

    def sample(
        self,
        n: int | None = None,
        fraction: float | None = None,
        *,
        with_replacement: bool = False,
    ) -> Expr:
        return self.__class__(
            lambda plx: self._call(plx).sample(
                n, fraction=fraction, with_replacement=with_replacement
            )
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


class ExprDateTimeNamespace:
    def __init__(self, expr: Expr) -> None:
        self._expr = expr

    def year(self) -> Expr:
        return self._expr.__class__(lambda plx: self._expr._call(plx).dt.year())


def col(*names: str | Iterable[str]) -> Expr:
    """
    Instantiate an expression, similar to `polars.col`.
    """
    return Expr(lambda plx: plx.col(*names))


def all() -> Expr:
    """
    Instantiate an expression representing all columns, similar to `polars.all`.
    """
    return Expr(lambda plx: plx.all())


def len() -> Expr:
    """
    Instantiate an expression representing the length of a dataframe, similar to `polars.len`.
    """

    def func(plx: Any) -> Any:
        if (
            not hasattr(plx, "_implementation")
            and (pl := get_polars()) is not None
            and parse_version(pl.__version__) < parse_version("0.20.4")
        ):  # pragma: no cover
            return plx.count()
        return plx.len()

    return Expr(func)


def sum(*columns: str) -> Expr:
    """
    Instantiate an expression representing the sum of one or more columns, similar to `polars.sum`.
    """
    return Expr(lambda plx: plx.sum(*columns))


def mean(*columns: str) -> Expr:
    """
    Instantiate an expression representing the mean of one or more columns, similar to `polars.mean`.
    """
    return Expr(lambda plx: plx.mean(*columns))


def min(*columns: str) -> Expr:
    """
    Instantiate an expression representing the minimum of one or more columns, similar to `polars.min`.
    """
    return Expr(lambda plx: plx.min(*columns))


def max(*columns: str) -> Expr:
    """
    Instantiate an expression representing the maximum of one or more columns, similar to `polars.max`.
    """
    return Expr(lambda plx: plx.max(*columns))


def sum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Instantiate an expression representing the horizontal sum of one or more expressions, similar to `polars.sum_horizontal`.
    """
    return Expr(
        lambda plx: plx.sum_horizontal([extract_native(plx, v) for v in flatten(exprs)])
    )


__all__ = [
    "Expr",
]
