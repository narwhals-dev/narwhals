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
        """
<<<<<<< HEAD
        Redefine an objects data type.

        Arguments:
            dtype: Data type that the object will be cast into.

        Examples:
                >>> import pandas as pd
                >>> import narwhals as nw
                >>> from datetime import date
                >>> df_pd = pd.DataFrame(
                >>> {"foo": [1, 2, 3],
                >>> "bar": [6.0, 7.0, 8.0],
                >>> "ham": [date(2020, 1, 2), date(2021, 3, 4), date(2022, 5, 6)],})
                >>> df_pl = pl.DataFrame
                >>> {"foo": [1, 2, 3],
                >>> "bar": [6.0, 7.0, 8.0],
                >>> "ham": [date(2020, 1, 2), date(2021, 3, 4), date(2022, 5, 6)],})

                Let's define a dataframe-agnostic function:

                >>> def func(df_any):
                    ... df = nw.from_native(df_any)
                    ... df = df.select(nw.col('foo').cast(nw.Float32), nw.col('bar'): nw.UInt8))
                    ... return nw.to_native(df)

                >>> func(df_pd)
                    foo     bar           ham
                0   1.0     6      2020-01-02
                1   2.0     7      2021-03-04
                1   3.0     8      2022-05-06

                >>> func(df_pl)
                shape: (3, 3)
                ┌─────┬─────┬────────────┐
                │ foo ┆ bar ┆ ham        │
                │ --- ┆ --- ┆ ---        │
                │ f32 ┆ u8  ┆ date       │
                ╞═════╪═════╪════════════╡
                │ 1.0 ┆ 6   ┆ 2020-01-02 │
                │ 2.0 ┆ 7   ┆ 2021-03-04 │
                │ 3.0 ┆ 8   ┆ 2022-05-06 │
                └─────┴─────┴────────────┘


                >>> df = df.cast({"foo": pl.Float32, "bar": pl.UInt8})
                >>> nw.to_native(df)
                shape: (3, 3)
                ┌─────┬─────┬────────────┐
                │ foo ┆ bar ┆ ham        │
                │ --- ┆ --- ┆ ---        │
                │ f32 ┆ u8  ┆ date       │
                ╞═════╪═════╪════════════╡
                │ 1.0 ┆ 6   ┆ 2020-01-02 │
                │ 2.0 ┆ 7   ┆ 2021-03-04 │
                │ 3.0 ┆ 8   ┆ 2022-05-06 │
                └─────┴─────┴────────────┘    

            Let's Define a dataframe-agnostic function:
                >>> def func(df_any):
                    ... df = nw.from_native(df_any)
                    ... df.select(nw.col('ham').(cast(nw.Date: nw.Datetime))
                    ... return nw.to_native(df)
=======
                        Redefine an objects data type.

                        Arguments:
                            dtype: Data type that the object will be cast into.

                    Examples:
                            >>> import pandas as pd
                            >>> import narwhals as nw
                            >>> from datetime import date
                            >>> df_pd = pd.DataFrame(
                            >>> {"foo": [1, 2, 3],
                            >>> "bar": [6.0, 7.0, 8.0],
                            >>> "ham": [date(2020, 1, 2), date(2021, 3, 4), date(2022, 5, 6)],})
                            >>> df_pl = pl.DataFrame
                            >>> {"foo": [1, 2, 3],
                            >>> "bar": [6.0, 7.0, 8.0],
                            >>> "ham": [date(2020, 1, 2), date(2021, 3, 4), date(2022, 5, 6)],})

                            Let's define a dataframe-agnostic function:

                            >>> def func(df_any):
                                ... df = nw.from_native(df_any)
                                ... df = df.select(nw.col('foo').cast(nw.Float32), nw.col('bar'): nw.UInt8))
                                ... return nw.to_native(df)

                            >>> func(df_pd)
                                foo     bar           ham
                            0   1.0     6      2020-01-02
                            1   2.0     7      2021-03-04
                            1   3.0     8      2022-05-06

                            >>> func(df_pl)
                            shape: (3, 3)
                            ┌─────┬─────┬────────────┐
                            │ foo ┆ bar ┆ ham        │
                            │ --- ┆ --- ┆ ---        │
                            │ f32 ┆ u8  ┆ date       │
                            ╞═════╪═════╪════════════╡
                            │ 1.0 ┆ 6   ┆ 2020-01-02 │
                            │ 2.0 ┆ 7   ┆ 2021-03-04 │
                            │ 3.0 ┆ 8   ┆ 2022-05-06 │
                            └─────┴─────┴────────────┘


                                >>> df = df.cast({"foo": pl.Float32, "bar": pl.UInt8})
                                >>> nw.to_native(df)
                                shape: (3, 3)
                                ┌─────┬─────┬────────────┐
                                │ foo ┆ bar ┆ ham        │
                                │ --- ┆ --- ┆ ---        │
                                │ f32 ┆ u8  ┆ date       │
                                ╞═════╪═════╪════════════╡
                                │ 1.0 ┆ 6   ┆ 2020-01-02 │
                                │ 2.0 ┆ 7   ┆ 2021-03-04 │
                                │ 3.0 ┆ 8   ┆ 2022-05-06 │
                                └─────┴─────┴────────────┘

        <<<<<<< HEAD
                    Let's Define a dataframe agnostic function:
                    >>> def func(df_any):
                        ... df = nw.from_native(df_any)
                        ... df.select(nw.col('ham').(cast(nw.Date: nw.Datetime))
                        ... return nw.to_native(df)
        =======
                <<<<<<< HEAD
                            Let's Define a dataframe agnostic function:
                            >>> def func(df_any):
                                ... df = nw.from_native(df_any)
                                ... df.select(nw.col('ham').(cast(nw.Date: nw.Datetime))
                                ... return nw.to_native(df)
        >>>>>>> ec9edcade5719d6c5058dde65c77f047fe6af0bc
>>>>>>> a6a276cdf68a35db465d8255274f090059f25a98

                            >>> func(df_pd)
                            foo  bar                  ham
                            1    6.0  2020-01-02 00:00:00
                            2    7.0  2021-03-04 00:00:00
                            3    8.0  2022-05-06 00:00:00

                            >>> func(df_pl)
                            shape: (3, 3)
                            ┌─────┬─────┬─────────────────────┐
                            │ foo ┆ bar ┆ ham                 │
                            │ --- ┆ --- ┆ ---                 │
                            │ i64 ┆ f64 ┆ datetime[μs]        │
                            ╞═════╪═════╪═════════════════════╡
                            │ 1   ┆ 6.0 ┆ 2020-01-02 00:00:00 │
                            │ 2   ┆ 7.0 ┆ 2021-03-04 00:00:00 │
                            │ 3   ┆ 8.0 ┆ 2022-05-06 00:00:00 │
                            └─────┴─────┴─────────────────────┘
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
        Return whether any values are True

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame({'a': [True, False], 'b': [True, True]})
            >>> df_pl = pl.DataFrame({'a': [True, False], 'b': [True, True]})

<<<<<<< HEAD
            Let's define a dataframe-agnostic function: 
=======
            Let's define a datafram agnostic function:
>>>>>>> a6a276cdf68a35db465d8255274f090059f25a98

            >>> def func(df_any)
                ... df = nw.from_native(df_any)
                ... df = df.select(nw.col('a', 'b').any())
                ... return nw.to_native(df)

            We can then pass any type of dataframe to `func`:

            >>> func(df_pd)
                   a        b
            0   True    False
            >>> func(df_pl)
            shape: (1,2)
            ┌───────┬──────┐
            │ a     ┆ b    │
            │ ---   ┆ ---  │
            │ bool  ┆ bool │
            ╞═══════╪══════╡
            │ true  ┆ true │
            └───────┴──────┘

        """
        return self.__class__(lambda plx: self._call(plx).any())

    def all(self) -> Expr:
        """
        Return whether all values are True.

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
        return self.__class__(lambda plx: self._call(plx).mean())

    def std(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).std())

    def sum(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).sum())

    def min(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).min())

    def max(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).max())

    def n_unique(self) -> Expr:
        return self.__class__(lambda plx: self._call(plx).n_unique())

    def unique(self) -> Expr:
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
