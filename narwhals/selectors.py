from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn

from narwhals.expr import Expr
from narwhals.utils import flatten

if TYPE_CHECKING:
    from typing_extensions import Self


class Selector(Expr):
    def _to_expr(self: Self) -> Expr:
        return Expr(
            to_compliant_expr=self._to_compliant_expr,
            is_order_dependent=self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def __add__(self: Self, other: Any) -> Expr:  # type: ignore[override]
        if isinstance(other, Selector):
            msg = "unsupported operand type(s) for op: ('Selector' + 'Selector')"
            raise TypeError(msg)
        return self._to_expr() + other  # type: ignore[no-any-return]

    def __rsub__(self: Self, other: Any) -> NoReturn:
        raise NotImplementedError

    def __rand__(self: Self, other: Any) -> NoReturn:
        raise NotImplementedError

    def __ror__(self: Self, other: Any) -> NoReturn:
        raise NotImplementedError


def by_dtype(*dtypes: Any) -> Expr:
    """Select columns based on their dtype.

    Arguments:
        dtypes: one or data types to select

    Returns:
        A new expression.

    Examples:
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> import pandas as pd
        >>> import polars as pl
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [4.1, 2.3]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)

        Let's define a dataframe-agnostic function to select int64 and float64
        dtypes and multiplies each value by 2:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(ncs.by_dtype(nw.Int64, nw.Float64) * 2)

        We can then pass either pandas or Polars dataframes:

        >>> func(df_pd)
           a    c
        0  2  8.2
        1  4  4.6
        >>> func(df_pl)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ c   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 2   ┆ 8.2 │
        │ 4   ┆ 4.6 │
        └─────┴─────┘
    """
    return Selector(
        lambda plx: plx.selectors.by_dtype(flatten(dtypes)),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def matches(pattern: str) -> Expr:
    """Select all columns that match the given regex pattern.

    Arguments:
        pattern: A valid regular expression pattern.

    Returns:
        A new expression.

    Examples:
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> import pandas as pd
        >>> import polars as pl
        >>>
        >>> data = {
        ...     "foo": ["x", "y"],
        ...     "bar": [123, 456],
        ...     "baz": [2.0, 5.5],
        ...     "zap": [0, 1],
        ... }
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)

        Let's define a dataframe-agnostic function to select column names
        containing an 'a', preceded by a character that is not 'z':

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(ncs.matches("[^z]a"))

        We can then pass either pandas or Polars dataframes:

        >>> func(df_pd)
           bar  baz
        0  123  2.0
        1  456  5.5
        >>> func(df_pl)
        shape: (2, 2)
        ┌─────┬─────┐
        │ bar ┆ baz │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 123 ┆ 2.0 │
        │ 456 ┆ 5.5 │
        └─────┴─────┘
    """
    return Selector(
        lambda plx: plx.selectors.matches(pattern),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def numeric() -> Expr:
    """Select numeric columns.

    Returns:
        A new expression.

    Examples:
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> import pandas as pd
        >>> import polars as pl
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [4.1, 2.3]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)

        Let's define a dataframe-agnostic function to select numeric
        dtypes and multiplies each value by 2:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(ncs.numeric() * 2)

        We can then pass either pandas or Polars dataframes:

        >>> func(df_pd)
           a    c
        0  2  8.2
        1  4  4.6
        >>> func(df_pl)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ c   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 2   ┆ 8.2 │
        │ 4   ┆ 4.6 │
        └─────┴─────┘
    """
    return Selector(
        lambda plx: plx.selectors.numeric(),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def boolean() -> Expr:
    """Select boolean columns.

    Returns:
        A new expression.

    Examples:
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> import pandas as pd
        >>> import polars as pl
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)

        Let's define a dataframe-agnostic function to select boolean
        dtypes:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(ncs.boolean())

        We can then pass either pandas or Polars dataframes:

        >>> func(df_pd)
               c
        0  False
        1   True
        >>> func(df_pl)
        shape: (2, 1)
        ┌───────┐
        │ c     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ false │
        │ true  │
        └───────┘
    """
    return Selector(
        lambda plx: plx.selectors.boolean(),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def string() -> Expr:
    """Select string columns.

    Returns:
        A new expression.

    Examples:
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> import pandas as pd
        >>> import polars as pl
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)

        Let's define a dataframe-agnostic function to select string
        dtypes:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(ncs.string())

        We can then pass either pandas or Polars dataframes:

        >>> func(df_pd)
           b
        0  x
        1  y
        >>> func(df_pl)
        shape: (2, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ str │
        ╞═════╡
        │ x   │
        │ y   │
        └─────┘
    """
    return Selector(
        lambda plx: plx.selectors.string(),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def categorical() -> Expr:
    """Select categorical columns.

    Returns:
        A new expression.

    Examples:
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> import pandas as pd
        >>> import polars as pl
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data).astype({"b": "category"})
        >>> df_pl = pl.DataFrame(data, schema_overrides={"b": pl.Categorical})

        Let's define a dataframe-agnostic function to select string
        dtypes:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(ncs.categorical())

        We can then pass either pandas or Polars dataframes:

        >>> func(df_pd)
           b
        0  x
        1  y
        >>> func(df_pl)
        shape: (2, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ cat │
        ╞═════╡
        │ x   │
        │ y   │
        └─────┘
    """
    return Selector(
        lambda plx: plx.selectors.categorical(),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


def all() -> Expr:
    """Select all columns.

    Returns:
        A new expression.

    Examples:
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> import pandas as pd
        >>> import polars as pl
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data).astype({"b": "category"})
        >>> df_pl = pl.DataFrame(data, schema_overrides={"b": pl.Categorical})

        Let's define a dataframe-agnostic function to select string
        dtypes:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(ncs.all())

        We can then pass either pandas or Polars dataframes:

        >>> func(df_pd)
           a  b      c
        0  1  x  False
        1  2  y   True
        >>> func(df_pl)
        shape: (2, 3)
        ┌─────┬─────┬───────┐
        │ a   ┆ b   ┆ c     │
        │ --- ┆ --- ┆ ---   │
        │ i64 ┆ cat ┆ bool  │
        ╞═════╪═════╪═══════╡
        │ 1   ┆ x   ┆ false │
        │ 2   ┆ y   ┆ true  │
        └─────┴─────┴───────┘
    """
    return Selector(
        lambda plx: plx.selectors.all(),
        is_order_dependent=False,
        changes_length=False,
        aggregates=False,
    )


__all__ = [
    "all",
    "boolean",
    "by_dtype",
    "categorical",
    "numeric",
    "string",
]
