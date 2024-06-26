from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._expression import Expr
from narwhals.dtypes import translate_dtype
from narwhals.utils import flatten
from narwhals.versions import DEFAULT_API_VERSION

if TYPE_CHECKING:
    from narwhals.typing import API_VERSION


class Selector(Expr): ...


def by_dtype(*dtypes: Any, api_version: API_VERSION = DEFAULT_API_VERSION) -> Expr:
    """
    Select columns based on their dtype.

    Arguments:
        dtypes: one or data types to select
        api_version: Narwhals API version to use, if you want to ensure perfect
            backwards-compatibility. The easiest way to use this is to set it once
            in `narwhals.StableAPI` and then just import that.

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

        >>> def func(df_any):
        ...     df = nw.from_native(df_any)
        ...     df = df.select(ncs.by_dtype(nw.Int64, nw.Float64) * 2)
        ...     return nw.to_native(df)

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
        lambda plx: plx.selectors.by_dtype(
            [translate_dtype(plx, dtype) for dtype in flatten(dtypes)]
        ),
        api_version=api_version or DEFAULT_API_VERSION,
    )


def numeric(api_version: API_VERSION = DEFAULT_API_VERSION) -> Expr:
    """
    Select numeric columns.

    Arguments:
        api_version: Narwhals API version to use, if you want to ensure perfect
            backwards-compatibility. The easiest way to use this is to set it once
            in `narwhals.StableAPI` and then just import that.

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

        >>> def func(df_any):
        ...     df = nw.from_native(df_any)
        ...     df = df.select(ncs.numeric() * 2)
        ...     return nw.to_native(df)

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
        api_version=api_version or DEFAULT_API_VERSION,
    )


def boolean(api_version: API_VERSION = DEFAULT_API_VERSION) -> Expr:
    """
    Select boolean columns.

    Arguments:
        api_version: Narwhals API version to use, if you want to ensure perfect
            backwards-compatibility. The easiest way to use this is to set it once
            in `narwhals.StableAPI` and then just import that.

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

        >>> def func(df_any):
        ...     df = nw.from_native(df_any)
        ...     df = df.select(ncs.boolean())
        ...     return nw.to_native(df)

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
        api_version=api_version or DEFAULT_API_VERSION,
    )


def string(api_version: API_VERSION = DEFAULT_API_VERSION) -> Expr:
    """
    Select string columns.

    Arguments:
        api_version: Narwhals API version to use, if you want to ensure perfect
            backwards-compatibility. The easiest way to use this is to set it once
            in `narwhals.StableAPI` and then just import that.

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

        >>> def func(df_any):
        ...     df = nw.from_native(df_any)
        ...     df = df.select(ncs.string())
        ...     return nw.to_native(df)

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
        lambda plx: plx.selectors.string(), api_version=api_version or DEFAULT_API_VERSION
    )


def categorical(api_version: API_VERSION = DEFAULT_API_VERSION) -> Expr:
    """
    Select categorical columns.

    Arguments
        api_version: Narwhals API version to use, if you want to ensure perfect
            backwards-compatibility. The easiest way to use this is to set it once
            in `narwhals.StableAPI` and then just import that.

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

        >>> def func(df_any):
        ...     df = nw.from_native(df_any)
        ...     df = df.select(ncs.categorical())
        ...     return nw.to_native(df)

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
        api_version=api_version or DEFAULT_API_VERSION,
    )


def all(api_version: API_VERSION = DEFAULT_API_VERSION) -> Expr:
    """
    Select all columns.

    Arguments:
        api_version: Narwhals API version to use, if you want to ensure perfect
            backwards-compatibility. The easiest way to use this is to set it once
            in `narwhals.StableAPI` and then just import that.

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

        >>> def func(df_any):
        ...     df = nw.from_native(df_any)
        ...     df = df.select(ncs.all())
        ...     return nw.to_native(df)

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
        lambda plx: plx.selectors.all(), api_version=api_version or DEFAULT_API_VERSION
    )
