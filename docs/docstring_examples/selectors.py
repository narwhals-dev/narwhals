from __future__ import annotations

EXAMPLES = {
    "by_dtype": """
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
    """,
    "numeric": """
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
    """,
    "boolean": """
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
    """,
    "string": """
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
    """,
    "categorical": """
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
    """,
    "all": """
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
    """,
}
