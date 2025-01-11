from __future__ import annotations

EXAMPLES = {
    "get_categories": """
            Let's create some dataframes:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"fruits": ["apple", "mango", "mango"]}
            >>> df_pd = pd.DataFrame(data, dtype="category")
            >>> df_pl = pl.DataFrame(data, schema={"fruits": pl.Categorical})

            We define a dataframe-agnostic function to get unique categories
            from column 'fruits':

            >>> def agnostic_cat_get_categories(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("fruits").cat.get_categories()).to_native()

           We can then pass any supported library such as pandas or Polars to
           `agnostic_cat_get_categories`:

            >>> agnostic_cat_get_categories(df_pd)
              fruits
            0  apple
            1  mango

            >>> agnostic_cat_get_categories(df_pl)
            shape: (2, 1)
            ┌────────┐
            │ fruits │
            │ ---    │
            │ str    │
            ╞════════╡
            │ apple  │
            │ mango  │
            └────────┘
        """,
}
