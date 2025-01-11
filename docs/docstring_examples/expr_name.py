from __future__ import annotations

EXAMPLES = {
    "keep": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrame
            >>>
            >>> data = {"foo": [1, 2], "BAR": [4, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_name_keep(df_native: IntoFrame) -> list[str]:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("foo").alias("alias_for_foo").name.keep()).columns

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_name_keep`:

            >>> agnostic_name_keep(df_pd)
            ['foo']

            >>> agnostic_name_keep(df_pl)
            ['foo']

            >>> agnostic_name_keep(df_pa)
            ['foo']
        """,
    "map": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrame
            >>>
            >>> data = {"foo": [1, 2], "BAR": [4, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> renaming_func = lambda s: s[::-1]  # reverse column name
            >>> def agnostic_name_map(df_native: IntoFrame) -> list[str]:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("foo", "BAR").name.map(renaming_func)).columns

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_name_map`:

            >>> agnostic_name_map(df_pd)
            ['oof', 'RAB']

            >>> agnostic_name_map(df_pl)
            ['oof', 'RAB']

            >>> agnostic_name_map(df_pa)
            ['oof', 'RAB']
        """,
    "prefix": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrame
            >>>
            >>> data = {"foo": [1, 2], "BAR": [4, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_name_prefix(df_native: IntoFrame, prefix: str) -> list[str]:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("foo", "BAR").name.prefix(prefix)).columns

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_name_prefix`:

            >>> agnostic_name_prefix(df_pd, "with_prefix_")
            ['with_prefix_foo', 'with_prefix_BAR']

            >>> agnostic_name_prefix(df_pl, "with_prefix_")
            ['with_prefix_foo', 'with_prefix_BAR']

            >>> agnostic_name_prefix(df_pa, "with_prefix_")
            ['with_prefix_foo', 'with_prefix_BAR']
        """,
    "suffix": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrame
            >>>
            >>> data = {"foo": [1, 2], "BAR": [4, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_name_suffix(df_native: IntoFrame, suffix: str) -> list[str]:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("foo", "BAR").name.suffix(suffix)).columns

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_name_suffix`:

            >>> agnostic_name_suffix(df_pd, "_with_suffix")
            ['foo_with_suffix', 'BAR_with_suffix']

            >>> agnostic_name_suffix(df_pl, "_with_suffix")
            ['foo_with_suffix', 'BAR_with_suffix']

            >>> agnostic_name_suffix(df_pa, "_with_suffix")
            ['foo_with_suffix', 'BAR_with_suffix']
        """,
    "to_lowercase": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrame
            >>>
            >>> data = {"foo": [1, 2], "BAR": [4, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_name_to_lowercase(df_native: IntoFrame) -> list[str]:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("foo", "BAR").name.to_lowercase()).columns

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_name_to_lowercase`:

            >>> agnostic_name_to_lowercase(df_pd)
            ['foo', 'bar']

            >>> agnostic_name_to_lowercase(df_pl)
            ['foo', 'bar']

            >>> agnostic_name_to_lowercase(df_pa)
            ['foo', 'bar']
        """,
    "to_uppercase": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrame
            >>>
            >>> data = {"foo": [1, 2], "BAR": [4, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_name_to_uppercase(df_native: IntoFrame) -> list[str]:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("foo", "BAR").name.to_uppercase()).columns

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_name_to_uppercase`:

            >>> agnostic_name_to_uppercase(df_pd)
            ['FOO', 'BAR']

            >>> agnostic_name_to_uppercase(df_pl)
            ['FOO', 'BAR']

            >>> agnostic_name_to_uppercase(df_pa)
            ['FOO', 'BAR']
        """,
}
