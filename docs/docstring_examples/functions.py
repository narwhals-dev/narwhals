from __future__ import annotations

EXAMPLES = {
    "concat": """
        Let's take an example of vertical concatenation:

        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> data_1 = {"a": [1, 2, 3], "b": [4, 5, 6]}
        >>> data_2 = {"a": [5, 2], "b": [1, 4]}

        >>> df_pd_1 = pd.DataFrame(data_1)
        >>> df_pd_2 = pd.DataFrame(data_2)
        >>> df_pl_1 = pl.DataFrame(data_1)
        >>> df_pl_2 = pl.DataFrame(data_2)

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def agnostic_vertical_concat(df1, df2):
        ...     return nw.concat([df1, df2], how="vertical")

        >>> agnostic_vertical_concat(df_pd_1, df_pd_2)
           a  b
        0  1  4
        1  2  5
        2  3  6
        0  5  1
        1  2  4
        >>> agnostic_vertical_concat(df_pl_1, df_pl_2)
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        │ 5   ┆ 1   │
        │ 2   ┆ 4   │
        └─────┴─────┘

        Let's look at case a for horizontal concatenation:

        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> data_1 = {"a": [1, 2, 3], "b": [4, 5, 6]}
        >>> data_2 = {"c": [5, 2], "d": [1, 4]}

        >>> df_pd_1 = pd.DataFrame(data_1)
        >>> df_pd_2 = pd.DataFrame(data_2)
        >>> df_pl_1 = pl.DataFrame(data_1)
        >>> df_pl_2 = pl.DataFrame(data_2)

        Defining a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def agnostic_horizontal_concat(df1, df2):
        ...     return nw.concat([df1, df2], how="horizontal")

        >>> agnostic_horizontal_concat(df_pd_1, df_pd_2)
           a  b    c    d
        0  1  4  5.0  1.0
        1  2  5  2.0  4.0
        2  3  6  NaN  NaN

        >>> agnostic_horizontal_concat(df_pl_1, df_pl_2)
        shape: (3, 4)
        ┌─────┬─────┬──────┬──────┐
        │ a   ┆ b   ┆ c    ┆ d    │
        │ --- ┆ --- ┆ ---  ┆ ---  │
        │ i64 ┆ i64 ┆ i64  ┆ i64  │
        ╞═════╪═════╪══════╪══════╡
        │ 1   ┆ 4   ┆ 5    ┆ 1    │
        │ 2   ┆ 5   ┆ 2    ┆ 4    │
        │ 3   ┆ 6   ┆ null ┆ null │
        └─────┴─────┴──────┴──────┘

        Let's look at case a for diagonal concatenation:

        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> data_1 = {"a": [1, 2], "b": [3.5, 4.5]}
        >>> data_2 = {"a": [3, 4], "z": ["x", "y"]}

        >>> df_pd_1 = pd.DataFrame(data_1)
        >>> df_pd_2 = pd.DataFrame(data_2)
        >>> df_pl_1 = pl.DataFrame(data_1)
        >>> df_pl_2 = pl.DataFrame(data_2)

        Defining a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def agnostic_diagonal_concat(df1, df2):
        ...     return nw.concat([df1, df2], how="diagonal")

        >>> agnostic_diagonal_concat(df_pd_1, df_pd_2)
           a    b    z
        0  1  3.5  NaN
        1  2  4.5  NaN
        0  3  NaN    x
        1  4  NaN    y

        >>> agnostic_diagonal_concat(df_pl_1, df_pl_2)
        shape: (4, 3)
        ┌─────┬──────┬──────┐
        │ a   ┆ b    ┆ z    │
        │ --- ┆ ---  ┆ ---  │
        │ i64 ┆ f64  ┆ str  │
        ╞═════╪══════╪══════╡
        │ 1   ┆ 3.5  ┆ null │
        │ 2   ┆ 4.5  ┆ null │
        │ 3   ┆ null ┆ x    │
        │ 4   ┆ null ┆ y    │
        └─────┴──────┴──────┘
    """,
    "new_series": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT, IntoSeriesT
        >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}

        Let's define a dataframe-agnostic function:

        >>> def agnostic_new_series(df_native: IntoFrameT) -> IntoSeriesT:
        ...     values = [4, 1, 2, 3]
        ...     native_namespace = nw.get_native_namespace(df_native)
        ...     return nw.new_series(
        ...         name="a",
        ...         values=values,
        ...         dtype=nw.Int32,
        ...         native_namespace=native_namespace,
        ...     ).to_native()

        We can then pass any supported eager library, such as pandas / Polars / PyArrow:

        >>> agnostic_new_series(pd.DataFrame(data))
        0    4
        1    1
        2    2
        3    3
        Name: a, dtype: int32
        >>> agnostic_new_series(pl.DataFrame(data))  # doctest: +NORMALIZE_WHITESPACE
        shape: (4,)
        Series: 'a' [i32]
        [
           4
           1
           2
           3
        ]
        >>> agnostic_new_series(pa.table(data))
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            4,
            1,
            2,
            3
          ]
        ]
    """,
    "from_dict": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}

        Let's create a new dataframe of the same class as the dataframe we started with, from a dict of new data:

        >>> def agnostic_from_dict(df_native: IntoFrameT) -> IntoFrameT:
        ...     new_data = {"c": [5, 2], "d": [1, 4]}
        ...     native_namespace = nw.get_native_namespace(df_native)
        ...     return nw.from_dict(new_data, native_namespace=native_namespace).to_native()

        Let's see what happens when passing pandas, Polars or PyArrow input:

        >>> agnostic_from_dict(pd.DataFrame(data))
           c  d
        0  5  1
        1  2  4
        >>> agnostic_from_dict(pl.DataFrame(data))
        shape: (2, 2)
        ┌─────┬─────┐
        │ c   ┆ d   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 5   ┆ 1   │
        │ 2   ┆ 4   │
        └─────┴─────┘
        >>> agnostic_from_dict(pa.table(data))
        pyarrow.Table
        c: int64
        d: int64
        ----
        c: [[5,2]]
        d: [[1,4]]
    """,
    "from_numpy": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import numpy as np
        >>> from narwhals.typing import IntoFrameT
        >>> data = {"a": [1, 2], "b": [3, 4]}

        Let's create a new dataframe of the same class as the dataframe we started with, from a NumPy ndarray of new data:

        >>> def agnostic_from_numpy(df_native: IntoFrameT) -> IntoFrameT:
        ...     new_data = np.array([[5, 2, 1], [1, 4, 3]])
        ...     df = nw.from_native(df_native)
        ...     native_namespace = nw.get_native_namespace(df)
        ...     return nw.from_numpy(new_data, native_namespace=native_namespace).to_native()

        Let's see what happens when passing pandas, Polars or PyArrow input:

        >>> agnostic_from_numpy(pd.DataFrame(data))
           column_0  column_1  column_2
        0         5         2         1
        1         1         4         3
        >>> agnostic_from_numpy(pl.DataFrame(data))
        shape: (2, 3)
        ┌──────────┬──────────┬──────────┐
        │ column_0 ┆ column_1 ┆ column_2 │
        │ ---      ┆ ---      ┆ ---      │
        │ i64      ┆ i64      ┆ i64      │
        ╞══════════╪══════════╪══════════╡
        │ 5        ┆ 2        ┆ 1        │
        │ 1        ┆ 4        ┆ 3        │
        └──────────┴──────────┴──────────┘
        >>> agnostic_from_numpy(pa.table(data))
        pyarrow.Table
        column_0: int64
        column_1: int64
        column_2: int64
        ----
        column_0: [[5,1]]
        column_1: [[2,4]]
        column_2: [[1,3]]

        Let's specify the column names:

        >>> def agnostic_from_numpy(df_native: IntoFrameT) -> IntoFrameT:
        ...     new_data = np.array([[5, 2, 1], [1, 4, 3]])
        ...     schema = ["c", "d", "e"]
        ...     df = nw.from_native(df_native)
        ...     native_namespace = nw.get_native_namespace(df)
        ...     return nw.from_numpy(
        ...         new_data, native_namespace=native_namespace, schema=schema
        ...     ).to_native()

        Let's see the modified outputs:

        >>> agnostic_from_numpy(pd.DataFrame(data))
           c  d  e
        0  5  2  1
        1  1  4  3
        >>> agnostic_from_numpy(pl.DataFrame(data))
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ c   ┆ d   ┆ e   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 5   ┆ 2   ┆ 1   │
        │ 1   ┆ 4   ┆ 3   │
        └─────┴─────┴─────┘
        >>> agnostic_from_numpy(pa.table(data))
        pyarrow.Table
        c: int64
        d: int64
        e: int64
        ----
        c: [[5,1]]
        d: [[2,4]]
        e: [[1,3]]

        Let's modify the function so that it specifies the schema:

        >>> def agnostic_from_numpy(df_native: IntoFrameT) -> IntoFrameT:
        ...     new_data = np.array([[5, 2, 1], [1, 4, 3]])
        ...     schema = {"c": nw.Int16(), "d": nw.Float32(), "e": nw.Int8()}
        ...     df = nw.from_native(df_native)
        ...     native_namespace = nw.get_native_namespace(df)
        ...     return nw.from_numpy(
        ...         new_data, native_namespace=native_namespace, schema=schema
        ...     ).to_native()

        Let's see the outputs:

        >>> agnostic_from_numpy(pd.DataFrame(data))
           c    d  e
        0  5  2.0  1
        1  1  4.0  3
        >>> agnostic_from_numpy(pl.DataFrame(data))
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ c   ┆ d   ┆ e   │
        │ --- ┆ --- ┆ --- │
        │ i16 ┆ f32 ┆ i8  │
        ╞═════╪═════╪═════╡
        │ 5   ┆ 2.0 ┆ 1   │
        │ 1   ┆ 4.0 ┆ 3   │
        └─────┴─────┴─────┘
        >>> agnostic_from_numpy(pa.table(data))
        pyarrow.Table
        c: int16
        d: float
        e: int8
        ----
        c: [[5,1]]
        d: [[2,4]]
        e: [[1,3]]
    """,
    "from_arrow": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}

        Let's define a dataframe-agnostic function which creates a PyArrow
        Table.

        >>> def agnostic_to_arrow(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return nw.from_arrow(df, native_namespace=pa).to_native()

        Let's see what happens when passing pandas / Polars input:

        >>> agnostic_to_arrow(pd.DataFrame(data))
        pyarrow.Table
        a: int64
        b: int64
        ----
        a: [[1,2,3]]
        b: [[4,5,6]]
        >>> agnostic_to_arrow(pl.DataFrame(data))
        pyarrow.Table
        a: int64
        b: int64
        ----
        a: [[1,2,3]]
        b: [[4,5,6]]
    """,
    "show_versions": """
        >>> from narwhals import show_versions
        >>> show_versions()  # doctest: +SKIP
    """,
    "read_csv": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoDataFrame
        >>> from types import ModuleType

        Let's create an agnostic function that reads a csv file with a specified native namespace:

        >>> def agnostic_read_csv(native_namespace: ModuleType) -> IntoDataFrame:
        ...     return nw.read_csv("file.csv", native_namespace=native_namespace).to_native()

        Then we can read the file by passing pandas, Polars or PyArrow namespaces:

        >>> agnostic_read_csv(native_namespace=pd)  # doctest:+SKIP
           a  b
        0  1  4
        1  2  5
        2  3  6
        >>> agnostic_read_csv(native_namespace=pl)  # doctest:+SKIP
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        >>> agnostic_read_csv(native_namespace=pa)  # doctest:+SKIP
        pyarrow.Table
        a: int64
        b: int64
        ----
        a: [[1,2,3]]
        b: [[4,5,6]]
    """,
    "scan_csv": """
        >>> import dask.dataframe as dd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrame
        >>> from types import ModuleType

        Let's create an agnostic function that lazily reads a csv file with a specified native namespace:

        >>> def agnostic_scan_csv(native_namespace: ModuleType) -> IntoFrame:
        ...     return nw.scan_csv("file.csv", native_namespace=native_namespace).to_native()

        Then we can read the file by passing, for example, Polars or Dask namespaces:

        >>> agnostic_scan_csv(native_namespace=pl).collect()  # doctest:+SKIP
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        >>> agnostic_scan_csv(native_namespace=dd).compute()  # doctest:+SKIP
           a  b
        0  1  4
        1  2  5
        2  3  6
    """,
    "read_parquet": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoDataFrame
        >>> from types import ModuleType

        Let's create an agnostic function that reads a parquet file with a specified native namespace:

        >>> def agnostic_read_parquet(native_namespace: ModuleType) -> IntoDataFrame:
        ...     return nw.read_parquet(
        ...         "file.parquet", native_namespace=native_namespace
        ...     ).to_native()

        Then we can read the file by passing pandas, Polars or PyArrow namespaces:

        >>> agnostic_read_parquet(native_namespace=pd)  # doctest:+SKIP
           a  b
        0  1  4
        1  2  5
        2  3  6
        >>> agnostic_read_parquet(native_namespace=pl)  # doctest:+SKIP
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        >>> agnostic_read_parquet(native_namespace=pa)  # doctest:+SKIP
        pyarrow.Table
        a: int64
        b: int64
        ----
        a: [[1,2,3]]
        b: [[4,5,6]]
    """,
    "scan_parquet": """
        >>> import dask.dataframe as dd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrame
        >>> from types import ModuleType

        Let's create an agnostic function that lazily reads a parquet file with a specified native namespace:

        >>> def agnostic_scan_parquet(native_namespace: ModuleType) -> IntoFrame:
        ...     return nw.scan_parquet(
        ...         "file.parquet", native_namespace=native_namespace
        ...     ).to_native()

        Then we can read the file by passing, for example, Polars or Dask namespaces:

        >>> agnostic_scan_parquet(native_namespace=pl).collect()  # doctest:+SKIP
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        >>> agnostic_scan_parquet(native_namespace=dd).compute()  # doctest:+SKIP
           a  b
        0  1  4
        1  2  5
        2  3  6
    """,
    "col": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)
        >>> df_pa = pa.table(data)

        We define a dataframe-agnostic function:

        >>> def agnostic_col(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.col("a") * nw.col("b")).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_col`:

        >>> agnostic_col(df_pd)
           a
        0  3
        1  8

        >>> agnostic_col(df_pl)
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 3   │
        │ 8   │
        └─────┘

        >>> agnostic_col(df_pa)
        pyarrow.Table
        a: int64
        ----
        a: [[3,8]]
    """,
    "nth": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)
        >>> df_pa = pa.table(data)

        We define a dataframe-agnostic function:

        >>> def agnostic_nth(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.nth(0) * 2).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to `agnostic_nth`:

        >>> agnostic_nth(df_pd)
           a
        0  2
        1  4

        >>> agnostic_nth(df_pl)
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        │ 4   │
        └─────┘

        >>> agnostic_nth(df_pa)
        pyarrow.Table
        a: int64
        ----
        a: [[2,4]]
    """,
    "all_": """
        >>> import polars as pl
        >>> import pandas as pd
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function:

        >>> def agnostic_all(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.all() * 2).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_all`:

        >>> agnostic_all(df_pd)
           a   b
        0  2   8
        1  4  10
        2  6  12

        >>> agnostic_all(df_pl)
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

        >>> agnostic_all(df_pa)
        pyarrow.Table
        a: int64
        b: int64
        ----
        a: [[2,4,6]]
        b: [[8,10,12]]
    """,
    "len_": """
        >>> import polars as pl
        >>> import pandas as pd
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": [5, 10]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function:

        >>> def agnostic_len(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.len()).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_len`:

        >>> agnostic_len(df_pd)
           len
        0    2
        >>> agnostic_len(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ len │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 2   │
        └─────┘
        >>> agnostic_len(df_pa)
        pyarrow.Table
        len: int64
        ----
        len: [[2]]
    """,
    "sum": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2]}
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)
        >>> df_pa = pa.table(data)

        We define a dataframe-agnostic function:

        >>> def agnostic_sum(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.sum("a")).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_sum`:

        >>> agnostic_sum(df_pd)
           a
        0  3

        >>> agnostic_sum(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 3   │
        └─────┘

        >>> agnostic_sum(df_pa)
        pyarrow.Table
        a: int64
        ----
        a: [[3]]
    """,
    "mean": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 8, 3]}
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)
        >>> df_pa = pa.table(data)

        We define a dataframe agnostic function:

        >>> def agnostic_mean(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.mean("a")).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_mean`:

        >>> agnostic_mean(df_pd)
             a
        0  4.0

        >>> agnostic_mean(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 4.0 │
        └─────┘

        >>> agnostic_mean(df_pa)
        pyarrow.Table
        a: double
        ----
        a: [[4]]
    """,
    "median": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [4, 5, 2]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe agnostic function:

        >>> def agnostic_median(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.median("a")).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_median`:

        >>> agnostic_median(df_pd)
             a
        0  4.0

        >>> agnostic_median(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 4.0 │
        └─────┘

        >>> agnostic_median(df_pa)
        pyarrow.Table
        a: double
        ----
        a: [[4]]
    """,
    "min": """
        >>> import polars as pl
        >>> import pandas as pd
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": [5, 10]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function:

        >>> def agnostic_min(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.min("b")).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_min`:

        >>> agnostic_min(df_pd)
           b
        0  5

        >>> agnostic_min(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 5   │
        └─────┘

        >>> agnostic_min(df_pa)
        pyarrow.Table
        b: int64
        ----
        b: [[5]]
    """,
    "max": """
        >>> import polars as pl
        >>> import pandas as pd
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": [5, 10]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function:

        >>> def agnostic_max(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.max("a")).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_max`:

        >>> agnostic_max(df_pd)
           a
        0  2

        >>> agnostic_max(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        └─────┘

        >>> agnostic_max(df_pa)
        pyarrow.Table
        a: int64
        ----
        a: [[2]]
    """,
    "sum_horizontal": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2, 3], "b": [5, 10, None]}
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)
        >>> df_pa = pa.table(data)

        We define a dataframe-agnostic function:

        >>> def agnostic_sum_horizontal(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.sum_horizontal("a", "b")).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to `agnostic_sum_horizontal`:

        >>> agnostic_sum_horizontal(df_pd)
              a
        0   6.0
        1  12.0
        2   3.0

        >>> agnostic_sum_horizontal(df_pl)
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 6   │
        │ 12  │
        │ 3   │
        └─────┘

        >>> agnostic_sum_horizontal(df_pa)
        pyarrow.Table
        a: int64
        ----
        a: [[6,12,3]]
    """,
    "min_horizontal": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {
        ...     "a": [1, 8, 3],
        ...     "b": [4, 5, None],
        ...     "c": ["x", "y", "z"],
        ... }

        We define a dataframe-agnostic function that computes the horizontal min of "a"
        and "b" columns:

        >>> def agnostic_min_horizontal(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.min_horizontal("a", "b")).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_min_horizontal`:

        >>> agnostic_min_horizontal(pd.DataFrame(data))
             a
        0  1.0
        1  5.0
        2  3.0

        >>> agnostic_min_horizontal(pl.DataFrame(data))
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 5   │
        │ 3   │
        └─────┘

        >>> agnostic_min_horizontal(pa.table(data))
        pyarrow.Table
        a: int64
        ----
        a: [[1,5,3]]
    """,
    "max_horizontal": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {
        ...     "a": [1, 8, 3],
        ...     "b": [4, 5, None],
        ...     "c": ["x", "y", "z"],
        ... }

        We define a dataframe-agnostic function that computes the horizontal max of "a"
        and "b" columns:

        >>> def agnostic_max_horizontal(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.max_horizontal("a", "b")).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_max_horizontal`:

        >>> agnostic_max_horizontal(pd.DataFrame(data))
             a
        0  4.0
        1  8.0
        2  3.0

        >>> agnostic_max_horizontal(pl.DataFrame(data))
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 4   │
        │ 8   │
        │ 3   │
        └─────┘

        >>> agnostic_max_horizontal(pa.table(data))
        pyarrow.Table
        a: int64
        ----
        a: [[4,8,3]]
    """,
    "when": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2, 3], "b": [5, 10, 15]}
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)
        >>> df_pa = pa.table(data)

        We define a dataframe-agnostic function:

        >>> def agnostic_when_then_otherwise(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.with_columns(
        ...         nw.when(nw.col("a") < 3).then(5).otherwise(6).alias("a_when")
        ...     ).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_when_then_otherwise`:

        >>> agnostic_when_then_otherwise(df_pd)
           a   b  a_when
        0  1   5       5
        1  2  10       5
        2  3  15       6

        >>> agnostic_when_then_otherwise(df_pl)
        shape: (3, 3)
        ┌─────┬─────┬────────┐
        │ a   ┆ b   ┆ a_when │
        │ --- ┆ --- ┆ ---    │
        │ i64 ┆ i64 ┆ i32    │
        ╞═════╪═════╪════════╡
        │ 1   ┆ 5   ┆ 5      │
        │ 2   ┆ 10  ┆ 5      │
        │ 3   ┆ 15  ┆ 6      │
        └─────┴─────┴────────┘

        >>> agnostic_when_then_otherwise(df_pa)
        pyarrow.Table
        a: int64
        b: int64
        a_when: int64
        ----
        a: [[1,2,3]]
        b: [[5,10,15]]
        a_when: [[5,5,6]]
    """,
    "all_horizontal": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {
        ...     "a": [False, False, True, True, False, None],
        ...     "b": [False, True, True, None, None, None],
        ... }
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")
        >>> df_pa = pa.table(data)

        We define a dataframe-agnostic function:

        >>> def agnostic_all_horizontal(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select("a", "b", all=nw.all_horizontal("a", "b")).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_all_horizontal`:

        >>> agnostic_all_horizontal(df_pd)
               a      b    all
        0  False  False  False
        1  False   True  False
        2   True   True   True
        3   True   <NA>   <NA>
        4  False   <NA>  False
        5   <NA>   <NA>   <NA>

        >>> agnostic_all_horizontal(df_pl)
        shape: (6, 3)
        ┌───────┬───────┬───────┐
        │ a     ┆ b     ┆ all   │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ false ┆ false ┆ false │
        │ false ┆ true  ┆ false │
        │ true  ┆ true  ┆ true  │
        │ true  ┆ null  ┆ null  │
        │ false ┆ null  ┆ false │
        │ null  ┆ null  ┆ null  │
        └───────┴───────┴───────┘

        >>> agnostic_all_horizontal(df_pa)
        pyarrow.Table
        a: bool
        b: bool
        all: bool
        ----
        a: [[false,false,true,true,false,null]]
        b: [[false,true,true,null,null,null]]
        all: [[false,false,true,null,false,null]]
    """,
    "lit": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2]}
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)
        >>> df_pa = pa.table(data)

        We define a dataframe-agnostic function:

        >>> def agnostic_lit(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.with_columns(nw.lit(3)).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_lit`:

        >>> agnostic_lit(df_pd)
           a  literal
        0  1        3
        1  2        3

        >>> agnostic_lit(df_pl)
        shape: (2, 2)
        ┌─────┬─────────┐
        │ a   ┆ literal │
        │ --- ┆ ---     │
        │ i64 ┆ i32     │
        ╞═════╪═════════╡
        │ 1   ┆ 3       │
        │ 2   ┆ 3       │
        └─────┴─────────┘

        >>> agnostic_lit(df_pa)
        pyarrow.Table
        a: int64
        literal: int64
        ----
        a: [[1,2]]
        literal: [[3,3]]
    """,
    "any_horizontal": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {
        ...     "a": [False, False, True, True, False, None],
        ...     "b": [False, True, True, None, None, None],
        ... }
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")
        >>> df_pa = pa.table(data)

        We define a dataframe-agnostic function:

        >>> def agnostic_any_horizontal(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select("a", "b", any=nw.any_horizontal("a", "b")).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_any_horizontal`:

        >>> agnostic_any_horizontal(df_pd)
               a      b    any
        0  False  False  False
        1  False   True   True
        2   True   True   True
        3   True   <NA>   True
        4  False   <NA>   <NA>
        5   <NA>   <NA>   <NA>

        >>> agnostic_any_horizontal(df_pl)
        shape: (6, 3)
        ┌───────┬───────┬───────┐
        │ a     ┆ b     ┆ any   │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ false ┆ false ┆ false │
        │ false ┆ true  ┆ true  │
        │ true  ┆ true  ┆ true  │
        │ true  ┆ null  ┆ true  │
        │ false ┆ null  ┆ null  │
        │ null  ┆ null  ┆ null  │
        └───────┴───────┴───────┘

        >>> agnostic_any_horizontal(df_pa)
        pyarrow.Table
        a: bool
        b: bool
        any: bool
        ----
        a: [[false,false,true,true,false,null]]
        b: [[false,true,true,null,null,null]]
        any: [[false,true,true,true,null,null]]
    """,
    "mean_horizontal": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {
        ...     "a": [1, 8, 3],
        ...     "b": [4, 5, None],
        ...     "c": ["x", "y", "z"],
        ... }
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)
        >>> df_pa = pa.table(data)

        We define a dataframe-agnostic function that computes the horizontal mean of "a"
        and "b" columns:

        >>> def agnostic_mean_horizontal(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(nw.mean_horizontal("a", "b")).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow to
        `agnostic_mean_horizontal`:

        >>> agnostic_mean_horizontal(df_pd)
             a
        0  2.5
        1  6.5
        2  3.0

        >>> agnostic_mean_horizontal(df_pl)
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 2.5 │
        │ 6.5 │
        │ 3.0 │
        └─────┘

        >>> agnostic_mean_horizontal(df_pa)
        pyarrow.Table
        a: double
        ----
        a: [[2.5,6.5,3]]
    """,
    "concat_str": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {
        ...     "a": [1, 2, 3],
        ...     "b": ["dogs", "cats", None],
        ...     "c": ["play", "swim", "walk"],
        ... }

        We define a dataframe-agnostic function that computes the horizontal string
        concatenation of different columns

        >>> def agnostic_concat_str(df_native: IntoFrameT) -> IntoFrameT:
        ...     df = nw.from_native(df_native)
        ...     return df.select(
        ...         nw.concat_str(
        ...             [
        ...                 nw.col("a") * 2,
        ...                 nw.col("b"),
        ...                 nw.col("c"),
        ...             ],
        ...             separator=" ",
        ...         ).alias("full_sentence")
        ...     ).to_native()

        We can pass any supported library such as Pandas, Polars, or PyArrow
        to `agnostic_concat_str`:

        >>> agnostic_concat_str(pd.DataFrame(data))
          full_sentence
        0   2 dogs play
        1   4 cats swim
        2          None

        >>> agnostic_concat_str(pl.DataFrame(data))
        shape: (3, 1)
        ┌───────────────┐
        │ full_sentence │
        │ ---           │
        │ str           │
        ╞═══════════════╡
        │ 2 dogs play   │
        │ 4 cats swim   │
        │ null          │
        └───────────────┘

        >>> agnostic_concat_str(pa.table(data))
        pyarrow.Table
        full_sentence: string
        ----
        full_sentence: [["2 dogs play","4 cats swim",null]]
    """,
}
