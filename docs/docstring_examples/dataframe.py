from __future__ import annotations

EXAMPLES = {
    "implementation": """
            >>> import narwhals as nw
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> lf_pl = pl.LazyFrame({"a": [1, 2, 3]})
            >>> lf_dask = dd.from_dict({"a": [1, 2, 3]}, npartitions=2)

            >>> lf = nw.from_native(lf_pl)
            >>> lf.implementation
            <Implementation.POLARS: 6>
            >>> lf.implementation.is_pandas()
            False
            >>> lf.implementation.is_polars()
            True

            >>> lf = nw.from_native(lf_dask)
            >>> lf.implementation
            <Implementation.DASK: 7>
            >>> lf.implementation.is_dask()
            True
        """,
    "lazy": """
            Construct pandas and Polars objects:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> df = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> lf_pl = pl.LazyFrame(df)

            We define a library agnostic function:

            >>> def agnostic_lazy(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.lazy().to_native()

            Note that then, pandas dataframe stay eager, and the Polars LazyFrame stays lazy:

            >>> agnostic_lazy(df_pd)
               foo  bar ham
            0    1  6.0   a
            1    2  7.0   b
            2    3  8.0   c
            >>> agnostic_lazy(lf_pl)
            <LazyFrame ...>
        """,
    "to_native": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>>
            >>> data = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            Calling `to_native` on a Narwhals LazyFrame returns the native object:

            >>> nw.from_native(lf_pl).to_native().collect()
            shape: (3, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ f64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6.0 ┆ a   │
            │ 2   ┆ 7.0 ┆ b   │
            │ 3   ┆ 8.0 ┆ c   │
            └─────┴─────┴─────┘
            >>> nw.from_native(lf_dask).to_native().compute()
               foo  bar ham
            0    1  6.0   a
            1    2  7.0   b
            2    3  8.0   c
        """,
    "to_pandas": """
            Construct pandas, Polars (eager) and PyArrow DataFrames:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> data = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_to_pandas(df_native: IntoDataFrame) -> pd.DataFrame:
            ...     df = nw.from_native(df_native)
            ...     return df.to_pandas()

            We can then pass any supported library such as pandas, Polars (eager), or
            PyArrow to `agnostic_to_pandas`:

            >>> agnostic_to_pandas(df_pd)
               foo  bar ham
            0    1  6.0   a
            1    2  7.0   b
            2    3  8.0   c
            >>> agnostic_to_pandas(df_pl)
               foo  bar ham
            0    1  6.0   a
            1    2  7.0   b
            2    3  8.0   c
            >>> agnostic_to_pandas(df_pa)
               foo  bar ham
            0    1  6.0   a
            1    2  7.0   b
            2    3  8.0   c
        """,
    "write_csv": """
            Construct pandas, Polars (eager) and PyArrow DataFrames:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> data = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_write_csv(df_native: IntoDataFrame) -> str:
            ...     df = nw.from_native(df_native)
            ...     return df.write_csv()

            We can pass any supported library such as pandas, Polars or PyArrow to `agnostic_write_csv`:

            >>> agnostic_write_csv(df_pd)
            'foo,bar,ham\\n1,6.0,a\\n2,7.0,b\\n3,8.0,c\\n'
            >>> agnostic_write_csv(df_pl)
            'foo,bar,ham\\n1,6.0,a\\n2,7.0,b\\n3,8.0,c\\n'
            >>> agnostic_write_csv(df_pa)
            '"foo","bar","ham"\\n1,6,"a"\\n2,7,"b"\\n3,8,"c"\\n'

            If we had passed a file name to `write_csv`, it would have been
            written to that file.
        """,
    "write_parquet": """
            Construct pandas, Polars and PyArrow DataFrames:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> data = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_write_parquet(df_native: IntoDataFrame):
            ...     df = nw.from_native(df_native)
            ...     df.write_parquet("foo.parquet")

            We can then pass either pandas, Polars or PyArrow to `agnostic_write_parquet`:

            >>> agnostic_write_parquet(df_pd)  # doctest:+SKIP
            >>> agnostic_write_parquet(df_pl)  # doctest:+SKIP
            >>> agnostic_write_parquet(df_pa)  # doctest:+SKIP
        """,
    "to_numpy": """
            Construct pandas and polars DataFrames:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> import numpy as np
            >>> from narwhals.typing import IntoDataFrame
            >>> data = {"foo": [1, 2, 3], "bar": [6.5, 7.0, 8.5], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_to_numpy(df_native: IntoDataFrame) -> np.ndarray:
            ...     df = nw.from_native(df_native)
            ...     return df.to_numpy()

            We can then pass either pandas, Polars or PyArrow to `agnostic_to_numpy`:

            >>> agnostic_to_numpy(df_pd)
            array([[1, 6.5, 'a'],
                   [2, 7.0, 'b'],
                   [3, 8.5, 'c']], dtype=object)
            >>> agnostic_to_numpy(df_pl)
            array([[1, 6.5, 'a'],
                   [2, 7.0, 'b'],
                   [3, 8.5, 'c']], dtype=object)
            >>> agnostic_to_numpy(df_pa)
            array([[1, 6.5, 'a'],
                   [2, 7.0, 'b'],
                   [3, 8.5, 'c']], dtype=object)
        """,
    "shape": """
            Construct pandas and polars DataFrames:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> data = {"foo": [1, 2, 3, 4, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_shape(df_native: IntoDataFrame) -> tuple[int, int]:
            ...     df = nw.from_native(df_native)
            ...     return df.shape

            We can then pass either pandas, Polars or PyArrow to `agnostic_shape`:

            >>> agnostic_shape(df_pd)
            (5, 1)
            >>> agnostic_shape(df_pl)
            (5, 1)
            >>> agnostic_shape(df_pa)
            (5, 1)
        """,
    "get_column": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> from narwhals.typing import IntoSeries
            >>> data = {"a": [1, 2], "b": [3, 4]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_get_column(df_native: IntoDataFrame) -> IntoSeries:
            ...     df = nw.from_native(df_native)
            ...     name = df.columns[0]
            ...     return df.get_column(name).to_native()

            We can then pass either pandas, Polars or PyArrow to `agnostic_get_column`:

            >>> agnostic_get_column(df_pd)
            0    1
            1    2
            Name: a, dtype: int64
            >>> agnostic_get_column(df_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (2,)
            Series: 'a' [i64]
            [
                1
                2
            ]
            >>> agnostic_get_column(df_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                2
              ]
            ]
        """,
    "estimated_size": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrameT
            >>> data = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6.0, 7.0, 8.0],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_estimated_size(df_native: IntoDataFrameT) -> int | float:
            ...     df = nw.from_native(df_native)
            ...     return df.estimated_size()

            We can then pass either pandas, Polars or PyArrow to `agnostic_estimated_size`:

            >>> agnostic_estimated_size(df_pd)
            np.int64(330)
            >>> agnostic_estimated_size(df_pl)
            51
            >>> agnostic_estimated_size(df_pa)
            63
        """,
    "__getitem__": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> from narwhals.typing import IntoSeries
            >>> data = {"a": [1, 2], "b": [3, 4]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_slice(df_native: IntoDataFrame) -> IntoSeries:
            ...     df = nw.from_native(df_native)
            ...     return df["a"].to_native()

            We can then pass either pandas, Polars or PyArrow to `agnostic_slice`:

            >>> agnostic_slice(df_pd)
            0    1
            1    2
            Name: a, dtype: int64
            >>> agnostic_slice(df_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (2,)
            Series: 'a' [i64]
            [
                1
                2
            ]
            >>> agnostic_slice(df_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                2
              ]
            ]
        """,
    "to_dict": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> data = {
            ...     "A": [1, 2, 3, 4, 5],
            ...     "fruits": ["banana", "banana", "apple", "apple", "banana"],
            ...     "B": [5, 4, 3, 2, 1],
            ...     "animals": ["beetle", "fly", "beetle", "beetle", "beetle"],
            ...     "optional": [28, 300, None, 2, -30],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_to_dict(
            ...     df_native: IntoDataFrame,
            ... ) -> dict[str, list[int | str | float | None]]:
            ...     df = nw.from_native(df_native)
            ...     return df.to_dict(as_series=False)

            We can then pass either pandas, Polars or PyArrow to `agnostic_to_dict`:

            >>> agnostic_to_dict(df_pd)
            {'A': [1, 2, 3, 4, 5], 'fruits': ['banana', 'banana', 'apple', 'apple', 'banana'], 'B': [5, 4, 3, 2, 1], 'animals': ['beetle', 'fly', 'beetle', 'beetle', 'beetle'], 'optional': [28.0, 300.0, nan, 2.0, -30.0]}
            >>> agnostic_to_dict(df_pl)
            {'A': [1, 2, 3, 4, 5], 'fruits': ['banana', 'banana', 'apple', 'apple', 'banana'], 'B': [5, 4, 3, 2, 1], 'animals': ['beetle', 'fly', 'beetle', 'beetle', 'beetle'], 'optional': [28, 300, None, 2, -30]}
            >>> agnostic_to_dict(df_pa)
            {'A': [1, 2, 3, 4, 5], 'fruits': ['banana', 'banana', 'apple', 'apple', 'banana'], 'B': [5, 4, 3, 2, 1], 'animals': ['beetle', 'fly', 'beetle', 'beetle', 'beetle'], 'optional': [28, 300, None, 2, -30]}
        """,
    "row": """
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> from narwhals.typing import IntoDataFrame
            >>> from typing import Any
            >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a library-agnostic function to get the second row.

            >>> def agnostic_row(df_native: IntoDataFrame) -> tuple[Any, ...]:
            ...     return nw.from_native(df_native).row(1)

            We can then pass either pandas, Polars or PyArrow to `agnostic_row`:

            >>> agnostic_row(df_pd)
            (2, 5)
            >>> agnostic_row(df_pl)
            (2, 5)
            >>> agnostic_row(df_pa)
            (<pyarrow.Int64Scalar: 2>, <pyarrow.Int64Scalar: 5>)
        """,
    "pipe": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3], "ba": [4, 5, 6]}
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_pipe(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.pipe(lambda _df: _df.select("a")).collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_pipe`:

            >>> agnostic_pipe(lf_pl)
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
            >>> agnostic_pipe(lf_dask)
               a
            0  1
            1  2
            2  3
        """,
    "drop_nulls": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1.0, 2.0, None], "ba": [1.0, None, 2.0]}
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_drop_nulls(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.drop_nulls().collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_drop_nulls`:

            >>> agnostic_drop_nulls(lf_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ ba  │
            │ --- ┆ --- │
            │ f64 ┆ f64 │
            ╞═════╪═════╡
            │ 1.0 ┆ 1.0 │
            └─────┴─────┘
            >>> agnostic_drop_nulls(lf_dask)
                 a   ba
            0  1.0  1.0
        """,
    "with_row_index": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_with_row_index(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_row_index().collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_with_row_index`:

            >>> agnostic_with_row_index(lf_pl)
            shape: (3, 3)
            ┌───────┬─────┬─────┐
            │ index ┆ a   ┆ b   │
            │ ---   ┆ --- ┆ --- │
            │ u32   ┆ i64 ┆ i64 │
            ╞═══════╪═════╪═════╡
            │ 0     ┆ 1   ┆ 4   │
            │ 1     ┆ 2   ┆ 5   │
            │ 2     ┆ 3   ┆ 6   │
            └───────┴─────┴─────┘
            >>> agnostic_with_row_index(lf_dask)
               index  a  b
            0      0  1  4
            1      1  2  5
            2      2  3  6
        """,
    "schema": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> data = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6.0, 7.0, 8.0],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            >>> lf = nw.from_native(lf_pl)
            >>> lf.schema  # doctest: +SKIP
            Schema({'foo': Int64, 'bar': Float64, 'ham': String})

            >>> lf = nw.from_native(lf_dask)
            >>> lf.schema  # doctest: +SKIP
            Schema({'foo': Int64, 'bar': Float64, 'ham': String})
        """,
    "collect_schema": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> data = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6.0, 7.0, 8.0],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            >>> lf = nw.from_native(lf_pl)
            >>> lf.collect_schema()
            Schema({'foo': Int64, 'bar': Float64, 'ham': String})

            >>> lf = nw.from_native(lf_dask)
            >>> lf.collect_schema()
            Schema({'foo': Int64, 'bar': Float64, 'ham': String})
        """,
    "columns": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrame
            >>>
            >>> data = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            We define a library agnostic function:

            >>> def agnostic_columns(df_native: IntoFrame) -> list[str]:
            ...     df = nw.from_native(df_native)
            ...     return df.columns

            We can then pass any supported library such as Polars or Dask to `agnostic_columns`:

            >>> agnostic_columns(lf_pl)  # doctest: +SKIP
            ['foo', 'bar', 'ham']
            >>> agnostic_columns(lf_dask)
            ['foo', 'bar', 'ham']
        """,
    "rows": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> data = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_rows(df_native: IntoDataFrame, *, named: bool):
            ...     return nw.from_native(df_native, eager_only=True).rows(named=named)

            We can then pass any supported library such as Pandas, Polars, or PyArrow
            to `agnostic_rows`:

            >>> agnostic_rows(df_pd, named=False)
            [(1, 6.0, 'a'), (2, 7.0, 'b'), (3, 8.0, 'c')]
            >>> agnostic_rows(df_pd, named=True)
            [{'foo': 1, 'bar': 6.0, 'ham': 'a'}, {'foo': 2, 'bar': 7.0, 'ham': 'b'}, {'foo': 3, 'bar': 8.0, 'ham': 'c'}]
            >>> agnostic_rows(df_pl, named=False)
            [(1, 6.0, 'a'), (2, 7.0, 'b'), (3, 8.0, 'c')]
            >>> agnostic_rows(df_pl, named=True)
            [{'foo': 1, 'bar': 6.0, 'ham': 'a'}, {'foo': 2, 'bar': 7.0, 'ham': 'b'}, {'foo': 3, 'bar': 8.0, 'ham': 'c'}]
            >>> agnostic_rows(df_pa, named=False)
            [(1, 6.0, 'a'), (2, 7.0, 'b'), (3, 8.0, 'c')]
            >>> agnostic_rows(df_pa, named=True)
            [{'foo': 1, 'bar': 6.0, 'ham': 'a'}, {'foo': 2, 'bar': 7.0, 'ham': 'b'}, {'foo': 3, 'bar': 8.0, 'ham': 'c'}]
        """,
    "iter_rows": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> data = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_iter_rows(df_native: IntoDataFrame, *, named: bool):
            ...     return nw.from_native(df_native, eager_only=True).iter_rows(named=named)

            We can then pass any supported library such as Pandas, Polars, or PyArrow
            to `agnostic_iter_rows`:

            >>> [row for row in agnostic_iter_rows(df_pd, named=False)]
            [(1, 6.0, 'a'), (2, 7.0, 'b'), (3, 8.0, 'c')]
            >>> [row for row in agnostic_iter_rows(df_pd, named=True)]
            [{'foo': 1, 'bar': 6.0, 'ham': 'a'}, {'foo': 2, 'bar': 7.0, 'ham': 'b'}, {'foo': 3, 'bar': 8.0, 'ham': 'c'}]
            >>> [row for row in agnostic_iter_rows(df_pl, named=False)]
            [(1, 6.0, 'a'), (2, 7.0, 'b'), (3, 8.0, 'c')]
            >>> [row for row in agnostic_iter_rows(df_pl, named=True)]
            [{'foo': 1, 'bar': 6.0, 'ham': 'a'}, {'foo': 2, 'bar': 7.0, 'ham': 'b'}, {'foo': 3, 'bar': 8.0, 'ham': 'c'}]
            >>> [row for row in agnostic_iter_rows(df_pa, named=False)]
            [(1, 6.0, 'a'), (2, 7.0, 'b'), (3, 8.0, 'c')]
            >>> [row for row in agnostic_iter_rows(df_pa, named=True)]
            [{'foo': 1, 'bar': 6.0, 'ham': 'a'}, {'foo': 2, 'bar': 7.0, 'ham': 'b'}, {'foo': 3, 'bar': 8.0, 'ham': 'c'}]
        """,
    "with_columns": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": [1, 2, 3, 4],
            ...     "b": [0.5, 4, 10, 13],
            ...     "c": [True, True, False, True],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            Let's define a dataframe-agnostic function in which we pass an expression
            to add it as a new column:

            >>> def agnostic_with_columns(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return (
            ...         df.with_columns((nw.col("a") * 2).alias("2a")).collect().to_native()
            ...     )

            We can then pass any supported library such as Polars or Dask to `agnostic_with_columns`:

            >>> agnostic_with_columns(lf_pl)
            shape: (4, 4)
            ┌─────┬──────┬───────┬─────┐
            │ a   ┆ b    ┆ c     ┆ 2a  │
            │ --- ┆ ---  ┆ ---   ┆ --- │
            │ i64 ┆ f64  ┆ bool  ┆ i64 │
            ╞═════╪══════╪═══════╪═════╡
            │ 1   ┆ 0.5  ┆ true  ┆ 2   │
            │ 2   ┆ 4.0  ┆ true  ┆ 4   │
            │ 3   ┆ 10.0 ┆ false ┆ 6   │
            │ 4   ┆ 13.0 ┆ true  ┆ 8   │
            └─────┴──────┴───────┴─────┘
            >>> agnostic_with_columns(lf_dask)
               a     b      c  2a
            0  1   0.5   True   2
            1  2   4.0   True   4
            2  3  10.0  False   6
            3  4  13.0   True   8
        """,
    "select": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6, 7, 8],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            Let's define a dataframe-agnostic function in which we pass the name of a
            column to select that column.

            >>> def agnostic_select(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select("foo").collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_select`:

            >>> agnostic_select(lf_pl)
            shape: (3, 1)
            ┌─────┐
            │ foo │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            │ 2   │
            │ 3   │
            └─────┘
            >>> agnostic_select(lf_dask)
               foo
            0    1
            1    2
            2    3

            Multiple columns can be selected by passing a list of column names.

            >>> def agnostic_select(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(["foo", "bar"]).collect().to_native()

            >>> agnostic_select(lf_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 6   │
            │ 2   ┆ 7   │
            │ 3   ┆ 8   │
            └─────┴─────┘
            >>> agnostic_select(lf_dask)
               foo  bar
            0    1    6
            1    2    7
            2    3    8

            Multiple columns can also be selected using positional arguments instead of a
            list. Expressions are also accepted.

            >>> def agnostic_select(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("foo"), nw.col("bar") + 1).collect().to_native()

            >>> agnostic_select(lf_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 7   │
            │ 2   ┆ 8   │
            │ 3   ┆ 9   │
            └─────┴─────┘
            >>> agnostic_select(lf_dask)
               foo  bar
            0    1    7
            1    2    8
            2    3    9

            Use keyword arguments to easily name your expression inputs.

            >>> def agnostic_select(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(threshold=nw.col("foo") * 2).collect().to_native()

            >>> agnostic_select(lf_pl)
            shape: (3, 1)
            ┌───────────┐
            │ threshold │
            │ ---       │
            │ i64       │
            ╞═══════════╡
            │ 2         │
            │ 4         │
            │ 6         │
            └───────────┘
            >>> agnostic_select(lf_dask)
               threshold
            0          2
            1          4
            2          6
        """,
    "rename": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]}
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            We define a library agnostic function:

            >>> def agnostic_rename(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.rename({"foo": "apple"}).collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_rename`:

            >>> agnostic_rename(lf_pl)
            shape: (3, 3)
            ┌───────┬─────┬─────┐
            │ apple ┆ bar ┆ ham │
            │ ---   ┆ --- ┆ --- │
            │ i64   ┆ i64 ┆ str │
            ╞═══════╪═════╪═════╡
            │ 1     ┆ 6   ┆ a   │
            │ 2     ┆ 7   ┆ b   │
            │ 3     ┆ 8   ┆ c   │
            └───────┴─────┴─────┘
            >>> agnostic_rename(lf_dask)
               apple  bar ham
            0      1    6   a
            1      2    7   b
            2      3    8   c
        """,
    "head": """
            >>> import narwhals as nw
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": [1, 2, 3, 4, 5, 6],
            ...     "b": [7, 8, 9, 10, 11, 12],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            Let's define a dataframe-agnostic function that gets the first 3 rows.

            >>> def agnostic_head(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.head(3).collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_head`:

            >>> agnostic_head(lf_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 7   │
            │ 2   ┆ 8   │
            │ 3   ┆ 9   │
            └─────┴─────┘
            >>> agnostic_head(lf_dask)
               a  b
            0  1  7
            1  2  8
            2  3  9
        """,
    "tail": """
            >>> import narwhals as nw
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": [1, 2, 3, 4, 5, 6],
            ...     "b": [7, 8, 9, 10, 11, 12],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=1)

            Let's define a dataframe-agnostic function that gets the last 3 rows.

            >>> def agnostic_tail(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.tail(3).collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_tail`:

            >>> agnostic_tail(lf_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 4   ┆ 10  │
            │ 5   ┆ 11  │
            │ 6   ┆ 12  │
            └─────┴─────┘
            >>> agnostic_tail(lf_dask)
               a   b
            3  4  10
            4  5  11
            5  6  12
        """,
    "drop": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            We define a library agnostic function:

            >>> def agnostic_drop(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.drop("ham").collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_drop`:

            >>> agnostic_drop(lf_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ i64 ┆ f64 │
            ╞═════╪═════╡
            │ 1   ┆ 6.0 │
            │ 2   ┆ 7.0 │
            │ 3   ┆ 8.0 │
            └─────┴─────┘
            >>> agnostic_drop(lf_dask)
               foo  bar
            0    1  6.0
            1    2  7.0
            2    3  8.0

            Use positional arguments to drop multiple columns.

            >>> def agnostic_drop(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.drop("foo", "ham").collect().to_native()

            >>> agnostic_drop(lf_pl)
            shape: (3, 1)
            ┌─────┐
            │ bar │
            │ --- │
            │ f64 │
            ╞═════╡
            │ 6.0 │
            │ 7.0 │
            │ 8.0 │
            └─────┘
            >>> agnostic_drop(lf_dask)
               bar
            0  6.0
            1  7.0
            2  8.0
        """,
    "unique": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "foo": [1, 2, 3, 1],
            ...     "bar": ["a", "a", "a", "a"],
            ...     "ham": ["b", "b", "b", "b"],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            We define a library agnostic function:

            >>> def agnostic_unique(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.unique(["bar", "ham"]).collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_unique`:

            >>> agnostic_unique(lf_pl)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ str ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ a   ┆ b   │
            └─────┴─────┴─────┘
            >>> agnostic_unique(lf_dask)
               foo bar ham
            0    1   a   b
        """,
    "filter": """
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6, 7, 8],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            Let's define a dataframe-agnostic function in which we filter on
            one condition.

            >>> def agnostic_filter(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.filter(nw.col("foo") > 1).collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_filter`:

            >>> agnostic_filter(lf_pl)
            shape: (2, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 2   ┆ 7   ┆ b   │
            │ 3   ┆ 8   ┆ c   │
            └─────┴─────┴─────┘
            >>> agnostic_filter(lf_dask)
               foo  bar ham
            1    2    7   b
            2    3    8   c

            Filter on multiple conditions:

            >>> def agnostic_filter(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return (
            ...         df.filter((nw.col("foo") < 3) & (nw.col("ham") == "a"))
            ...         .collect()
            ...         .to_native()
            ...     )

            >>> agnostic_filter(lf_pl)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            └─────┴─────┴─────┘
            >>> agnostic_filter(lf_dask)
               foo  bar ham
            0    1    6   a

            Provide multiple filters using `*args` syntax:

            >>> def agnostic_filter(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return (
            ...         df.filter(
            ...             nw.col("foo") == 1,
            ...             nw.col("ham") == "a",
            ...         )
            ...         .collect()
            ...         .to_native()
            ...     )

            >>> agnostic_filter(lf_pl)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            └─────┴─────┴─────┘
            >>> agnostic_filter(lf_dask)
               foo  bar ham
            0    1    6   a

            Filter on an OR condition:

            >>> def agnostic_filter(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return (
            ...         df.filter((nw.col("foo") == 1) | (nw.col("ham") == "c"))
            ...         .collect()
            ...         .to_native()
            ...     )

            >>> agnostic_filter(lf_pl)
            shape: (2, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            │ 3   ┆ 8   ┆ c   │
            └─────┴─────┴─────┘
            >>> agnostic_filter(lf_dask)
               foo  bar ham
            0    1    6   a
            2    3    8   c

            Provide multiple filters using `**kwargs` syntax:

            >>> def agnostic_filter(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.filter(foo=2, ham="b").collect().to_native()

            >>> agnostic_filter(lf_pl)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 2   ┆ 7   ┆ b   │
            └─────┴─────┴─────┘
            >>> agnostic_filter(lf_dask)
               foo  bar ham
            1    2    7   b
        """,
    "group_by": """
            Group by one column and call `agg` to compute the grouped sum of
            another column.

            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": ["a", "b", "a", "b", "c"],
            ...     "b": [1, 2, 1, 3, 3],
            ...     "c": [5, 4, 3, 2, 1],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            Let's define a dataframe-agnostic function in which we group by one column
            and call `agg` to compute the grouped sum of another column.

            >>> def agnostic_group_by_agg(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return (
            ...         df.group_by("a")
            ...         .agg(nw.col("b").sum())
            ...         .sort("a")
            ...         .collect()
            ...         .to_native()
            ...     )

            We can then pass any supported library such as Polars or Dask to `agnostic_group_by_agg`:

            >>> agnostic_group_by_agg(lf_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ str ┆ i64 │
            ╞═════╪═════╡
            │ a   ┆ 2   │
            │ b   ┆ 5   │
            │ c   ┆ 3   │
            └─────┴─────┘
            >>> agnostic_group_by_agg(lf_dask)
               a  b
            0  a  2
            1  b  5
            2  c  3

            Group by multiple columns by passing a list of column names.

            >>> def agnostic_group_by_agg(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return (
            ...         df.group_by(["a", "b"])
            ...         .agg(nw.max("c"))
            ...         .sort(["a", "b"])
            ...         .collect()
            ...         .to_native()
            ...     )

            >>> agnostic_group_by_agg(lf_pl)
            shape: (4, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ a   ┆ 1   ┆ 5   │
            │ b   ┆ 2   ┆ 4   │
            │ b   ┆ 3   ┆ 2   │
            │ c   ┆ 3   ┆ 1   │
            └─────┴─────┴─────┘
            >>> agnostic_group_by_agg(lf_dask)
               a  b  c
            0  a  1  5
            1  b  2  4
            2  b  3  2
            3  c  3  1
        """,
    "sort": """
            >>> import narwhals as nw
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": [1, 2, None],
            ...     "b": [6.0, 5.0, 4.0],
            ...     "c": ["a", "c", "b"],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            Let's define a dataframe-agnostic function in which we sort by multiple
            columns in different orders

            >>> def agnostic_sort(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.sort("c", "a", descending=[False, True]).collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_sort`:

            >>> agnostic_sort(lf_pl)
            shape: (3, 3)
            ┌──────┬─────┬─────┐
            │ a    ┆ b   ┆ c   │
            │ ---  ┆ --- ┆ --- │
            │ i64  ┆ f64 ┆ str │
            ╞══════╪═════╪═════╡
            │ 1    ┆ 6.0 ┆ a   │
            │ null ┆ 4.0 ┆ b   │
            │ 2    ┆ 5.0 ┆ c   │
            └──────┴─────┴─────┘
            >>> agnostic_sort(lf_dask)
                 a    b  c
            0  1.0  6.0  a
            2  NaN  4.0  b
            1  2.0  5.0  c
        """,
    "join": """
            >>> import narwhals as nw
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6.0, 7.0, 8.0],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> data_other = {
            ...     "apple": ["x", "y", "z"],
            ...     "ham": ["a", "b", "d"],
            ... }

            >>> lf_pl = pl.LazyFrame(data)
            >>> other_pl = pl.LazyFrame(data_other)
            >>> lf_dask = dd.from_dict(data, npartitions=2)
            >>> other_dask = dd.from_dict(data_other, npartitions=2)

            Let's define a dataframe-agnostic function in which we join over "ham" column:

            >>> def agnostic_join_on_ham(
            ...     df_native: IntoFrameT,
            ...     other_native: IntoFrameT,
            ... ) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     other = nw.from_native(other_native)
            ...     return (
            ...         df.join(other, left_on="ham", right_on="ham")
            ...         .sort("ham")
            ...         .collect()
            ...         .to_native()
            ...     )

            We can then pass any supported library such as Polars or Dask to `agnostic_join_on_ham`:

            >>> agnostic_join_on_ham(lf_pl, other_pl)
            shape: (2, 4)
            ┌─────┬─────┬─────┬───────┐
            │ foo ┆ bar ┆ ham ┆ apple │
            │ --- ┆ --- ┆ --- ┆ ---   │
            │ i64 ┆ f64 ┆ str ┆ str   │
            ╞═════╪═════╪═════╪═══════╡
            │ 1   ┆ 6.0 ┆ a   ┆ x     │
            │ 2   ┆ 7.0 ┆ b   ┆ y     │
            └─────┴─────┴─────┴───────┘
            >>> agnostic_join_on_ham(lf_dask, other_dask)
               foo  bar ham apple
            0    1  6.0   a     x
            0    2  7.0   b     y
        """,
    "join_asof": """
            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> from typing import Literal
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data_gdp = {
            ...     "datetime": [
            ...         datetime(2016, 1, 1),
            ...         datetime(2017, 1, 1),
            ...         datetime(2018, 1, 1),
            ...         datetime(2019, 1, 1),
            ...         datetime(2020, 1, 1),
            ...     ],
            ...     "gdp": [4164, 4411, 4566, 4696, 4827],
            ... }
            >>> data_population = {
            ...     "datetime": [
            ...         datetime(2016, 3, 1),
            ...         datetime(2018, 8, 1),
            ...         datetime(2019, 1, 1),
            ...     ],
            ...     "population": [82.19, 82.66, 83.12],
            ... }
            >>> gdp_pl = pl.LazyFrame(data_gdp)
            >>> population_pl = pl.LazyFrame(data_population)
            >>> gdp_dask = dd.from_dict(data_gdp, npartitions=2)
            >>> population_dask = dd.from_dict(data_population, npartitions=2)

            Let's define a dataframe-agnostic function in which we join over "datetime" column:

            >>> def agnostic_join_asof_datetime(
            ...     df_native: IntoFrameT,
            ...     other_native: IntoFrameT,
            ...     strategy: Literal["backward", "forward", "nearest"],
            ... ) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     other = nw.from_native(other_native)
            ...     return (
            ...         df.sort("datetime")
            ...         .join_asof(other, on="datetime", strategy=strategy)
            ...         .collect()
            ...         .to_native()
            ...     )

            We can then pass any supported library such as Polars or Dask to `agnostic_join_asof_datetime`:

            >>> agnostic_join_asof_datetime(population_pl, gdp_pl, strategy="backward")
            shape: (3, 3)
            ┌─────────────────────┬────────────┬──────┐
            │ datetime            ┆ population ┆ gdp  │
            │ ---                 ┆ ---        ┆ ---  │
            │ datetime[μs]        ┆ f64        ┆ i64  │
            ╞═════════════════════╪════════════╪══════╡
            │ 2016-03-01 00:00:00 ┆ 82.19      ┆ 4164 │
            │ 2018-08-01 00:00:00 ┆ 82.66      ┆ 4566 │
            │ 2019-01-01 00:00:00 ┆ 83.12      ┆ 4696 │
            └─────────────────────┴────────────┴──────┘
            >>> agnostic_join_asof_datetime(population_dask, gdp_dask, strategy="backward")
                datetime  population   gdp
            0 2016-03-01       82.19  4164
            1 2018-08-01       82.66  4566
            0 2019-01-01       83.12  4696

            Here is a real-world times-series example that uses `by` argument.

            >>> from datetime import datetime
            >>> import narwhals as nw
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data_quotes = {
            ...     "datetime": [
            ...         datetime(2016, 5, 25, 13, 30, 0, 23),
            ...         datetime(2016, 5, 25, 13, 30, 0, 23),
            ...         datetime(2016, 5, 25, 13, 30, 0, 30),
            ...         datetime(2016, 5, 25, 13, 30, 0, 41),
            ...         datetime(2016, 5, 25, 13, 30, 0, 48),
            ...         datetime(2016, 5, 25, 13, 30, 0, 49),
            ...         datetime(2016, 5, 25, 13, 30, 0, 72),
            ...         datetime(2016, 5, 25, 13, 30, 0, 75),
            ...     ],
            ...     "ticker": [
            ...         "GOOG",
            ...         "MSFT",
            ...         "MSFT",
            ...         "MSFT",
            ...         "GOOG",
            ...         "AAPL",
            ...         "GOOG",
            ...         "MSFT",
            ...     ],
            ...     "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
            ...     "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
            ... }
            >>> data_trades = {
            ...     "datetime": [
            ...         datetime(2016, 5, 25, 13, 30, 0, 23),
            ...         datetime(2016, 5, 25, 13, 30, 0, 38),
            ...         datetime(2016, 5, 25, 13, 30, 0, 48),
            ...         datetime(2016, 5, 25, 13, 30, 0, 49),
            ...         datetime(2016, 5, 25, 13, 30, 0, 48),
            ...     ],
            ...     "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
            ...     "price": [51.95, 51.95, 720.77, 720.92, 98.0],
            ...     "quantity": [75, 155, 100, 100, 100],
            ... }
            >>> quotes_pl = pl.LazyFrame(data_quotes)
            >>> trades_pl = pl.LazyFrame(data_trades)
            >>> quotes_dask = dd.from_dict(data_quotes, npartitions=2)
            >>> trades_dask = dd.from_dict(data_trades, npartitions=2)

            Let's define a dataframe-agnostic function in which we join over "datetime" and by "ticker" columns:

            >>> def agnostic_join_asof_datetime_by_ticker(
            ...     df_native: IntoFrameT,
            ...     other_native: IntoFrameT,
            ... ) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     other = nw.from_native(other_native)
            ...     return (
            ...         df.sort("datetime", "ticker")
            ...         .join_asof(other, on="datetime", by="ticker")
            ...         .sort("datetime", "ticker")
            ...         .collect()
            ...         .to_native()
            ...     )

            We can then pass any supported library such as Polars or Dask to `agnostic_join_asof_datetime_by_ticker`:

            >>> agnostic_join_asof_datetime_by_ticker(trades_pl, quotes_pl)
            shape: (5, 6)
            ┌────────────────────────────┬────────┬────────┬──────────┬───────┬────────┐
            │ datetime                   ┆ ticker ┆ price  ┆ quantity ┆ bid   ┆ ask    │
            │ ---                        ┆ ---    ┆ ---    ┆ ---      ┆ ---   ┆ ---    │
            │ datetime[μs]               ┆ str    ┆ f64    ┆ i64      ┆ f64   ┆ f64    │
            ╞════════════════════════════╪════════╪════════╪══════════╪═══════╪════════╡
            │ 2016-05-25 13:30:00.000023 ┆ MSFT   ┆ 51.95  ┆ 75       ┆ 51.95 ┆ 51.96  │
            │ 2016-05-25 13:30:00.000038 ┆ MSFT   ┆ 51.95  ┆ 155      ┆ 51.97 ┆ 51.98  │
            │ 2016-05-25 13:30:00.000048 ┆ AAPL   ┆ 98.0   ┆ 100      ┆ null  ┆ null   │
            │ 2016-05-25 13:30:00.000048 ┆ GOOG   ┆ 720.77 ┆ 100      ┆ 720.5 ┆ 720.93 │
            │ 2016-05-25 13:30:00.000049 ┆ GOOG   ┆ 720.92 ┆ 100      ┆ 720.5 ┆ 720.93 │
            └────────────────────────────┴────────┴────────┴──────────┴───────┴────────┘
            >>> agnostic_join_asof_datetime_by_ticker(trades_dask, quotes_dask)
                                datetime ticker   price  quantity     bid     ask
            0 2016-05-25 13:30:00.000023   MSFT   51.95        75   51.95   51.96
            0 2016-05-25 13:30:00.000038   MSFT   51.95       155   51.97   51.98
            1 2016-05-25 13:30:00.000048   AAPL   98.00       100     NaN     NaN
            2 2016-05-25 13:30:00.000048   GOOG  720.77       100  720.50  720.93
            3 2016-05-25 13:30:00.000049   GOOG  720.92       100  720.50  720.93
        """,
    "is_duplicated": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> from narwhals.typing import IntoSeries
            >>> data = {
            ...     "a": [1, 2, 3, 1],
            ...     "b": ["x", "y", "z", "x"],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_duplicated(df_native: IntoDataFrame) -> IntoSeries:
            ...     df = nw.from_native(df_native, eager_only=True)
            ...     return df.is_duplicated().to_native()

            We can then pass any supported library such as Pandas, Polars, or PyArrow
            to `agnostic_is_duplicated`:

            >>> agnostic_is_duplicated(df_pd)
            0     True
            1    False
            2    False
            3     True
            dtype: bool

            >>> agnostic_is_duplicated(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
                true
                false
                false
                true
            ]
            >>> agnostic_is_duplicated(df_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                true,
                false,
                false,
                true
              ]
            ]
        """,
    "is_empty": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame

            Let's define a dataframe-agnostic function that filters rows in which "foo"
            values are greater than 10, and then checks if the result is empty or not:

            >>> def agnostic_is_empty(df_native: IntoDataFrame) -> bool:
            ...     df = nw.from_native(df_native, eager_only=True)
            ...     return df.filter(nw.col("foo") > 10).is_empty()

            We can then pass any supported library such as Pandas, Polars, or PyArrow
            to `agnostic_is_empty`:

            >>> data = {"foo": [1, 2, 3], "bar": [4, 5, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)
            >>> agnostic_is_empty(df_pd), agnostic_is_empty(df_pl), agnostic_is_empty(df_pa)
            (True, True, True)

            >>> data = {"foo": [100, 2, 3], "bar": [4, 5, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)
            >>> agnostic_is_empty(df_pd), agnostic_is_empty(df_pl), agnostic_is_empty(df_pa)
            (False, False, False)
        """,
    "is_unique": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> from narwhals.typing import IntoSeries
            >>> data = {
            ...     "a": [1, 2, 3, 1],
            ...     "b": ["x", "y", "z", "x"],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_unique(df_native: IntoDataFrame) -> IntoSeries:
            ...     df = nw.from_native(df_native, eager_only=True)
            ...     return df.is_unique().to_native()

            We can then pass any supported library such as Pandas, Polars, or PyArrow
            to `agnostic_is_unique`:

            >>> agnostic_is_unique(df_pd)
            0    False
            1     True
            2     True
            3    False
            dtype: bool

            >>> agnostic_is_unique(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
                false
                 true
                 true
                false
            ]
            >>> agnostic_is_unique(df_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                false,
                true,
                true,
                false
              ]
            ]
        """,
    "null_count": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>> data = {
            ...     "foo": [1, None, 3],
            ...     "bar": [6, 7, None],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function that returns the null count of
            each columns:

            >>> def agnostic_null_count(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.null_count().to_native()

            We can then pass any supported library such as Pandas, Polars, or PyArrow to
            `agnostic_null_count`:

            >>> agnostic_null_count(df_pd)
               foo  bar  ham
            0    1    1    0

            >>> agnostic_null_count(df_pl)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ u32 ┆ u32 ┆ u32 │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 1   ┆ 0   │
            └─────┴─────┴─────┘

            >>> agnostic_null_count(df_pa)
            pyarrow.Table
            foo: int64
            bar: int64
            ham: int64
            ----
            foo: [[1]]
            bar: [[1]]
            ham: [[0]]
        """,
    "item": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function that returns item at given row/column

            >>> def agnostic_item(
            ...     df_native: IntoDataFrame, row: int | None, column: int | str | None
            ... ):
            ...     df = nw.from_native(df_native, eager_only=True)
            ...     return df.item(row, column)

            We can then pass any supported library such as Pandas, Polars, or PyArrow
            to `agnostic_item`:

            >>> agnostic_item(df_pd, 1, 1), agnostic_item(df_pd, 2, "b")
            (np.int64(5), np.int64(6))
            >>> agnostic_item(df_pl, 1, 1), agnostic_item(df_pl, 2, "b")
            (5, 6)
            >>> agnostic_item(df_pa, 1, 1), agnostic_item(df_pa, 2, "b")
            (5, 6)
        """,
    "clone": """
            >>> import narwhals as nw
            >>> import polars as pl
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2], "b": [3, 4]}
            >>> lf_pl = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function in which we copy the DataFrame:

            >>> def agnostic_clone(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.clone().collect().to_native()

            We can then pass any supported library such as Polars to `agnostic_clone`:

            >>> agnostic_clone(lf_pl)
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 3   │
            │ 2   ┆ 4   │
            └─────┴─────┘
        """,
    "gather_every": """
            >>> import narwhals as nw
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            Let's define a dataframe-agnostic function in which we gather every 2 rows,
            starting from a offset of 1:

            >>> def agnostic_gather_every(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.gather_every(n=2, offset=1).collect().to_native()

            We can then pass any supported library such as Polars or Dask to `agnostic_gather_every`:

            >>> agnostic_gather_every(lf_pl)
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 2   ┆ 6   │
            │ 4   ┆ 8   │
            └─────┴─────┘
            >>> agnostic_gather_every(lf_dask)
               a  b
            1  2  6
            3  4  8
        """,
    "pivot": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrameT
            >>> data = {
            ...     "ix": [1, 1, 2, 2, 1, 2],
            ...     "col": ["a", "a", "a", "a", "b", "b"],
            ...     "foo": [0, 1, 2, 2, 7, 1],
            ...     "bar": [0, 2, 0, 0, 9, 4],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_pivot(df_native: IntoDataFrameT) -> IntoDataFrameT:
            ...     df = nw.from_native(df_native, eager_only=True)
            ...     return df.pivot("col", index="ix", aggregate_function="sum").to_native()

            We can then pass any supported library such as Pandas or Polars
            to `agnostic_pivot`:

            >>> agnostic_pivot(df_pd)
               ix  foo_a  foo_b  bar_a  bar_b
            0   1      1      7      2      9
            1   2      4      1      0      4
            >>> agnostic_pivot(df_pl)
            shape: (2, 5)
            ┌─────┬───────┬───────┬───────┬───────┐
            │ ix  ┆ foo_a ┆ foo_b ┆ bar_a ┆ bar_b │
            │ --- ┆ ---   ┆ ---   ┆ ---   ┆ ---   │
            │ i64 ┆ i64   ┆ i64   ┆ i64   ┆ i64   │
            ╞═════╪═══════╪═══════╪═══════╪═══════╡
            │ 1   ┆ 1     ┆ 7     ┆ 2     ┆ 9     │
            │ 2   ┆ 4     ┆ 1     ┆ 0     ┆ 4     │
            └─────┴───────┴───────┴───────┴───────┘
        """,
    "to_arrow": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> data = {"foo": [1, 2, 3], "bar": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function that converts to arrow table:

            >>> def agnostic_to_arrow(df_native: IntoDataFrame) -> pa.Table:
            ...     df = nw.from_native(df_native, eager_only=True)
            ...     return df.to_arrow()

            We can then pass any supported library such as Pandas, Polars, or PyArrow
            to `agnostic_to_arrow`:

            >>> agnostic_to_arrow(df_pd)
            pyarrow.Table
            foo: int64
            bar: string
            ----
            foo: [[1,2,3]]
            bar: [["a","b","c"]]

            >>> agnostic_to_arrow(df_pl)
            pyarrow.Table
            foo: int64
            bar: large_string
            ----
            foo: [[1,2,3]]
            bar: [["a","b","c"]]

            >>> agnostic_to_arrow(df_pa)
            pyarrow.Table
            foo: int64
            bar: string
            ----
            foo: [[1,2,3]]
            bar: [["a","b","c"]]
        """,
    "sample": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrameT
            >>> data = {"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_sample(df_native: IntoDataFrameT) -> IntoDataFrameT:
            ...     df = nw.from_native(df_native, eager_only=True)
            ...     return df.sample(n=2, seed=123).to_native()

            We can then pass any supported library such as Pandas, Polars, or PyArrow
            to `agnostic_sample`:

            >>> agnostic_sample(df_pd)
               a  b
            3  4  y
            0  1  x
            >>> agnostic_sample(df_pl)
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ str │
            ╞═════╪═════╡
            │ 2   ┆ y   │
            │ 3   ┆ x   │
            └─────┴─────┘
            >>> agnostic_sample(df_pa)
            pyarrow.Table
            a: int64
            b: string
            ----
            a: [[1,3]]
            b: [["x","x"]]

            As you can see, by using the same seed, the result will be consistent within
            the same backend, but not necessarely across different backends.
        """,
    "unpivot": """
            >>> import narwhals as nw
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": ["x", "y", "z"],
            ...     "b": [1, 3, 5],
            ...     "c": [2, 4, 6],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            We define a library agnostic function:

            >>> def agnostic_unpivot(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return (
            ...         (df.unpivot(on=["b", "c"], index="a").sort(["variable", "a"]))
            ...         .collect()
            ...         .to_native()
            ...     )

            We can then pass any supported library such as Polars or Dask to `agnostic_unpivot`:

            >>> agnostic_unpivot(lf_pl)
            shape: (6, 3)
            ┌─────┬──────────┬───────┐
            │ a   ┆ variable ┆ value │
            │ --- ┆ ---      ┆ ---   │
            │ str ┆ str      ┆ i64   │
            ╞═════╪══════════╪═══════╡
            │ x   ┆ b        ┆ 1     │
            │ y   ┆ b        ┆ 3     │
            │ z   ┆ b        ┆ 5     │
            │ x   ┆ c        ┆ 2     │
            │ y   ┆ c        ┆ 4     │
            │ z   ┆ c        ┆ 6     │
            └─────┴──────────┴───────┘
            >>> agnostic_unpivot(lf_dask)
               a variable  value
            0  x        b      1
            1  y        b      3
            0  z        b      5
            2  x        c      2
            3  y        c      4
            1  z        c      6
        """,
    "explode": """
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>> import polars as pl
            >>> data = {
            ...     "a": ["x", "y", "z", "w"],
            ...     "lst1": [[1, 2], None, [None], []],
            ...     "lst2": [[3, None], None, [42], []],
            ... }

            We define a library agnostic function:

            >>> def agnostic_explode(df_native: IntoFrameT) -> IntoFrameT:
            ...     return (
            ...         nw.from_native(df_native)
            ...         .with_columns(nw.col("lst1", "lst2").cast(nw.List(nw.Int32())))
            ...         .explode("lst1", "lst2")
            ...         .collect()
            ...         .to_native()
            ...     )

            We can then pass any supported library such as Polars to `agnostic_explode`:

            >>> agnostic_explode(pl.LazyFrame(data))
            shape: (5, 3)
            ┌─────┬──────┬──────┐
            │ a   ┆ lst1 ┆ lst2 │
            │ --- ┆ ---  ┆ ---  │
            │ str ┆ i32  ┆ i32  │
            ╞═════╪══════╪══════╡
            │ x   ┆ 1    ┆ 3    │
            │ x   ┆ 2    ┆ null │
            │ y   ┆ null ┆ null │
            │ z   ┆ null ┆ 42   │
            │ w   ┆ null ┆ null │
            └─────┴──────┴──────┘
        """,
    "collect": """
            >>> import narwhals as nw
            >>> import polars as pl
            >>> import dask.dataframe as dd
            >>> data = {
            ...     "a": ["a", "b", "a", "b", "b", "c"],
            ...     "b": [1, 2, 3, 4, 5, 6],
            ...     "c": [6, 5, 4, 3, 2, 1],
            ... }
            >>> lf_pl = pl.LazyFrame(data)
            >>> lf_dask = dd.from_dict(data, npartitions=2)

            >>> lf = nw.from_native(lf_pl)
            >>> lf  # doctest:+ELLIPSIS
            ┌─────────────────────────────┐
            |     Narwhals LazyFrame      |
            |-----------------------------|
            |<LazyFrame at ...
            └─────────────────────────────┘
            >>> df = lf.group_by("a").agg(nw.all().sum()).collect()
            >>> df.to_native().sort("a")
            shape: (3, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ a   ┆ 4   ┆ 10  │
            │ b   ┆ 11  ┆ 10  │
            │ c   ┆ 6   ┆ 1   │
            └─────┴─────┴─────┘

            >>> lf = nw.from_native(lf_dask)
            >>> lf
            ┌───────────────────────────────────┐
            |        Narwhals LazyFrame         |
            |-----------------------------------|
            |Dask DataFrame Structure:          |
            |                    a      b      c|
            |npartitions=2                      |
            |0              string  int64  int64|
            |3                 ...    ...    ...|
            |5                 ...    ...    ...|
            |Dask Name: frompandas, 1 expression|
            |Expr=df                            |
            └───────────────────────────────────┘
            >>> df = lf.group_by("a").agg(nw.col("b", "c").sum()).collect()
            >>> df.to_native()
               a   b   c
            0  a   4  10
            1  b  11  10
            2  c   6   1
        """,
}
