from __future__ import annotations

EXAMPLES = {
    "alias": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2], "b": [4, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_alias(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select((nw.col("b") + 10).alias("c")).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_alias`:

            >>> agnostic_alias(df_pd)
                c
            0  14
            1  15

            >>> agnostic_alias(df_pl)
            shape: (2, 1)
            ┌─────┐
            │ c   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 14  │
            │ 15  │
            └─────┘

            >>> agnostic_alias(df_pa)
            pyarrow.Table
            c: int64
            ----
            c: [[14,15]]
        """,
    "pipe": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3, 4]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Lets define a library-agnostic function:

            >>> def agnostic_pipe(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").pipe(lambda x: x + 1)).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_pipe`:

            >>> agnostic_pipe(df_pd)
               a
            0  2
            1  3
            2  4
            3  5

            >>> agnostic_pipe(df_pl)
            shape: (4, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 2   │
            │ 3   │
            │ 4   │
            │ 5   │
            └─────┘

            >>> agnostic_pipe(df_pa)
            pyarrow.Table
            a: int64
            ----
            a: [[2,3,4,5]]
        """,
    "cast": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_cast(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("foo").cast(nw.Float32), nw.col("bar").cast(nw.UInt8)
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_cast`:

            >>> agnostic_cast(df_pd)
               foo  bar
            0  1.0    6
            1  2.0    7
            2  3.0    8
            >>> agnostic_cast(df_pl)
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
            >>> agnostic_cast(df_pa)
            pyarrow.Table
            foo: float
            bar: uint8
            ----
            foo: [[1,2,3]]
            bar: [[6,7,8]]
        """,
    "any": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [True, False], "b": [True, True]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_any(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").any()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_any`:

            >>> agnostic_any(df_pd)
                  a     b
            0  True  True

            >>> agnostic_any(df_pl)
            shape: (1, 2)
            ┌──────┬──────┐
            │ a    ┆ b    │
            │ ---  ┆ ---  │
            │ bool ┆ bool │
            ╞══════╪══════╡
            │ true ┆ true │
            └──────┴──────┘

            >>> agnostic_any(df_pa)
            pyarrow.Table
            a: bool
            b: bool
            ----
            a: [[true]]
            b: [[true]]
        """,
    "all": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [True, False], "b": [True, True]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_all(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").all()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_all`:

            >>> agnostic_all(df_pd)
                   a     b
            0  False  True

            >>> agnostic_all(df_pl)
            shape: (1, 2)
            ┌───────┬──────┐
            │ a     ┆ b    │
            │ ---   ┆ ---  │
            │ bool  ┆ bool │
            ╞═══════╪══════╡
            │ false ┆ true │
            └───────┴──────┘

            >>> agnostic_all(df_pa)
            pyarrow.Table
            a: bool
            b: bool
            ----
            a: [[false]]
            b: [[true]]
        """,
    "ewm_mean": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a library agnostic function:

            >>> def agnostic_ewm_mean(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a").ewm_mean(com=1, ignore_nulls=False)
            ...     ).to_native()

            We can then pass either pandas or Polars to `agnostic_ewm_mean`:

            >>> agnostic_ewm_mean(df_pd)
                      a
            0  1.000000
            1  1.666667
            2  2.428571

            >>> agnostic_ewm_mean(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3, 1)
            ┌──────────┐
            │ a        │
            │ ---      │
            │ f64      │
            ╞══════════╡
            │ 1.0      │
            │ 1.666667 │
            │ 2.428571 │
            └──────────┘
        """,
    "mean": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [-1, 0, 1], "b": [2, 4, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_mean(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").mean()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_mean`:

            >>> agnostic_mean(df_pd)
                 a    b
            0  0.0  4.0

            >>> agnostic_mean(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ f64 ┆ f64 │
            ╞═════╪═════╡
            │ 0.0 ┆ 4.0 │
            └─────┴─────┘

            >>> agnostic_mean(df_pa)
            pyarrow.Table
            a: double
            b: double
            ----
            a: [[0]]
            b: [[4]]
        """,
    "median": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 8, 3], "b": [4, 5, 2]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_median(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").median()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_median`:

            >>> agnostic_median(df_pd)
                 a    b
            0  3.0  4.0

            >>> agnostic_median(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ f64 ┆ f64 │
            ╞═════╪═════╡
            │ 3.0 ┆ 4.0 │
            └─────┴─────┘

            >>> agnostic_median(df_pa)
            pyarrow.Table
            a: double
            b: double
            ----
            a: [[3]]
            b: [[4]]
        """,
    "std": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [20, 25, 60], "b": [1.5, 1, -1.4]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_std(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").std(ddof=0)).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_std`:

            >>> agnostic_std(df_pd)
                      a         b
            0  17.79513  1.265789
            >>> agnostic_std(df_pl)
            shape: (1, 2)
            ┌──────────┬──────────┐
            │ a        ┆ b        │
            │ ---      ┆ ---      │
            │ f64      ┆ f64      │
            ╞══════════╪══════════╡
            │ 17.79513 ┆ 1.265789 │
            └──────────┴──────────┘
            >>> agnostic_std(df_pa)
            pyarrow.Table
            a: double
            b: double
            ----
            a: [[17.795130420052185]]
            b: [[1.2657891697365016]]
        """,
    "var": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [20, 25, 60], "b": [1.5, 1, -1.4]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_var(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").var(ddof=0)).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_var`:

            >>> agnostic_var(df_pd)
                        a         b
            0  316.666667  1.602222

            >>> agnostic_var(df_pl)
            shape: (1, 2)
            ┌────────────┬──────────┐
            │ a          ┆ b        │
            │ ---        ┆ ---      │
            │ f64        ┆ f64      │
            ╞════════════╪══════════╡
            │ 316.666667 ┆ 1.602222 │
            └────────────┴──────────┘

            >>> agnostic_var(df_pa)
            pyarrow.Table
            a: double
            b: double
            ----
            a: [[316.6666666666667]]
            b: [[1.6022222222222222]]
        """,
    "map_batches": """
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

            >>> def agnostic_map_batches(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a", "b").map_batches(
            ...             lambda s: s.to_numpy() + 1, return_dtype=nw.Float64
            ...         )
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_map_batches`:

            >>> agnostic_map_batches(df_pd)
                 a    b
            0  2.0  5.0
            1  3.0  6.0
            2  4.0  7.0
            >>> agnostic_map_batches(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ f64 ┆ f64 │
            ╞═════╪═════╡
            │ 2.0 ┆ 5.0 │
            │ 3.0 ┆ 6.0 │
            │ 4.0 ┆ 7.0 │
            └─────┴─────┘
            >>> agnostic_map_batches(df_pa)
            pyarrow.Table
            a: double
            b: double
            ----
            a: [[2,3,4]]
            b: [[5,6,7]]
        """,
    "skew": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3, 4, 5], "b": [1, 1, 2, 10, 100]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_skew(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").skew()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_skew`:

            >>> agnostic_skew(df_pd)
                 a         b
            0  0.0  1.472427

            >>> agnostic_skew(df_pl)
            shape: (1, 2)
            ┌─────┬──────────┐
            │ a   ┆ b        │
            │ --- ┆ ---      │
            │ f64 ┆ f64      │
            ╞═════╪══════════╡
            │ 0.0 ┆ 1.472427 │
            └─────┴──────────┘

            >>> agnostic_skew(df_pa)
            pyarrow.Table
            a: double
            b: double
            ----
            a: [[0]]
            b: [[1.4724267269058975]]
        """,
    "sum": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [5, 10], "b": [50, 100]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_sum(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").sum()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_sum`:

            >>> agnostic_sum(df_pd)
                a    b
            0  15  150
            >>> agnostic_sum(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 15  ┆ 150 │
            └─────┴─────┘
            >>> agnostic_sum(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[15]]
            b: [[150]]
        """,
    "min": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2], "b": [4, 3]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_min(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.min("a", "b")).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_min`:

            >>> agnostic_min(df_pd)
               a  b
            0  1  3

            >>> agnostic_min(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 3   │
            └─────┴─────┘

            >>> agnostic_min(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[1]]
            b: [[3]]
        """,
    "max": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [10, 20], "b": [50, 100]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_max(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.max("a", "b")).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_max`:

            >>> agnostic_max(df_pd)
                a    b
            0  20  100

            >>> agnostic_max(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 20  ┆ 100 │
            └─────┴─────┘

            >>> agnostic_max(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[20]]
            b: [[100]]
        """,
    "arg_min": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [10, 20], "b": [150, 100]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_arg_min(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a", "b").arg_min().name.suffix("_arg_min")
            ...     ).to_native()

            We can then pass any supported library such as Pandas, Polars, or
            PyArrow to `agnostic_arg_min`:

            >>> agnostic_arg_min(df_pd)
               a_arg_min  b_arg_min
            0          0          1

            >>> agnostic_arg_min(df_pl)
            shape: (1, 2)
            ┌───────────┬───────────┐
            │ a_arg_min ┆ b_arg_min │
            │ ---       ┆ ---       │
            │ u32       ┆ u32       │
            ╞═══════════╪═══════════╡
            │ 0         ┆ 1         │
            └───────────┴───────────┘

            >>> agnostic_arg_min(df_pa)
            pyarrow.Table
            a_arg_min: int64
            b_arg_min: int64
            ----
            a_arg_min: [[0]]
            b_arg_min: [[1]]
        """,
    "arg_max": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [10, 20], "b": [150, 100]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_arg_max(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a", "b").arg_max().name.suffix("_arg_max")
            ...     ).to_native()

            We can then pass any supported library such as Pandas, Polars, or
            PyArrow to `agnostic_arg_max`:

            >>> agnostic_arg_max(df_pd)
               a_arg_max  b_arg_max
            0          1          0

            >>> agnostic_arg_max(df_pl)
            shape: (1, 2)
            ┌───────────┬───────────┐
            │ a_arg_max ┆ b_arg_max │
            │ ---       ┆ ---       │
            │ u32       ┆ u32       │
            ╞═══════════╪═══════════╡
            │ 1         ┆ 0         │
            └───────────┴───────────┘

            >>> agnostic_arg_max(df_pa)
            pyarrow.Table
            a_arg_max: int64
            b_arg_max: int64
            ----
            a_arg_max: [[1]]
            b_arg_max: [[0]]
        """,
    "count": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3], "b": [None, 4, 4]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_count(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.all().count()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_count`:

            >>> agnostic_count(df_pd)
               a  b
            0  3  2

            >>> agnostic_count(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ u32 ┆ u32 │
            ╞═════╪═════╡
            │ 3   ┆ 2   │
            └─────┴─────┘

            >>> agnostic_count(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[3]]
            b: [[2]]
        """,
    "n_unique": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3, 4, 5], "b": [1, 1, 3, 3, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_n_unique(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").n_unique()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_n_unique`:

            >>> agnostic_n_unique(df_pd)
               a  b
            0  5  3
            >>> agnostic_n_unique(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ u32 ┆ u32 │
            ╞═════╪═════╡
            │ 5   ┆ 3   │
            └─────┴─────┘
            >>> agnostic_n_unique(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[5]]
            b: [[3]]
        """,
    "unique": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 1, 3, 5, 5], "b": [2, 4, 4, 6, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_unique(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").unique(maintain_order=True)).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_unique`:

            >>> agnostic_unique(df_pd)
               a  b
            0  1  2
            1  3  4
            2  5  6

            >>> agnostic_unique(df_pl)
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

            >>> agnostic_unique(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[1,3,5]]
            b: [[2,4,6]]
        """,
    "abs": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, -2], "b": [-3, 4]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_abs(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").abs()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_abs`:

            >>> agnostic_abs(df_pd)
               a  b
            0  1  3
            1  2  4

            >>> agnostic_abs(df_pl)
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 3   │
            │ 2   ┆ 4   │
            └─────┴─────┘

            >>> agnostic_abs(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[1,2]]
            b: [[3,4]]
        """,
    "cum_sum": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 1, 3, 5, 5], "b": [2, 4, 4, 6, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_cum_sum(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a", "b").cum_sum()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_cum_sum`:

            >>> agnostic_cum_sum(df_pd)
                a   b
            0   1   2
            1   2   6
            2   5  10
            3  10  16
            4  15  22
            >>> agnostic_cum_sum(df_pl)
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
            >>> agnostic_cum_sum(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[1,2,5,10,15]]
            b: [[2,6,10,16,22]]
        """,
    "diff": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 1, 3, 5, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_diff(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(a_diff=nw.col("a").diff()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_diff`:

            >>> agnostic_diff(df_pd)
               a_diff
            0     NaN
            1     0.0
            2     2.0
            3     2.0
            4     0.0

            >>> agnostic_diff(df_pl)
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

            >>> agnostic_diff(df_pa)
            pyarrow.Table
            a_diff: int64
            ----
            a_diff: [[null,0,2,2,0]]
        """,
    "shift": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 1, 3, 5, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_shift(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(a_shift=nw.col("a").shift(n=1)).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_shift`:

            >>> agnostic_shift(df_pd)
               a_shift
            0      NaN
            1      1.0
            2      1.0
            3      3.0
            4      5.0

            >>> agnostic_shift(df_pl)
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

            >>> agnostic_shift(df_pa)
            pyarrow.Table
            a_shift: int64
            ----
            a_shift: [[null,1,1,3,5]]
        """,
    "replace_strict": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [3, 0, 1, 2]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define dataframe-agnostic functions:

            >>> def agnostic_replace_strict(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         b=nw.col("a").replace_strict(
            ...             [0, 1, 2, 3],
            ...             ["zero", "one", "two", "three"],
            ...             return_dtype=nw.String,
            ...         )
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_replace_strict`:

            >>> agnostic_replace_strict(df_pd)
               a      b
            0  3  three
            1  0   zero
            2  1    one
            3  2    two

            >>> agnostic_replace_strict(df_pl)
            shape: (4, 2)
            ┌─────┬───────┐
            │ a   ┆ b     │
            │ --- ┆ ---   │
            │ i64 ┆ str   │
            ╞═════╪═══════╡
            │ 3   ┆ three │
            │ 0   ┆ zero  │
            │ 1   ┆ one   │
            │ 2   ┆ two   │
            └─────┴───────┘

            >>> agnostic_replace_strict(df_pa)
            pyarrow.Table
            a: int64
            b: string
            ----
            a: [[3,0,1,2]]
            b: [["three","zero","one","two"]]
        """,
    "sort": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [5, None, 1, 2]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define dataframe-agnostic functions:

            >>> def agnostic_sort(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").sort()).to_native()

            >>> def agnostic_sort_descending(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").sort(descending=True)).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_sort` and `agnostic_sort_descending`:

            >>> agnostic_sort(df_pd)
                 a
            1  NaN
            2  1.0
            3  2.0
            0  5.0

            >>> agnostic_sort(df_pl)
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

            >>> agnostic_sort(df_pa)
            pyarrow.Table
            a: int64
            ----
            a: [[null,1,2,5]]

            >>> agnostic_sort_descending(df_pd)
                 a
            1  NaN
            0  5.0
            3  2.0
            2  1.0

            >>> agnostic_sort_descending(df_pl)
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

            >>> agnostic_sort_descending(df_pa)
            pyarrow.Table
            a: int64
            ----
            a: [[null,5,2,1]]
        """,
    "is_between": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3, 4, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_between(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").is_between(2, 4, "right")).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_between`:

            >>> agnostic_is_between(df_pd)
                   a
            0  False
            1  False
            2   True
            3   True
            4  False

            >>> agnostic_is_between(df_pl)
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

            >>> agnostic_is_between(df_pa)
            pyarrow.Table
            a: bool
            ----
            a: [[false,false,true,true,false]]
        """,
    "is_in": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 9, 10]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_in(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(b=nw.col("a").is_in([1, 2])).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_in`:

            >>> agnostic_is_in(df_pd)
                a      b
            0   1   True
            1   2   True
            2   9  False
            3  10  False

            >>> agnostic_is_in(df_pl)
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

            >>> agnostic_is_in(df_pa)
            pyarrow.Table
            a: int64
            b: bool
            ----
            a: [[1,2,9,10]]
            b: [[true,true,false,false]]
        """,
    "filter": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [2, 3, 4, 5, 6, 7], "b": [10, 11, 12, 13, 14, 15]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_filter(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a").filter(nw.col("a") > 4),
            ...         nw.col("b").filter(nw.col("b") < 13),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_filter`:

            >>> agnostic_filter(df_pd)
               a   b
            3  5  10
            4  6  11
            5  7  12

            >>> agnostic_filter(df_pl)
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

            >>> agnostic_filter(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[5,6,7]]
            b: [[10,11,12]]
        """,
    "is_null": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> df_pd = pd.DataFrame(
            ...     {
            ...         "a": [2, 4, None, 3, 5],
            ...         "b": [2.0, 4.0, float("nan"), 3.0, 5.0],
            ...     }
            ... )
            >>> data = {
            ...     "a": [2, 4, None, 3, 5],
            ...     "b": [2.0, 4.0, None, 3.0, 5.0],
            ... }
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_null(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         a_is_null=nw.col("a").is_null(), b_is_null=nw.col("b").is_null()
            ...     ).to_native()

            We can then pass any supported library such as Pandas, Polars, or
            PyArrow to `agnostic_is_null`:

            >>> agnostic_is_null(df_pd)
                 a    b  a_is_null  b_is_null
            0  2.0  2.0      False      False
            1  4.0  4.0      False      False
            2  NaN  NaN       True       True
            3  3.0  3.0      False      False
            4  5.0  5.0      False      False

            >>> agnostic_is_null(df_pl)
            shape: (5, 4)
            ┌──────┬──────┬───────────┬───────────┐
            │ a    ┆ b    ┆ a_is_null ┆ b_is_null │
            │ ---  ┆ ---  ┆ ---       ┆ ---       │
            │ i64  ┆ f64  ┆ bool      ┆ bool      │
            ╞══════╪══════╪═══════════╪═══════════╡
            │ 2    ┆ 2.0  ┆ false     ┆ false     │
            │ 4    ┆ 4.0  ┆ false     ┆ false     │
            │ null ┆ null ┆ true      ┆ true      │
            │ 3    ┆ 3.0  ┆ false     ┆ false     │
            │ 5    ┆ 5.0  ┆ false     ┆ false     │
            └──────┴──────┴───────────┴───────────┘

            >>> agnostic_is_null(df_pa)
            pyarrow.Table
            a: int64
            b: double
            a_is_null: bool
            b_is_null: bool
            ----
            a: [[2,4,null,3,5]]
            b: [[2,4,null,3,5]]
            a_is_null: [[false,false,true,false,false]]
            b_is_null: [[false,false,true,false,false]]
        """,
    "is_nan": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"orig": [0.0, None, 2.0]}
            >>> df_pd = pd.DataFrame(data).astype({"orig": "Float64"})
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_self_div_is_nan(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         divided=nw.col("orig") / nw.col("orig"),
            ...         divided_is_nan=(nw.col("orig") / nw.col("orig")).is_nan(),
            ...     ).to_native()

            We can then pass any supported library such as Pandas, Polars, or
            PyArrow to `agnostic_self_div_is_nan`:

            >>> print(agnostic_self_div_is_nan(df_pd))
               orig  divided  divided_is_nan
            0   0.0      NaN            True
            1  <NA>     <NA>            <NA>
            2   2.0      1.0           False

            >>> print(agnostic_self_div_is_nan(df_pl))
            shape: (3, 3)
            ┌──────┬─────────┬────────────────┐
            │ orig ┆ divided ┆ divided_is_nan │
            │ ---  ┆ ---     ┆ ---            │
            │ f64  ┆ f64     ┆ bool           │
            ╞══════╪═════════╪════════════════╡
            │ 0.0  ┆ NaN     ┆ true           │
            │ null ┆ null    ┆ null           │
            │ 2.0  ┆ 1.0     ┆ false          │
            └──────┴─────────┴────────────────┘

            >>> print(agnostic_self_div_is_nan(df_pa))
            pyarrow.Table
            orig: double
            divided: double
            divided_is_nan: bool
            ----
            orig: [[0,null,2]]
            divided: [[nan,null,1]]
            divided_is_nan: [[true,null,false]]
        """,
    "arg_true": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, None, None, 2]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_arg_true(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").is_null().arg_true()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_arg_true`:

            >>> agnostic_arg_true(df_pd)
               a
            1  1
            2  2

            >>> agnostic_arg_true(df_pl)
            shape: (2, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ u32 │
            ╞═════╡
            │ 1   │
            │ 2   │
            └─────┘

            >>> agnostic_arg_true(df_pa)
            pyarrow.Table
            a: int64
            ----
            a: [[1,2]]
        """,
    "fill_null": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> df_pd = pd.DataFrame(
            ...     {
            ...         "a": [2, 4, None, None, 3, 5],
            ...         "b": [2.0, 4.0, float("nan"), float("nan"), 3.0, 5.0],
            ...     }
            ... )
            >>> data = {
            ...     "a": [2, 4, None, None, 3, 5],
            ...     "b": [2.0, 4.0, None, None, 3.0, 5.0],
            ... }
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_fill_null(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(nw.col("a", "b").fill_null(0)).to_native()

            We can then pass any supported library such as Pandas, Polars, or
            PyArrow to `agnostic_fill_null`:

            >>> agnostic_fill_null(df_pd)
                 a    b
            0  2.0  2.0
            1  4.0  4.0
            2  0.0  0.0
            3  0.0  0.0
            4  3.0  3.0
            5  5.0  5.0

            >>> agnostic_fill_null(df_pl)
            shape: (6, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ f64 │
            ╞═════╪═════╡
            │ 2   ┆ 2.0 │
            │ 4   ┆ 4.0 │
            │ 0   ┆ 0.0 │
            │ 0   ┆ 0.0 │
            │ 3   ┆ 3.0 │
            │ 5   ┆ 5.0 │
            └─────┴─────┘

            >>> agnostic_fill_null(df_pa)
            pyarrow.Table
            a: int64
            b: double
            ----
            a: [[2,4,0,0,3,5]]
            b: [[2,4,0,0,3,5]]

            Using a strategy:

            >>> def agnostic_fill_null_with_strategy(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("a", "b")
            ...         .fill_null(strategy="forward", limit=1)
            ...         .name.suffix("_filled")
            ...     ).to_native()

            >>> agnostic_fill_null_with_strategy(df_pd)
                 a    b  a_filled  b_filled
            0  2.0  2.0       2.0       2.0
            1  4.0  4.0       4.0       4.0
            2  NaN  NaN       4.0       4.0
            3  NaN  NaN       NaN       NaN
            4  3.0  3.0       3.0       3.0
            5  5.0  5.0       5.0       5.0

            >>> agnostic_fill_null_with_strategy(df_pl)
            shape: (6, 4)
            ┌──────┬──────┬──────────┬──────────┐
            │ a    ┆ b    ┆ a_filled ┆ b_filled │
            │ ---  ┆ ---  ┆ ---      ┆ ---      │
            │ i64  ┆ f64  ┆ i64      ┆ f64      │
            ╞══════╪══════╪══════════╪══════════╡
            │ 2    ┆ 2.0  ┆ 2        ┆ 2.0      │
            │ 4    ┆ 4.0  ┆ 4        ┆ 4.0      │
            │ null ┆ null ┆ 4        ┆ 4.0      │
            │ null ┆ null ┆ null     ┆ null     │
            │ 3    ┆ 3.0  ┆ 3        ┆ 3.0      │
            │ 5    ┆ 5.0  ┆ 5        ┆ 5.0      │
            └──────┴──────┴──────────┴──────────┘

            >>> agnostic_fill_null_with_strategy(df_pa)
            pyarrow.Table
            a: int64
            b: double
            a_filled: int64
            b_filled: double
            ----
            a: [[2,4,null,null,3,5]]
            b: [[2,4,null,null,3,5]]
            a_filled: [[2,4,4,null,3,5]]
            b_filled: [[2,4,4,null,3,5]]
        """,
    "drop_nulls": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> df_pd = pd.DataFrame({"a": [2.0, 4.0, float("nan"), 3.0, None, 5.0]})
            >>> df_pl = pl.DataFrame({"a": [2.0, 4.0, None, 3.0, None, 5.0]})
            >>> df_pa = pa.table({"a": [2.0, 4.0, None, 3.0, None, 5.0]})

            Let's define a dataframe-agnostic function:

            >>> def agnostic_drop_nulls(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").drop_nulls()).to_native()

            We can then pass any supported library such as Pandas, Polars, or
            PyArrow to `agnostic_drop_nulls`:

            >>> agnostic_drop_nulls(df_pd)
                 a
            0  2.0
            1  4.0
            3  3.0
            5  5.0

            >>> agnostic_drop_nulls(df_pl)
            shape: (4, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ f64 │
            ╞═════╡
            │ 2.0 │
            │ 4.0 │
            │ 3.0 │
            │ 5.0 │
            └─────┘

            >>> agnostic_drop_nulls(df_pa)
            pyarrow.Table
            a: double
            ----
            a: [[2,4,3,5]]
        """,
    "sample": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_sample(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a").sample(fraction=1.0, with_replacement=True)
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_sample`:

            >>> agnostic_sample(df_pd)  # doctest: +SKIP
               a
            2  3
            0  1
            2  3

            >>> agnostic_sample(df_pl)  # doctest: +SKIP
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

            >>> agnostic_sample(df_pa)  # doctest: +SKIP
            pyarrow.Table
            a: int64
            ----
            a: [[1,3,3]]
        """,
    "over": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3], "b": [1, 1, 2]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_min_over_b(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         a_min_per_group=nw.col("a").min().over("b")
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_min_over_b`:

            >>> agnostic_min_over_b(df_pd)
               a  b  a_min_per_group
            0  1  1                1
            1  2  1                1
            2  3  2                3

            >>> agnostic_min_over_b(df_pl)
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

            >>> agnostic_min_over_b(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            a_min_per_group: int64
            ----
            a: [[1,2,3]]
            b: [[1,1,2]]
            a_min_per_group: [[1,1,3]]

            Cumulative operations are also supported, but (currently) only for
            pandas and Polars:

            >>> def agnostic_cum_sum(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(c=nw.col("a").cum_sum().over("b")).to_native()

            >>> agnostic_cum_sum(df_pd)
               a  b  c
            0  1  1  1
            1  2  1  3
            2  3  2  3

            >>> agnostic_cum_sum(df_pl)
            shape: (3, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 1   ┆ 1   │
            │ 2   ┆ 1   ┆ 3   │
            │ 3   ┆ 2   ┆ 3   │
            └─────┴─────┴─────┘
        """,
    "is_duplicated": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_duplicated(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.all().is_duplicated()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_duplicated`:

            >>> agnostic_is_duplicated(df_pd)
                   a      b
            0   True   True
            1  False   True
            2  False  False
            3   True  False

            >>> agnostic_is_duplicated(df_pl)
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

            >>> agnostic_is_duplicated(df_pa)
            pyarrow.Table
            a: bool
            b: bool
            ----
            a: [[true,false,false,true]]
            b: [[true,true,false,false]]
        """,
    "is_unique": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_unique(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.all().is_unique()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_unique`:

            >>> agnostic_is_unique(df_pd)
                   a      b
            0  False  False
            1   True  False
            2   True   True
            3  False   True

            >>> agnostic_is_unique(df_pl)
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

            >>> agnostic_is_unique(df_pa)
            pyarrow.Table
            a: bool
            b: bool
            ----
            a: [[false,true,true,false]]
            b: [[false,false,true,true]]
        """,
    "null_count": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, None, 1], "b": ["a", None, "b", None]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_null_count(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.all().null_count()).to_native()

            We can then pass any supported library such as Pandas, Polars, or
            PyArrow to `agnostic_null_count`:

            >>> agnostic_null_count(df_pd)
               a  b
            0  1  2

            >>> agnostic_null_count(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ u32 ┆ u32 │
            ╞═════╪═════╡
            │ 1   ┆ 2   │
            └─────┴─────┘

            >>> agnostic_null_count(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[1]]
            b: [[2]]
        """,
    "is_first_distinct": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_first_distinct(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.all().is_first_distinct()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_first_distinct`:

            >>> agnostic_is_first_distinct(df_pd)
                   a      b
            0   True   True
            1   True  False
            2   True   True
            3  False   True

            >>> agnostic_is_first_distinct(df_pl)
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

            >>> agnostic_is_first_distinct(df_pa)
            pyarrow.Table
            a: bool
            b: bool
            ----
            a: [[true,true,true,false]]
            b: [[true,false,true,true]]
        """,
    "is_last_distinct": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_last_distinct(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.all().is_last_distinct()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_last_distinct`:

            >>> agnostic_is_last_distinct(df_pd)
                   a      b
            0  False  False
            1   True   True
            2   True   True
            3   True   True

            >>> agnostic_is_last_distinct(df_pl)
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

            >>> agnostic_is_last_distinct(df_pa)
            pyarrow.Table
            a: bool
            b: bool
            ----
            a: [[false,true,true,true]]
            b: [[false,true,true,true]]
        """,
    "quantile": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": list(range(50)), "b": list(range(50, 100))}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_quantile(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a", "b").quantile(0.5, interpolation="linear")
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_quantile`:

            >>> agnostic_quantile(df_pd)
                  a     b
            0  24.5  74.5

            >>> agnostic_quantile(df_pl)
            shape: (1, 2)
            ┌──────┬──────┐
            │ a    ┆ b    │
            │ ---  ┆ ---  │
            │ f64  ┆ f64  │
            ╞══════╪══════╡
            │ 24.5 ┆ 74.5 │
            └──────┴──────┘

            >>> agnostic_quantile(df_pa)
            pyarrow.Table
            a: double
            b: double
            ----
            a: [[24.5]]
            b: [[74.5]]
        """,
    "head": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": list(range(10))}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function that returns the first 3 rows:

            >>> def agnostic_head(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").head(3)).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_head`:

            >>> agnostic_head(df_pd)
               a
            0  0
            1  1
            2  2

            >>> agnostic_head(df_pl)
            shape: (3, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 0   │
            │ 1   │
            │ 2   │
            └─────┘

            >>> agnostic_head(df_pa)
            pyarrow.Table
            a: int64
            ----
            a: [[0,1,2]]
        """,
    "tail": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": list(range(10))}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function that returns the last 3 rows:

            >>> def agnostic_tail(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").tail(3)).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_tail`:

            >>> agnostic_tail(df_pd)
               a
            7  7
            8  8
            9  9

            >>> agnostic_tail(df_pl)
            shape: (3, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 7   │
            │ 8   │
            │ 9   │
            └─────┘

            >>> agnostic_tail(df_pa)
            pyarrow.Table
            a: int64
            ----
            a: [[7,8,9]]
        """,
    "round": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1.12345, 2.56789, 3.901234]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function that rounds to the first decimal:

            >>> def agnostic_round(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").round(1)).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_round`:

            >>> agnostic_round(df_pd)
                 a
            0  1.1
            1  2.6
            2  3.9

            >>> agnostic_round(df_pl)
            shape: (3, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ f64 │
            ╞═════╡
            │ 1.1 │
            │ 2.6 │
            │ 3.9 │
            └─────┘

            >>> agnostic_round(df_pa)
            pyarrow.Table
            a: double
            ----
            a: [[1.1,2.6,3.9]]
        """,
    "len": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": ["x", "y", "z"], "b": [1, 2, 1]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function that computes the len over
            different values of "b" column:

            >>> def agnostic_len(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a").filter(nw.col("b") == 1).len().alias("a1"),
            ...         nw.col("a").filter(nw.col("b") == 2).len().alias("a2"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_len`:

            >>> agnostic_len(df_pd)
               a1  a2
            0   2   1

            >>> agnostic_len(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a1  ┆ a2  │
            │ --- ┆ --- │
            │ u32 ┆ u32 │
            ╞═════╪═════╡
            │ 2   ┆ 1   │
            └─────┴─────┘

            >>> agnostic_len(df_pa)
            pyarrow.Table
            a1: int64
            a2: int64
            ----
            a1: [[2]]
            a2: [[1]]
        """,
    "gather_every": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function in which gather every 2 rows,
            starting from a offset of 1:

            >>> def agnostic_gather_every(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").gather_every(n=2, offset=1)).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_gather_every`:

            >>> agnostic_gather_every(df_pd)
               a
            1  2
            3  4

            >>> agnostic_gather_every(df_pl)
            shape: (2, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 2   │
            │ 4   │
            └─────┘

            >>> agnostic_gather_every(df_pa)
            pyarrow.Table
            a: int64
            ----
            a: [[2,4]]
        """,
    "clip": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_clip_lower(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").clip(2)).to_native()

            We can then pass any supported library such as Pandas, Polars, or
            PyArrow to `agnostic_clip_lower`:

            >>> agnostic_clip_lower(df_pd)
               a
            0  2
            1  2
            2  3

            >>> agnostic_clip_lower(df_pl)
            shape: (3, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 2   │
            │ 2   │
            │ 3   │
            └─────┘

            >>> agnostic_clip_lower(df_pa)
            pyarrow.Table
            a: int64
            ----
            a: [[2,2,3]]

            We define another library agnostic function:

            >>> def agnostic_clip_upper(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").clip(upper_bound=2)).to_native()

            We can then pass any supported library such as Pandas, Polars, or
            PyArrow to `agnostic_clip_upper`:

            >>> agnostic_clip_upper(df_pd)
               a
            0  1
            1  2
            2  2

            >>> agnostic_clip_upper(df_pl)
            shape: (3, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            │ 2   │
            │ 2   │
            └─────┘

            >>> agnostic_clip_upper(df_pa)
            pyarrow.Table
            a: int64
            ----
            a: [[1,2,2]]

            We can have both at the same time

            >>> data = {"a": [-1, 1, -3, 3, -5, 5]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_clip(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").clip(-1, 3)).to_native()

            We can pass any supported library such as Pandas, Polars, or
            PyArrow to `agnostic_clip`:

            >>> agnostic_clip(df_pd)
               a
            0 -1
            1  1
            2 -1
            3  3
            4 -1
            5  3

            >>> agnostic_clip(df_pl)
            shape: (6, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ -1  │
            │ 1   │
            │ -1  │
            │ 3   │
            │ -1  │
            │ 3   │
            └─────┘

            >>> agnostic_clip(df_pa)
            pyarrow.Table
            a: int64
            ----
            a: [[-1,1,-1,3,-1,3]]
        """,
    "mode": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": [1, 1, 2, 3],
            ...     "b": [1, 1, 2, 2],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_mode(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").mode()).sort("a").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_mode`:

            >>> agnostic_mode(df_pd)
               a
            0  1

            >>> agnostic_mode(df_pl)
            shape: (1, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            └─────┘

            >>> agnostic_mode(df_pa)
            pyarrow.Table
            a: int64
            ----
            a: [[1]]
        """,
    "is_finite": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [float("nan"), float("inf"), 2.0, None]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_is_finite(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").is_finite()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_finite`:

            >>> agnostic_is_finite(df_pd)
                   a
            0  False
            1  False
            2   True
            3  False

            >>> agnostic_is_finite(df_pl)
            shape: (4, 1)
            ┌───────┐
            │ a     │
            │ ---   │
            │ bool  │
            ╞═══════╡
            │ false │
            │ false │
            │ true  │
            │ null  │
            └───────┘

            >>> agnostic_is_finite(df_pa)
            pyarrow.Table
            a: bool
            ----
            a: [[false,false,true,null]]
        """,
    "cum_count": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": ["x", "k", None, "d"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_cum_count(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("a").cum_count().alias("cum_count"),
            ...         nw.col("a").cum_count(reverse=True).alias("cum_count_reverse"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_cum_count`:

            >>> agnostic_cum_count(df_pd)
                  a  cum_count  cum_count_reverse
            0     x          1                  3
            1     k          2                  2
            2  None          2                  1
            3     d          3                  1

            >>> agnostic_cum_count(df_pl)
            shape: (4, 3)
            ┌──────┬───────────┬───────────────────┐
            │ a    ┆ cum_count ┆ cum_count_reverse │
            │ ---  ┆ ---       ┆ ---               │
            │ str  ┆ u32       ┆ u32               │
            ╞══════╪═══════════╪═══════════════════╡
            │ x    ┆ 1         ┆ 3                 │
            │ k    ┆ 2         ┆ 2                 │
            │ null ┆ 2         ┆ 1                 │
            │ d    ┆ 3         ┆ 1                 │
            └──────┴───────────┴───────────────────┘

            >>> agnostic_cum_count(df_pa)
            pyarrow.Table
            a: string
            cum_count: uint32
            cum_count_reverse: uint32
            ----
            a: [["x","k",null,"d"]]
            cum_count: [[1,2,2,3]]
            cum_count_reverse: [[3,2,1,1]]
        """,
    "cum_min": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [3, 1, None, 2]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_cum_min(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("a").cum_min().alias("cum_min"),
            ...         nw.col("a").cum_min(reverse=True).alias("cum_min_reverse"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_cum_min`:

            >>> agnostic_cum_min(df_pd)
                 a  cum_min  cum_min_reverse
            0  3.0      3.0              1.0
            1  1.0      1.0              1.0
            2  NaN      NaN              NaN
            3  2.0      1.0              2.0

            >>> agnostic_cum_min(df_pl)
            shape: (4, 3)
            ┌──────┬─────────┬─────────────────┐
            │ a    ┆ cum_min ┆ cum_min_reverse │
            │ ---  ┆ ---     ┆ ---             │
            │ i64  ┆ i64     ┆ i64             │
            ╞══════╪═════════╪═════════════════╡
            │ 3    ┆ 3       ┆ 1               │
            │ 1    ┆ 1       ┆ 1               │
            │ null ┆ null    ┆ null            │
            │ 2    ┆ 1       ┆ 2               │
            └──────┴─────────┴─────────────────┘

            >>> agnostic_cum_min(df_pa)
            pyarrow.Table
            a: int64
            cum_min: int64
            cum_min_reverse: int64
            ----
            a: [[3,1,null,2]]
            cum_min: [[3,1,null,1]]
            cum_min_reverse: [[1,1,null,2]]
        """,
    "cum_max": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 3, None, 2]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_cum_max(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("a").cum_max().alias("cum_max"),
            ...         nw.col("a").cum_max(reverse=True).alias("cum_max_reverse"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_`:

            >>> agnostic_cum_max(df_pd)
                 a  cum_max  cum_max_reverse
            0  1.0      1.0              3.0
            1  3.0      3.0              3.0
            2  NaN      NaN              NaN
            3  2.0      3.0              2.0

            >>> agnostic_cum_max(df_pl)
            shape: (4, 3)
            ┌──────┬─────────┬─────────────────┐
            │ a    ┆ cum_max ┆ cum_max_reverse │
            │ ---  ┆ ---     ┆ ---             │
            │ i64  ┆ i64     ┆ i64             │
            ╞══════╪═════════╪═════════════════╡
            │ 1    ┆ 1       ┆ 3               │
            │ 3    ┆ 3       ┆ 3               │
            │ null ┆ null    ┆ null            │
            │ 2    ┆ 3       ┆ 2               │
            └──────┴─────────┴─────────────────┘

            >>> agnostic_cum_max(df_pa)
            pyarrow.Table
            a: int64
            cum_max: int64
            cum_max_reverse: int64
            ----
            a: [[1,3,null,2]]
            cum_max: [[1,3,null,3]]
            cum_max_reverse: [[3,3,null,2]]
        """,
    "cum_prod": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 3, None, 2]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_cum_prod(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("a").cum_prod().alias("cum_prod"),
            ...         nw.col("a").cum_prod(reverse=True).alias("cum_prod_reverse"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_cum_prod`:

            >>> agnostic_cum_prod(df_pd)
                 a  cum_prod  cum_prod_reverse
            0  1.0       1.0               6.0
            1  3.0       3.0               6.0
            2  NaN       NaN               NaN
            3  2.0       6.0               2.0

            >>> agnostic_cum_prod(df_pl)
            shape: (4, 3)
            ┌──────┬──────────┬──────────────────┐
            │ a    ┆ cum_prod ┆ cum_prod_reverse │
            │ ---  ┆ ---      ┆ ---              │
            │ i64  ┆ i64      ┆ i64              │
            ╞══════╪══════════╪══════════════════╡
            │ 1    ┆ 1        ┆ 6                │
            │ 3    ┆ 3        ┆ 6                │
            │ null ┆ null     ┆ null             │
            │ 2    ┆ 6        ┆ 2                │
            └──────┴──────────┴──────────────────┘

            >>> agnostic_cum_prod(df_pa)
            pyarrow.Table
            a: int64
            cum_prod: int64
            cum_prod_reverse: int64
            ----
            a: [[1,3,null,2]]
            cum_prod: [[1,3,null,6]]
            cum_prod_reverse: [[6,6,null,2]]
        """,
    "rolling_sum": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1.0, 2.0, None, 4.0]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_rolling_sum(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         b=nw.col("a").rolling_sum(window_size=3, min_periods=1)
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_rolling_sum`:

            >>> agnostic_rolling_sum(df_pd)
                 a    b
            0  1.0  1.0
            1  2.0  3.0
            2  NaN  3.0
            3  4.0  6.0

            >>> agnostic_rolling_sum(df_pl)
            shape: (4, 2)
            ┌──────┬─────┐
            │ a    ┆ b   │
            │ ---  ┆ --- │
            │ f64  ┆ f64 │
            ╞══════╪═════╡
            │ 1.0  ┆ 1.0 │
            │ 2.0  ┆ 3.0 │
            │ null ┆ 3.0 │
            │ 4.0  ┆ 6.0 │
            └──────┴─────┘

            >>> agnostic_rolling_sum(df_pa)
            pyarrow.Table
            a: double
            b: double
            ----
            a: [[1,2,null,4]]
            b: [[1,3,3,6]]
        """,
    "rolling_mean": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1.0, 2.0, None, 4.0]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_rolling_mean(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         b=nw.col("a").rolling_mean(window_size=3, min_periods=1)
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_rolling_mean`:

            >>> agnostic_rolling_mean(df_pd)
                 a    b
            0  1.0  1.0
            1  2.0  1.5
            2  NaN  1.5
            3  4.0  3.0

            >>> agnostic_rolling_mean(df_pl)
            shape: (4, 2)
            ┌──────┬─────┐
            │ a    ┆ b   │
            │ ---  ┆ --- │
            │ f64  ┆ f64 │
            ╞══════╪═════╡
            │ 1.0  ┆ 1.0 │
            │ 2.0  ┆ 1.5 │
            │ null ┆ 1.5 │
            │ 4.0  ┆ 3.0 │
            └──────┴─────┘

            >>> agnostic_rolling_mean(df_pa)
            pyarrow.Table
            a: double
            b: double
            ----
            a: [[1,2,null,4]]
            b: [[1,1.5,1.5,3]]
        """,
    "rolling_var": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1.0, 2.0, None, 4.0]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_rolling_var(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         b=nw.col("a").rolling_var(window_size=3, min_periods=1)
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_rolling_var`:

            >>> agnostic_rolling_var(df_pd)
                 a    b
            0  1.0  NaN
            1  2.0  0.5
            2  NaN  0.5
            3  4.0  2.0

            >>> agnostic_rolling_var(df_pl)  #  doctest:+SKIP
            shape: (4, 2)
            ┌──────┬──────┐
            │ a    ┆ b    │
            │ ---  ┆ ---  │
            │ f64  ┆ f64  │
            ╞══════╪══════╡
            │ 1.0  ┆ null │
            │ 2.0  ┆ 0.5  │
            │ null ┆ 0.5  │
            │ 4.0  ┆ 2.0  │
            └──────┴──────┘

            >>> agnostic_rolling_var(df_pa)
            pyarrow.Table
            a: double
            b: double
            ----
            a: [[1,2,null,4]]
            b: [[nan,0.5,0.5,2]]
        """,
    "rolling_std": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1.0, 2.0, None, 4.0]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_rolling_std(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         b=nw.col("a").rolling_std(window_size=3, min_periods=1)
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_rolling_std`:

            >>> agnostic_rolling_std(df_pd)
                 a         b
            0  1.0       NaN
            1  2.0  0.707107
            2  NaN  0.707107
            3  4.0  1.414214

            >>> agnostic_rolling_std(df_pl)  #  doctest:+SKIP
            shape: (4, 2)
            ┌──────┬──────────┐
            │ a    ┆ b        │
            │ ---  ┆ ---      │
            │ f64  ┆ f64      │
            ╞══════╪══════════╡
            │ 1.0  ┆ null     │
            │ 2.0  ┆ 0.707107 │
            │ null ┆ 0.707107 │
            │ 4.0  ┆ 1.414214 │
            └──────┴──────────┘

            >>> agnostic_rolling_std(df_pa)
            pyarrow.Table
            a: double
            b: double
            ----
            a: [[1,2,null,4]]
            b: [[nan,0.7071067811865476,0.7071067811865476,1.4142135623730951]]
        """,
    "rank": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [3, 6, 1, 1, 6]}

            We define a dataframe-agnostic function that computes the dense rank for
            the data:

            >>> def agnostic_dense_rank(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     result = df.with_columns(rnk=nw.col("a").rank(method="dense"))
            ...     return result.to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dense_rank`:

            >>> agnostic_dense_rank(pd.DataFrame(data))
               a  rnk
            0  3  2.0
            1  6  3.0
            2  1  1.0
            3  1  1.0
            4  6  3.0

            >>> agnostic_dense_rank(pl.DataFrame(data))
            shape: (5, 2)
            ┌─────┬─────┐
            │ a   ┆ rnk │
            │ --- ┆ --- │
            │ i64 ┆ u32 │
            ╞═════╪═════╡
            │ 3   ┆ 2   │
            │ 6   ┆ 3   │
            │ 1   ┆ 1   │
            │ 1   ┆ 1   │
            │ 6   ┆ 3   │
            └─────┴─────┘

            >>> agnostic_dense_rank(pa.table(data))
            pyarrow.Table
            a: int64
            rnk: uint64
            ----
            a: [[3,6,1,1,6]]
            rnk: [[2,3,1,1,3]]
        """,
}
