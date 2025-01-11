from __future__ import annotations

EXAMPLES = {
    "implementation": """
            >>> import narwhals as nw
            >>> import pandas as pd

            >>> s_native = pd.Series([1, 2, 3])
            >>> s = nw.from_native(s_native, series_only=True)

            >>> s.implementation
            <Implementation.PANDAS: 1>

            >>> s.implementation.is_pandas()
            True

            >>> s.implementation.is_pandas_like()
            True

            >>> s.implementation.is_polars()
            False
        """,
    "__getitem__": """
            >>> from typing import Any
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_get_first_item(s_native: IntoSeriesT) -> Any:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s[0]

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_get_first_item`:

            >>> agnostic_get_first_item(s_pd)
            np.int64(1)

            >>> agnostic_get_first_item(s_pl)
            1

            >>> agnostic_get_first_item(s_pa)
            1

            We can also make a function to slice the Series:

            >>> def agnostic_slice(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s[:2].to_native()

            >>> agnostic_slice(s_pd)
            0    1
            1    2
            dtype: int64

            >>> agnostic_slice(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
                1
                2
            ]

            >>> agnostic_slice(s_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                2
              ]
            ]
        """,
    "to_native": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_to_native(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_to_native`:

            >>> agnostic_to_native(s_pd)
            0    1
            1    2
            2    3
            dtype: int64

            >>> agnostic_to_native(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
                1
                2
                3
            ]

            >>> agnostic_to_native(s_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                2,
                3
              ]
            ]
        """,
    "scatter": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT

            >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_scatter(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(df["a"].scatter([0, 1], [999, 888])).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_scatter`:

            >>> agnostic_scatter(df_pd)
                 a  b
            0  999  4
            1  888  5
            2    3  6

            >>> agnostic_scatter(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 999 ┆ 4   │
            │ 888 ┆ 5   │
            │ 3   ┆ 6   │
            └─────┴─────┘

            >>> agnostic_scatter(df_pa)
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[999,888,3]]
            b: [[4,5,6]]
        """,
    "shape": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_shape(s_native: IntoSeries) -> tuple[int]:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.shape

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_shape`:

            >>> agnostic_shape(s_pd)
            (3,)

            >>> agnostic_shape(s_pl)
            (3,)

            >>> agnostic_shape(s_pa)
            (3,)
        """,
    "pipe": """
            >>> import polars as pl
            >>> import pandas as pd
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a function to pipe into:

            >>> def agnostic_pipe(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.pipe(lambda x: x + 2).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_pipe`:

            >>> agnostic_pipe(s_pd)
            0    3
            1    4
            2    5
            dtype: int64

            >>> agnostic_pipe(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               3
               4
               5
            ]

            >>> agnostic_pipe(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                3,
                4,
                5
              ]
            ]
        """,
    "len": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function that computes the len of the series:

            >>> def agnostic_len(s_native: IntoSeries) -> int:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.len()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_len`:

            >>> agnostic_len(s_pd)
            3

            >>> agnostic_len(s_pl)
            3

            >>> agnostic_len(s_pa)
            3
        """,
    "dtype": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_dtype(s_native: IntoSeriesT) -> nw.dtypes.DType:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dtype

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dtype`:

            >>> agnostic_dtype(s_pd)
            Int64

            >>> agnostic_dtype(s_pl)
            Int64

            >>> agnostic_dtype(s_pa)
            Int64
        """,
    "name": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data, name="foo")
            >>> s_pl = pl.Series("foo", data)

            We define a library agnostic function:

            >>> def agnostic_name(s_native: IntoSeries) -> str:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.name

            We can then pass any supported library such as pandas or Polars
            to `agnostic_name`:

            >>> agnostic_name(s_pd)
            'foo'

            >>> agnostic_name(s_pl)
            'foo'
        """,
    "ewm_mean": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(name="a", data=data)
            >>> s_pl = pl.Series(name="a", values=data)

            We define a library agnostic function:

            >>> def agnostic_ewm_mean(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.ewm_mean(com=1, ignore_nulls=False).to_native()

            We can then pass any supported library such as pandas or Polars
            to `agnostic_ewm_mean`:

            >>> agnostic_ewm_mean(s_pd)
            0    1.000000
            1    1.666667
            2    2.428571
            Name: a, dtype: float64

            >>> agnostic_ewm_mean(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: 'a' [f64]
            [
               1.0
               1.666667
               2.428571
            ]
        """,
    "cast": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [True, False, True]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_cast(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.cast(nw.Int64).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_cast`:

            >>> agnostic_cast(s_pd)
            0    1
            1    0
            2    1
            dtype: int64

            >>> agnostic_cast(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               1
               0
               1
            ]

            >>> agnostic_cast(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                0,
                1
              ]
            ]
        """,
    "to_frame": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2]
            >>> s_pd = pd.Series(data, name="a")
            >>> s_pl = pl.Series("a", data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_to_frame(s_native: IntoSeries) -> IntoDataFrame:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.to_frame().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_to_frame`:

            >>> agnostic_to_frame(s_pd)
               a
            0  1
            1  2

            >>> agnostic_to_frame(s_pl)
            shape: (2, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            │ 2   │
            └─────┘

            >>> agnostic_to_frame(s_pa)
            pyarrow.Table
            : int64
            ----
            : [[1,2]]
        """,
    "to_list": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_to_list(s_native: IntoSeries):
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.to_list()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_to_list`:

            >>> agnostic_to_list(s_pd)
            [1, 2, 3]

            >>> agnostic_to_list(s_pl)
            [1, 2, 3]

            >>> agnostic_to_list(s_pa)
            [1, 2, 3]
        """,
    "mean": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_mean(s_native: IntoSeries) -> float:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.mean()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_mean`:

            >>> agnostic_mean(s_pd)
            np.float64(2.0)

            >>> agnostic_mean(s_pl)
            2.0

            >>> agnostic_mean(s_pa)
            2.0
        """,
    "median": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [5, 3, 8]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a library agnostic function:

            >>> def agnostic_median(s_native: IntoSeries) -> float:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.median()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_median`:

            >>> agnostic_median(s_pd)
            np.float64(5.0)

            >>> agnostic_median(s_pl)
            5.0

            >>> agnostic_median(s_pa)
            5.0
        """,
    "skew": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 1, 2, 10, 100]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_skew(s_native: IntoSeries) -> float:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.skew()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_skew`:

            >>> agnostic_skew(s_pd)
            np.float64(1.4724267269058975)

            >>> agnostic_skew(s_pl)
            1.4724267269058975

            >>> agnostic_skew(s_pa)
            1.4724267269058975

        Notes:
            The skewness is a measure of the asymmetry of the probability distribution.
            A perfectly symmetric distribution has a skewness of 0.
        """,
    "count": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_count(s_native: IntoSeries) -> int:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.count()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_count`:

            >>> agnostic_count(s_pd)
            np.int64(3)

            >>> agnostic_count(s_pl)
            3

            >>> agnostic_count(s_pa)
            3
        """,
    "any": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [False, True, False]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_any(s_native: IntoSeries) -> bool:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.any()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_any`:

            >>> agnostic_any(s_pd)
            np.True_

            >>> agnostic_any(s_pl)
            True

            >>> agnostic_any(s_pa)
            True
        """,
    "all": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [False, True, False]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_all(s_native: IntoSeries) -> bool:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.all()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_all`:

            >>> agnostic_all(s_pd)
            np.False_

            >>> agnostic_all(s_pl)
            False

            >>> agnostic_all(s_pa)
            False
        """,
    "min": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_min(s_native: IntoSeries):
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.min()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_min`:

            >>> agnostic_min(s_pd)
            np.int64(1)

            >>> agnostic_min(s_pl)
            1

            >>> agnostic_min(s_pa)
            1
        """,
    "max": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_max(s_native: IntoSeries):
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.max()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_max`:

            >>> agnostic_max(s_pd)
            np.int64(3)

            >>> agnostic_max(s_pl)
            3

            >>> agnostic_max(s_pa)
            3
        """,
    "arg_min": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_arg_min(s_native: IntoSeries):
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.arg_min()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_arg_min`:

            >>> agnostic_arg_min(s_pd)
            np.int64(0)

            >>> agnostic_arg_min(s_pl)
            0

            >>> agnostic_arg_min(s_pa)
            0
        """,
    "arg_max": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_arg_max(s_native: IntoSeries):
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.arg_max()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_arg_max`:

            >>> agnostic_arg_max(s_pd)
            np.int64(2)

            >>> agnostic_arg_max(s_pl)
            2

            >>> agnostic_arg_max(s_pa)
            2
        """,
    "sum": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_sum(s_native: IntoSeries):
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.sum()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_sum`:

            >>> agnostic_sum(s_pd)
            np.int64(6)

            >>> agnostic_sum(s_pl)
            6

            >>> agnostic_sum(s_pa)
            6
        """,
    "std": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_std(s_native: IntoSeries) -> float:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.std()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_std`:

            >>> agnostic_std(s_pd)
            np.float64(1.0)

            >>> agnostic_std(s_pl)
            1.0

            >>> agnostic_std(s_pa)
            1.0
        """,
    "var": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_var(s_native: IntoSeries) -> float:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.var()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_var`:

            >>> agnostic_var(s_pd)
            np.float64(1.0)

            >>> agnostic_var(s_pl)
            1.0

            >>> agnostic_var(s_pa)
            1.0
        """,
    "clip": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_clip_lower(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.clip(2).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_clip_lower`:

            >>> agnostic_clip_lower(s_pd)
            0    2
            1    2
            2    3
            dtype: int64

            >>> agnostic_clip_lower(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               2
               2
               3
            ]

            >>> agnostic_clip_lower(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2,
                2,
                3
              ]
            ]

            We define another library agnostic function:

            >>> def agnostic_clip_upper(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.clip(upper_bound=2).to_native()

           We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_clip_upper`:

            >>> agnostic_clip_upper(s_pd)
            0    1
            1    2
            2    2
            dtype: int64

            >>> agnostic_clip_upper(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               1
               2
               2
            ]

            >>> agnostic_clip_upper(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                2,
                2
              ]
            ]

            We can have both at the same time

            >>> data = [-1, 1, -3, 3, -5, 5]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_clip(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.clip(-1, 3).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_clip`:

            >>> agnostic_clip(s_pd)
            0   -1
            1    1
            2   -1
            3    3
            4   -1
            5    3
            dtype: int64

            >>> agnostic_clip(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (6,)
            Series: '' [i64]
            [
               -1
                1
               -1
                3
               -1
                3
            ]

            >>> agnostic_clip_upper(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                -1,
                1,
                -3,
                2,
                -5,
                2
              ]
            ]
        """,
    "is_in": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_is_in(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.is_in([3, 2, 8]).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_in`:

            >>> agnostic_is_in(s_pd)
            0    False
            1     True
            2     True
            dtype: bool

            >>> agnostic_is_in(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [bool]
            [
               false
               true
               true
            ]

            >>> agnostic_is_in(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                false,
                true,
                true
              ]
            ]
        """,
    "arg_true": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, None, None, 2]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_arg_true(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.is_null().arg_true().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_arg_true`:

            >>> agnostic_arg_true(s_pd)
            1    1
            2    2
            dtype: int64

            >>> agnostic_arg_true(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [u32]
            [
               1
               2
            ]

            >>> agnostic_arg_true(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                2
              ]
            ]
        """,
    "drop_nulls": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [2, 4, None, 3, 5]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_drop_nulls(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.drop_nulls().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_drop_nulls`:

            >>> agnostic_drop_nulls(s_pd)
            0    2.0
            1    4.0
            3    3.0
            4    5.0
            dtype: float64

            >>> agnostic_drop_nulls(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
                2
                4
                3
                5
            ]

            >>> agnostic_drop_nulls(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2,
                4,
                3,
                5
              ]
            ]
        """,
    "abs": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [2, -4, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_abs(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.abs().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_abs`:

            >>> agnostic_abs(s_pd)
            0    2
            1    4
            2    3
            dtype: int64

            >>> agnostic_abs(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               2
               4
               3
            ]

            >>> agnostic_abs(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2,
                4,
                3
              ]
            ]
        """,
    "cum_sum": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [2, 4, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_cum_sum(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.cum_sum().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_cum_sum`:

            >>> agnostic_cum_sum(s_pd)
            0    2
            1    6
            2    9
            dtype: int64

            >>> agnostic_cum_sum(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               2
               6
               9
            ]

            >>> agnostic_cum_sum(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2,
                6,
                9
              ]
            ]
        """,
    "unique": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [2, 4, 4, 6]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_unique(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.unique(maintain_order=True).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_unique`:

            >>> agnostic_unique(s_pd)
            0    2
            1    4
            2    6
            dtype: int64

            >>> agnostic_unique(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               2
               4
               6
            ]

            >>> agnostic_unique(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2,
                4,
                6
              ]
            ]
        """,
    "diff": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [2, 4, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_diff(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.diff().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_diff`:

            >>> agnostic_diff(s_pd)
            0    NaN
            1    2.0
            2   -1.0
            dtype: float64

            >>> agnostic_diff(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               null
               2
               -1
            ]

            >>> agnostic_diff(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                null,
                2,
                -1
              ]
            ]
        """,
    "shift": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [2, 4, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_shift(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.shift(1).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_shift`:

            >>> agnostic_shift(s_pd)
            0    NaN
            1    2.0
            2    4.0
            dtype: float64

            >>> agnostic_shift(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               null
               2
               4
            ]

            >>> agnostic_shift(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                null,
                2,
                4
              ]
            ]
        """,
    "sample": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3, 4]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_sample(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.sample(fraction=1.0, with_replacement=True).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_sample`:

            >>> agnostic_sample(s_pd)  # doctest: +SKIP
               a
            2  3
            1  2
            3  4
            3  4

            >>> agnostic_sample(s_pl)  # doctest: +SKIP
            shape: (4,)
            Series: '' [i64]
            [
               1
               4
               3
               4
            ]

            >>> agnostic_sample(s_pa)  # doctest: +SKIP
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                4,
                3,
                4
              ]
            ]
        """,
    "alias": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data, name="foo")
            >>> s_pl = pl.Series("foo", data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_alias(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.alias("bar").to_native()

            We can then pass any supported library such as pandas or Polars, or
            PyArrow to `agnostic_alias`:

            >>> agnostic_alias(s_pd)
            0    1
            1    2
            2    3
            Name: bar, dtype: int64

            >>> agnostic_alias(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: 'bar' [i64]
            [
               1
               2
               3
            ]

            >>> agnostic_alias(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at 0x...>
            [
              [
                1,
                2,
                3
              ]
            ]
        """,
    "rename": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data, name="foo")
            >>> s_pl = pl.Series("foo", data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_rename(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.rename("bar").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_rename`:

            >>> agnostic_rename(s_pd)
            0    1
            1    2
            2    3
            Name: bar, dtype: int64

            >>> agnostic_rename(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: 'bar' [i64]
            [
               1
               2
               3
            ]

            >>> agnostic_rename(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at 0x...>
            [
              [
                1,
                2,
                3
              ]
            ]
        """,
    "replace_strict": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = {"a": [3, 0, 1, 2]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define dataframe-agnostic functions:

            >>> def agnostic_replace_strict(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.replace_strict(
            ...         [0, 1, 2, 3], ["zero", "one", "two", "three"], return_dtype=nw.String
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_replace_strict`:

            >>> agnostic_replace_strict(df_pd["a"])
            0    three
            1     zero
            2      one
            3      two
            Name: a, dtype: object

            >>> agnostic_replace_strict(df_pl["a"])  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: 'a' [str]
            [
                "three"
                "zero"
                "one"
                "two"
            ]

            >>> agnostic_replace_strict(df_pa["a"])
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "three",
                "zero",
                "one",
                "two"
              ]
            ]
        """,
    "sort": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [5, None, 1, 2]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define library agnostic functions:

            >>> def agnostic_sort(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.sort().to_native()

            >>> def agnostic_sort_descending(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.sort(descending=True).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_sort` and `agnostic_sort_descending`:

            >>> agnostic_sort(s_pd)
            1    NaN
            2    1.0
            3    2.0
            0    5.0
            dtype: float64

            >>> agnostic_sort(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
               null
               1
               2
               5
            ]

            >>> agnostic_sort(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                null,
                1,
                2,
                5
              ]
            ]

            >>> agnostic_sort_descending(s_pd)
            1    NaN
            0    5.0
            3    2.0
            2    1.0
            dtype: float64

            >>> agnostic_sort_descending(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
               null
               5
               2
               1
            ]

            >>> agnostic_sort_descending(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                null,
                5,
                2,
                1
              ]
            ]
        """,
    "is_null": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_null(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.is_null().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_null`:

            >>> agnostic_is_null(s_pd)
            0    False
            1    False
            2     True
            dtype: bool

            >>> agnostic_is_null(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [bool]
            [
               false
               false
               true
            ]

            >>> agnostic_is_null(s_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                false,
                false,
                true
              ]
            ]
        """,
    "is_nan": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [0.0, None, 2.0]
            >>> s_pd = pd.Series(data, dtype="Float64")
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data], type=pa.float64())

            >>> def agnostic_self_div_is_nan(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.is_nan().to_native()

            >>> print(agnostic_self_div_is_nan(s_pd))
            0    False
            1     <NA>
            2    False
            dtype: boolean

            >>> print(agnostic_self_div_is_nan(s_pl))  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [bool]
            [
                    false
                    null
                    false
            ]

            >>> print(agnostic_self_div_is_nan(s_pa))  # doctest: +NORMALIZE_WHITESPACE
            [
              [
                false,
                null,
                false
              ]
            ]
        """,
    "fill_null": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_fill_null(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.fill_null(5).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_fill_null`:

            >>> agnostic_fill_null(s_pd)
            0    1.0
            1    2.0
            2    5.0
            dtype: float64

            >>> agnostic_fill_null(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               1
               2
               5
            ]

            >>> agnostic_fill_null(s_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                2,
                5
              ]
            ]

            Using a strategy:

            >>> def agnostic_fill_null_with_strategy(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.fill_null(strategy="forward", limit=1).to_native()

            >>> agnostic_fill_null_with_strategy(s_pd)
            0    1.0
            1    2.0
            2    2.0
            dtype: float64

            >>> agnostic_fill_null_with_strategy(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               1
               2
               2
            ]

            >>> agnostic_fill_null_with_strategy(s_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                2,
                2
              ]
            ]
        """,
    "is_between": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3, 4, 5]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_is_between(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.is_between(2, 4, "right").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_between`:

            >>> agnostic_is_between(s_pd)
            0    False
            1    False
            2     True
            3     True
            4    False
            dtype: bool

            >>> agnostic_is_between(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [bool]
            [
               false
               false
               true
               true
               false
            ]

            >>> agnostic_is_between(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                false,
                false,
                true,
                true,
                false
              ]
            ]
        """,
    "n_unique": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_n_unique(s_native: IntoSeries) -> int:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.n_unique()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_n_unique`:

            >>> agnostic_n_unique(s_pd)
            3

            >>> agnostic_n_unique(s_pl)
            3

            >>> agnostic_n_unique(s_pa)
            3
        """,
    "to_numpy": """
            >>> import numpy as np
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data, name="a")
            >>> s_pl = pl.Series("a", data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_to_numpy(s_native: IntoSeries) -> np.ndarray:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.to_numpy()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_to_numpy`:

            >>> agnostic_to_numpy(s_pd)
            array([1, 2, 3]...)

            >>> agnostic_to_numpy(s_pl)
            array([1, 2, 3]...)

            >>> agnostic_to_numpy(s_pa)
            array([1, 2, 3]...)
        """,
    "to_pandas": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data, name="a")
            >>> s_pl = pl.Series("a", data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_to_pandas(s_native: IntoSeries) -> pd.Series:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.to_pandas()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_to_pandas`:

            >>> agnostic_to_pandas(s_pd)
            0    1
            1    2
            2    3
            Name: a, dtype: int64

            >>> agnostic_to_pandas(s_pl)
            0    1
            1    2
            2    3
            Name: a, dtype: int64

            >>> agnostic_to_pandas(s_pa)
            0    1
            1    2
            2    3
            Name: , dtype: int64
        """,
    "filter": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [4, 10, 15, 34, 50]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_filter(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.filter(s > 10).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_filter`:

            >>> agnostic_filter(s_pd)
            2    15
            3    34
            4    50
            dtype: int64

            >>> agnostic_filter(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               15
               34
               50
            ]

            >>> agnostic_filter(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                15,
                34,
                50
              ]
            ]
        """,
    "is_duplicated": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3, 1]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_duplicated(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.is_duplicated().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_duplicated`:

            >>> agnostic_is_duplicated(s_pd)
            0     True
            1    False
            2    False
            3     True
            dtype: bool

            >>> agnostic_is_duplicated(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
                true
                false
                false
                true
            ]

            >>> agnostic_is_duplicated(s_pa)  # doctest: +ELLIPSIS
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
            >>> from narwhals.typing import IntoSeries

            Let's define a dataframe-agnostic function that filters rows in which "foo"
            values are greater than 10, and then checks if the result is empty or not:

            >>> def agnostic_is_empty(s_native: IntoSeries) -> bool:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.filter(s > 10).is_empty()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_empty`:

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])
            >>> agnostic_is_empty(s_pd), agnostic_is_empty(s_pl), agnostic_is_empty(s_pa)
            (True, True, True)

            >>> data = [100, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])
            >>> agnostic_is_empty(s_pd), agnostic_is_empty(s_pl), agnostic_is_empty(s_pa)
            (False, False, False)
        """,
    "is_unique": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3, 1]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_unique(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.is_unique().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_unique`:

            >>> agnostic_is_unique(s_pd)
            0    False
            1     True
            2     True
            3    False
            dtype: bool

            >>> agnostic_is_unique(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
                false
                 true
                 true
                false
            ]
            >>> agnostic_is_unique(s_pa)  # doctest: +ELLIPSIS
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
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, None, None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function that returns the null count of
            the series:

            >>> def agnostic_null_count(s_native: IntoSeries) -> int:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.null_count()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_null_count`:

            >>> agnostic_null_count(s_pd)
            np.int64(2)

            >>> agnostic_null_count(s_pl)
            2

            >>> agnostic_null_count(s_pa)
            2
        """,
    "is_first_distinct": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 1, 2, 3, 2]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_first_distinct(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.is_first_distinct().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_first_distinct`:

            >>> agnostic_is_first_distinct(s_pd)
            0     True
            1    False
            2     True
            3     True
            4    False
            dtype: bool

            >>> agnostic_is_first_distinct(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [bool]
            [
                true
                false
                true
                true
                false
            ]

            >>> agnostic_is_first_distinct(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                true,
                false,
                true,
                true,
                false
              ]
            ]
        """,
    "is_last_distinct": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 1, 2, 3, 2]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_last_distinct(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.is_last_distinct().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_last_distinct`:

            >>> agnostic_is_last_distinct(s_pd)
            0    False
            1     True
            2    False
            3     True
            4     True
            dtype: bool

            >>> agnostic_is_last_distinct(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [bool]
            [
                false
                true
                false
                true
                true
            ]

            >>> agnostic_is_last_distinct(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                false,
                true,
                false,
                true,
                true
              ]
            ]
        """,
    "is_sorted": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> unsorted_data = [1, 3, 2]
            >>> sorted_data = [3, 2, 1]

            Let's define a dataframe-agnostic function:

            >>> def agnostic_is_sorted(s_native: IntoSeries, descending: bool = False):
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.is_sorted(descending=descending)

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_sorted`:

            >>> agnostic_is_sorted(pd.Series(unsorted_data))
            False

            >>> agnostic_is_sorted(pd.Series(sorted_data), descending=True)
            True

            >>> agnostic_is_sorted(pl.Series(unsorted_data))
            False

            >>> agnostic_is_sorted(pl.Series(sorted_data), descending=True)
            True

            >>> agnostic_is_sorted(pa.chunked_array([unsorted_data]))
            False

            >>> agnostic_is_sorted(pa.chunked_array([sorted_data]), descending=True)
            True
        """,
    "value_counts": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 1, 2, 3, 2]
            >>> s_pd = pd.Series(data, name="s")
            >>> s_pl = pl.Series(values=data, name="s")
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_value_counts(s_native: IntoSeries) -> IntoDataFrame:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.value_counts(sort=True).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_value_counts`:

            >>> agnostic_value_counts(s_pd)
               s  count
            0  1      2
            1  2      2
            2  3      1

            >>> agnostic_value_counts(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3, 2)
            ┌─────┬───────┐
            │ s   ┆ count │
            │ --- ┆ ---   │
            │ i64 ┆ u32   │
            ╞═════╪═══════╡
            │ 1   ┆ 2     │
            │ 2   ┆ 2     │
            │ 3   ┆ 1     │
            └─────┴───────┘

            >>> agnostic_value_counts(s_pa)
            pyarrow.Table
            : int64
            count: int64
            ----
            : [[1,2,3]]
            count: [[2,2,1]]
        """,
    "quantile": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = list(range(50))
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_quantile(s_native: IntoSeries) -> list[float]:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return [
            ...         s.quantile(quantile=q, interpolation="nearest")
            ...         for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            ...     ]

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_quantile`:

            >>> agnostic_quantile(s_pd)
            [np.int64(5), np.int64(12), np.int64(24), np.int64(37), np.int64(44)]

            >>> agnostic_quantile(s_pl)
            [5.0, 12.0, 25.0, 37.0, 44.0]

            >>> agnostic_quantile(s_pa)
            [5, 12, 24, 37, 44]
        """,
    "zip_with": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3, 4, 5]
            >>> other = [5, 4, 3, 2, 1]
            >>> mask = [True, False, True, False, True]

            Let's define a dataframe-agnostic function:

            >>> def agnostic_zip_with(
            ...     s1_native: IntoSeriesT, mask_native: IntoSeriesT, s2_native: IntoSeriesT
            ... ) -> IntoSeriesT:
            ...     s1 = nw.from_native(s1_native, series_only=True)
            ...     mask = nw.from_native(mask_native, series_only=True)
            ...     s2 = nw.from_native(s2_native, series_only=True)
            ...     return s1.zip_with(mask, s2).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_zip_with`:

            >>> agnostic_zip_with(
            ...     s1_native=pl.Series(data),
            ...     mask_native=pl.Series(mask),
            ...     s2_native=pl.Series(other),
            ... )  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [i64]
            [
               1
               4
               3
               2
               5
            ]

            >>> agnostic_zip_with(
            ...     s1_native=pd.Series(data),
            ...     mask_native=pd.Series(mask),
            ...     s2_native=pd.Series(other),
            ... )
            0    1
            1    4
            2    3
            3    2
            4    5
            dtype: int64

            >>> agnostic_zip_with(
            ...     s1_native=pa.chunked_array([data]),
            ...     mask_native=pa.chunked_array([mask]),
            ...     s2_native=pa.chunked_array([other]),
            ... )  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                4,
                3,
                2,
                5
              ]
            ]
        """,
    "item": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            Let's define a dataframe-agnostic function that returns item at given index

            >>> def agnostic_item(s_native: IntoSeries, index=None):
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.item(index)

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_item`:

            >>> (
            ...     agnostic_item(pl.Series("a", [1]), None),
            ...     agnostic_item(pd.Series([1]), None),
            ...     agnostic_item(pa.chunked_array([[1]]), None),
            ... )
            (1, np.int64(1), 1)

            >>> (
            ...     agnostic_item(pl.Series("a", [9, 8, 7]), -1),
            ...     agnostic_item(pl.Series([9, 8, 7]), -2),
            ...     agnostic_item(pa.chunked_array([[9, 8, 7]]), -3),
            ... )
            (7, 8, 9)
        """,
    "head": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = list(range(10))
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function that returns the first 3 rows:

            >>> def agnostic_head(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.head(3).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_head`:

            >>> agnostic_head(s_pd)
            0    0
            1    1
            2    2
            dtype: int64

            >>> agnostic_head(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               0
               1
               2
            ]

            >>> agnostic_head(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                0,
                1,
                2
              ]
            ]
        """,
    "tail": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = list(range(10))
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function that returns the last 3 rows:

            >>> def agnostic_tail(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.tail(3).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_tail`:

            >>> agnostic_tail(s_pd)
            7    7
            8    8
            9    9
            dtype: int64

            >>> agnostic_tail(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
               7
               8
               9
            ]

            >>> agnostic_tail(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                7,
                8,
                9
              ]
            ]
        """,
    "round": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1.12345, 2.56789, 3.901234]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function that rounds to the first decimal:

            >>> def agnostic_round(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.round(1).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_round`:

            >>> agnostic_round(s_pd)
            0    1.1
            1    2.6
            2    3.9
            dtype: float64

            >>> agnostic_round(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [f64]
            [
               1.1
               2.6
               3.9
            ]

            >>> agnostic_round(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1.1,
                2.6,
                3.9
              ]
            ]
        """,
    "to_dummies": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoDataFrame
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3]
            >>> s_pd = pd.Series(data, name="a")
            >>> s_pl = pl.Series("a", data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_to_dummies(
            ...     s_native: IntoSeries, drop_first: bool = False
            ... ) -> IntoDataFrame:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.to_dummies(drop_first=drop_first).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_to_dummies`:

            >>> agnostic_to_dummies(s_pd)
               a_1  a_2  a_3
            0    1    0    0
            1    0    1    0
            2    0    0    1

            >>> agnostic_to_dummies(s_pd, drop_first=True)
               a_2  a_3
            0    0    0
            1    1    0
            2    0    1

            >>> agnostic_to_dummies(s_pl)
            shape: (3, 3)
            ┌─────┬─────┬─────┐
            │ a_1 ┆ a_2 ┆ a_3 │
            │ --- ┆ --- ┆ --- │
            │ i8  ┆ i8  ┆ i8  │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 0   ┆ 0   │
            │ 0   ┆ 1   ┆ 0   │
            │ 0   ┆ 0   ┆ 1   │
            └─────┴─────┴─────┘

            >>> agnostic_to_dummies(s_pl, drop_first=True)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a_2 ┆ a_3 │
            │ --- ┆ --- │
            │ i8  ┆ i8  │
            ╞═════╪═════╡
            │ 0   ┆ 0   │
            │ 1   ┆ 0   │
            │ 0   ┆ 1   │
            └─────┴─────┘

            >>> agnostic_to_dummies(s_pa)
            pyarrow.Table
            _1: int8
            _2: int8
            _3: int8
            ----
            _1: [[1,0,0]]
            _2: [[0,1,0]]
            _3: [[0,0,1]]
            >>> agnostic_to_dummies(s_pa, drop_first=True)
            pyarrow.Table
            _2: int8
            _3: int8
            ----
            _2: [[0,1,0]]
            _3: [[0,0,1]]
        """,
    "gather_every": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 2, 3, 4]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function in which gather every 2 rows,
            starting from a offset of 1:

            >>> def agnostic_gather_every(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.gather_every(n=2, offset=1).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_gather_every`:

            >>> agnostic_gather_every(s_pd)
            1    2
            3    4
            dtype: int64

            >>> agnostic_gather_every(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
               2
               4
            ]

            >>> agnostic_gather_every(s_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2,
                4
              ]
            ]
        """,
    "to_arrow": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeries

            >>> data = [1, 2, 3, 4]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function that converts to arrow:

            >>> def agnostic_to_arrow(s_native: IntoSeries) -> pa.Array:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.to_arrow()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_to_arrow`:

            >>> agnostic_to_arrow(s_pd)  # doctest:+NORMALIZE_WHITESPACE
            <pyarrow.lib.Int64Array object at ...>
            [
                1,
                2,
                3,
                4
            ]

            >>> agnostic_to_arrow(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            <pyarrow.lib.Int64Array object at ...>
            [
                1,
                2,
                3,
                4
            ]

            >>> agnostic_to_arrow(s_pa)  # doctest:+NORMALIZE_WHITESPACE
            <pyarrow.lib.Int64Array object at ...>
            [
                1,
                2,
                3,
                4
            ]
        """,
    "mode": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 1, 2, 2, 3]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_mode(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.mode().sort().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_mode`:

            >>> agnostic_mode(s_pd)
            0    1
            1    2
            dtype: int64

            >>> agnostic_mode(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
               1
               2
            ]

            >>> agnostic_mode(s_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                2
              ]
            ]
        """,
    "is_finite": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [float("nan"), float("inf"), 2.0, None]

            We define a library agnostic function:

            >>> def agnostic_is_finite(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.is_finite().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_is_finite`:

            >>> agnostic_is_finite(pd.Series(data))
            0    False
            1    False
            2     True
            3    False
            dtype: bool

            >>> agnostic_is_finite(pl.Series(data))  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
               false
               false
               true
               null
            ]

            >>> agnostic_is_finite(pa.chunked_array([data]))  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                false,
                false,
                true,
                null
              ]
            ]
        """,
    "cum_count": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["x", "k", None, "d"]

            We define a library agnostic function:

            >>> def agnostic_cum_count(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.cum_count(reverse=True).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_cum_count`:

            >>> agnostic_cum_count(pd.Series(data))
            0    3
            1    2
            2    1
            3    1
            dtype: int64

            >>> agnostic_cum_count(pl.Series(data))  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [u32]
            [
                3
                2
                1
                1
            ]

            >>> agnostic_cum_count(pa.chunked_array([data]))  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                3,
                2,
                1,
                1
              ]
            ]
        """,
    "cum_min": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [3, 1, None, 2]

            We define a library agnostic function:

            >>> def agnostic_cum_min(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.cum_min().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_cum_min`:

            >>> agnostic_cum_min(pd.Series(data))
            0    3.0
            1    1.0
            2    NaN
            3    1.0
            dtype: float64

            >>> agnostic_cum_min(pl.Series(data))  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
               3
               1
               null
               1
            ]

            >>> agnostic_cum_min(pa.chunked_array([data]))  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                3,
                1,
                null,
                1
              ]
            ]
        """,
    "cum_max": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 3, None, 2]

            We define a library agnostic function:

            >>> def agnostic_cum_max(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.cum_max().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_cum_max`:

            >>> agnostic_cum_max(pd.Series(data))
            0    1.0
            1    3.0
            2    NaN
            3    3.0
            dtype: float64

            >>> agnostic_cum_max(pl.Series(data))  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
               1
               3
               null
               3
            ]

            >>> agnostic_cum_max(pa.chunked_array([data]))  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                3,
                null,
                3
              ]
            ]
        """,
    "cum_prod": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1, 3, None, 2]

            We define a library agnostic function:

            >>> def agnostic_cum_prod(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.cum_prod().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_cum_prod`:

            >>> agnostic_cum_prod(pd.Series(data))
            0    1.0
            1    3.0
            2    NaN
            3    6.0
            dtype: float64

            >>> agnostic_cum_prod(pl.Series(data))  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
               1
               3
               null
               6
            ]

            >>> agnostic_cum_prod(pa.chunked_array([data]))  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                3,
                null,
                6
              ]
            ]
        """,
    "rolling_sum": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1.0, 2.0, 3.0, 4.0]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_rolling_sum(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.rolling_sum(window_size=2).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_rolling_sum`:

            >>> agnostic_rolling_sum(s_pd)
            0    NaN
            1    3.0
            2    5.0
            3    7.0
            dtype: float64

            >>> agnostic_rolling_sum(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [f64]
            [
               null
               3.0
               5.0
               7.0
            ]

            >>> agnostic_rolling_sum(s_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                null,
                3,
                5,
                7
              ]
            ]
        """,
    "rolling_mean": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1.0, 2.0, 3.0, 4.0]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_rolling_mean(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.rolling_mean(window_size=2).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_rolling_mean`:

            >>> agnostic_rolling_mean(s_pd)
            0    NaN
            1    1.5
            2    2.5
            3    3.5
            dtype: float64

            >>> agnostic_rolling_mean(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [f64]
            [
               null
               1.5
               2.5
               3.5
            ]

            >>> agnostic_rolling_mean(s_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                null,
                1.5,
                2.5,
                3.5
              ]
            ]
        """,
    "rolling_var": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1.0, 3.0, 1.0, 4.0]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_rolling_var(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.rolling_var(window_size=2, min_periods=1).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_rolling_var`:

            >>> agnostic_rolling_var(s_pd)
            0    NaN
            1    2.0
            2    2.0
            3    4.5
            dtype: float64

            >>> agnostic_rolling_var(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [f64]
            [
               null
               2.0
               2.0
               4.5
            ]

            >>> agnostic_rolling_var(s_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                nan,
                2,
                2,
                4.5
              ]
            ]
        """,
    "rolling_std": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [1.0, 3.0, 1.0, 4.0]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_rolling_std(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.rolling_std(window_size=2, min_periods=1).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_rolling_std`:

            >>> agnostic_rolling_std(s_pd)
            0         NaN
            1    1.414214
            2    1.414214
            3    2.121320
            dtype: float64

            >>> agnostic_rolling_std(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [f64]
            [
               null
               1.414214
               1.414214
               2.12132
            ]

            >>> agnostic_rolling_std(s_pa)  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                nan,
                1.4142135623730951,
                1.4142135623730951,
                2.1213203435596424
              ]
            ]
        """,
    "rank": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT
            >>>
            >>> data = [3, 6, 1, 1, 6]

            We define a dataframe-agnostic function that computes the dense rank for
            the data:

            >>> def agnostic_dense_rank(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.rank(method="dense").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dense_rank`:

            >>> agnostic_dense_rank(pd.Series(data))
            0    2.0
            1    3.0
            2    1.0
            3    1.0
            4    3.0
            dtype: float64

            >>> agnostic_dense_rank(pl.Series(data))  # doctest:+NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [u32]
            [
               2
               3
               1
               1
               3
            ]

            >>> agnostic_dense_rank(pa.chunked_array([data]))  # doctest:+ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2,
                3,
                1,
                1,
                3
              ]
            ]
        """,
}
