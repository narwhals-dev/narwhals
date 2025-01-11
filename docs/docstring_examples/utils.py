from __future__ import annotations

EXAMPLES = {
    "is_pandas": """
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_pandas()
            True
        """,
    "is_pandas_like": """
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_pandas_like()
            True
        """,
    "is_polars": """
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_polars()
            True
        """,
    "is_cudf": """
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_cudf()
            False
        """,
    "is_modin": """
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_modin()
            False
        """,
    "is_pyspark": """
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_pyspark()
            False
        """,
    "is_pyarrow": """
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_pyarrow()
            False
        """,
    "is_dask": """
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_dask()
            False
        """,
    "is_duckdb": """
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_duckdb()
            False
        """,
    "is_ibis": """
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_ibis()
            False
        """,
    "maybe_align_index": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2]}, index=[3, 4])
        >>> s_pd = pd.Series([6, 7], index=[4, 3])
        >>> df = nw.from_native(df_pd)
        >>> s = nw.from_native(s_pd, series_only=True)
        >>> nw.to_native(nw.maybe_align_index(df, s))
           a
        4  2
        3  1
    """,
    "maybe_get_index": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [4, 5]})
        >>> df = nw.from_native(df_pd)
        >>> nw.maybe_get_index(df)
        RangeIndex(start=0, stop=2, step=1)
        >>> series_pd = pd.Series([1, 2])
        >>> series = nw.from_native(series_pd, series_only=True)
        >>> nw.maybe_get_index(series)
        RangeIndex(start=0, stop=2, step=1)
    """,
    "maybe_set_index": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [4, 5]})
        >>> df = nw.from_native(df_pd)
        >>> nw.to_native(nw.maybe_set_index(df, "b"))  # doctest: +NORMALIZE_WHITESPACE
           a
        b
        4  1
        5  2
    """,
    "maybe_reset_index": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [4, 5]}, index=([6, 7]))
        >>> df = nw.from_native(df_pd)
        >>> nw.to_native(nw.maybe_reset_index(df))
           a  b
        0  1  4
        1  2  5
        >>> series_pd = pd.Series([1, 2])
        >>> series = nw.from_native(series_pd, series_only=True)
        >>> nw.maybe_get_index(series)
        RangeIndex(start=0, stop=2, step=1)
    """,
    "maybe_convert_dtypes": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> import numpy as np
        >>> df_pd = pd.DataFrame(
        ...     {
        ...         "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
        ...         "b": pd.Series([True, False, np.nan], dtype=np.dtype("O")),
        ...     }
        ... )
        >>> df = nw.from_native(df_pd)
        >>> nw.to_native(nw.maybe_convert_dtypes(df)).dtypes  # doctest: +NORMALIZE_WHITESPACE
        a             Int32
        b           boolean
        dtype: object
    """,
    "is_ordered_categorical": """
        >>> import narwhals as nw
        >>> import pandas as pd
        >>> import polars as pl
        >>> data = ["x", "y"]
        >>> s_pd = pd.Series(data, dtype=pd.CategoricalDtype(ordered=True))
        >>> s_pl = pl.Series(data, dtype=pl.Categorical(ordering="physical"))

        Let's define a library-agnostic function:

        >>> @nw.narwhalify
        ... def func(s):
        ...     return nw.is_ordered_categorical(s)

        Then, we can pass any supported library to `func`:

        >>> func(s_pd)
        True
        >>> func(s_pl)
        True
    """,
    "generate_temporary_column_name": """
        >>> import narwhals as nw
        >>> columns = ["abc", "xyz"]
        >>> nw.generate_temporary_column_name(n_bytes=8, columns=columns) not in columns
        True
    """,
}
