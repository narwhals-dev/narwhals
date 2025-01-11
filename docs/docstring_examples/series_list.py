from __future__ import annotations

EXAMPLES = {
    "len": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [[1, 2], [3, 4, None], None, []]

            Let's define a dataframe-agnostic function:

            >>> def agnostic_list_len(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.list.len().to_native()

            We can then pass pandas / PyArrow / Polars / any other supported library:

            >>> agnostic_list_len(
            ...     pd.Series(data, dtype=pd.ArrowDtype(pa.list_(pa.int64())))
            ... )  # doctest: +SKIP
            0       2
            1       3
            2    <NA>
            3       0
            dtype: int32[pyarrow]

            >>> agnostic_list_len(pl.Series(data))  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [u32]
            [
               2
               3
               null
               0
            ]

            >>> agnostic_list_len(pa.chunked_array([data]))  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2,
                3,
                null,
                0
              ]
            ]
        """,
}
