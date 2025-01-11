from __future__ import annotations

EXAMPLES = {
    "get_categories": """
            Let's create some series:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["apple", "mango", "mango"]
            >>> s_pd = pd.Series(data, dtype="category")
            >>> s_pl = pl.Series(data, dtype=pl.Categorical)
            >>> s_pa = pa.chunked_array([data]).dictionary_encode()

            We define a dataframe-agnostic function to get unique categories
            from column 'fruits':

            >>> def agnostic_get_categories(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.cat.get_categories().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_get_categories`:

            >>> agnostic_get_categories(s_pd)
            0    apple
            1    mango
            dtype: object

            >>> agnostic_get_categories(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [str]
            [
               "apple"
               "mango"
            ]

            >>> agnostic_get_categories(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "apple",
                "mango"
              ]
            ]
        """,
}
