from __future__ import annotations

EXAMPLES = {
    "len_chars": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["foo", "Café", "345", "東京", None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_len_chars(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.str.len_chars().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_len_chars`:

            >>> agnostic_len_chars(s_pd)
            0    3.0
            1    4.0
            2    3.0
            3    2.0
            4    NaN
            dtype: float64

            >>> agnostic_len_chars(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [u32]
            [
               3
               4
               3
               2
               null
            ]

            >>> agnostic_len_chars(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                3,
                4,
                3,
                2,
                null
              ]
            ]
        """,
    "replace": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["123abc", "abc abc123"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_replace(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     s = s.str.replace("abc", "")
            ...     return s.to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_replace`:

            >>> agnostic_replace(s_pd)
            0        123
            1     abc123
            dtype: object

            >>> agnostic_replace(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [str]
            [
                "123"
                " abc123"
            ]

            >>> agnostic_replace(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "123",
                " abc123"
              ]
            ]
        """,
    "replace_all": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["123abc", "abc abc123"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_replace_all(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     s = s.str.replace_all("abc", "")
            ...     return s.to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_replace_all`:

            >>> agnostic_replace_all(s_pd)
            0     123
            1     123
            dtype: object

            >>> agnostic_replace_all(s_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [str]
            [
                "123"
                " 123"
            ]

            >>> agnostic_replace_all(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "123",
                " 123"
              ]
            ]
        """,
    "strip_chars": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["apple", "\\nmango"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_strip_chars(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     s = s.str.strip_chars()
            ...     return s.to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_strip_chars`:

            >>> agnostic_strip_chars(s_pd)
            0    apple
            1    mango
            dtype: object

            >>> agnostic_strip_chars(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [str]
            [
                "apple"
                "mango"
            ]

            >>> agnostic_strip_chars(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "apple",
                "mango"
              ]
            ]
        """,
    "starts_with": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["apple", "mango", None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_starts_with(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.str.starts_with("app").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_starts_with`:

            >>> agnostic_starts_with(s_pd)
            0     True
            1    False
            2     None
            dtype: object

            >>> agnostic_starts_with(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [bool]
            [
               true
               false
               null
            ]

            >>> agnostic_starts_with(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                true,
                false,
                null
              ]
            ]
        """,
    "ends_with": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["apple", "mango", None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_ends_with(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.str.ends_with("ngo").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_ends_with`:

            >>> agnostic_ends_with(s_pd)
            0    False
            1     True
            2     None
            dtype: object

            >>> agnostic_ends_with(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [bool]
            [
               false
               true
               null
            ]

            >>> agnostic_ends_with(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                false,
                true,
                null
              ]
            ]
        """,
    "contains": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["cat", "dog", "rabbit and parrot", "dove", None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_contains(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.str.contains("parrot|dove").to_native()

            We can then pass any supported library such as pandas, Polars, or PyArrow to `agnostic_contains`:

            >>> agnostic_contains(s_pd)
            0    False
            1    False
            2     True
            3     True
            4     None
            dtype: object

            >>> agnostic_contains(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: '' [bool]
            [
               false
               false
               true
               true
               null
            ]

            >>> agnostic_contains(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                false,
                false,
                true,
                true,
                null
              ]
            ]
        """,
    "slice": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["pear", None, "papaya", "dragonfruit"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_slice(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.str.slice(4, length=3).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_slice`:

            >>> agnostic_slice(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            0
            1    None
            2      ya
            3     onf
            dtype: object

            >>> agnostic_slice(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [str]
            [
               ""
               null
               "ya"
               "onf"
            ]

            >>> agnostic_slice(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "",
                null,
                "ya",
                "onf"
              ]
            ]

            Using negative indexes:

            >>> def agnostic_slice(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.str.slice(-3).to_native()

            >>> agnostic_slice(s_pd)  # doctest: +NORMALIZE_WHITESPACE
            0     ear
            1    None
            2     aya
            3     uit
            dtype: object

            >>> agnostic_slice(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [str]
            [
                "ear"
                null
                "aya"
                "uit"
            ]

            >>> agnostic_slice(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "ear",
                null,
                "aya",
                "uit"
              ]
            ]
        """,
    "head": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["Atatata", "taata", "taatatata", "zukkyun"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_head(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.str.head().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_head`:

            >>> agnostic_head(s_pd)
            0    Atata
            1    taata
            2    taata
            3    zukky
            dtype: object

            >>> agnostic_head(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [str]
            [
               "Atata"
               "taata"
               "taata"
               "zukky"
            ]

            >>> agnostic_head(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "Atata",
                "taata",
                "taata",
                "zukky"
              ]
            ]
        """,
    "tail": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["Atatata", "taata", "taatatata", "zukkyun"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_tail(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.str.tail().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_tail`:

            >>> agnostic_tail(s_pd)
            0    atata
            1    taata
            2    atata
            3    kkyun
            dtype: object

            >>> agnostic_tail(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [str]
            [
               "atata"
               "taata"
               "atata"
               "kkyun"
            ]

            >>> agnostic_tail(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "atata",
                "taata",
                "atata",
                "kkyun"
              ]
            ]
        """,
    "to_uppercase": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["apple", "mango", None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_to_uppercase(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.str.to_uppercase().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_to_uppercase`:

            >>> agnostic_to_uppercase(s_pd)
            0    APPLE
            1    MANGO
            2     None
            dtype: object

            >>> agnostic_to_uppercase(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [str]
            [
               "APPLE"
               "MANGO"
               null
            ]

            >>> agnostic_to_uppercase(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "APPLE",
                "MANGO",
                null
              ]
            ]
        """,
    "to_lowercase": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["APPLE", "MANGO", None]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_to_lowercase(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.str.to_lowercase().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_to_lowercase`:

            >>> agnostic_to_lowercase(s_pd)
            0    apple
            1    mango
            2     None
            dtype: object

            >>> agnostic_to_lowercase(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [str]
            [
               "apple"
               "mango"
               null
            ]

            >>> agnostic_to_lowercase(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "apple",
                "mango",
                null
              ]
            ]
        """,
    "to_datetime": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["2020-01-01", "2020-01-02"]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_to_datetime(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.str.to_datetime(format="%Y-%m-%d").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_to_datetime`:

            >>> agnostic_to_datetime(s_pd)
            0   2020-01-01
            1   2020-01-02
            dtype: datetime64[ns]

            >>> agnostic_to_datetime(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [datetime[μs]]
            [
               2020-01-01 00:00:00
               2020-01-02 00:00:00
            ]

            >>> agnostic_to_datetime(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at 0x...>
            [
              [
                2020-01-01 00:00:00.000000,
                2020-01-02 00:00:00.000000
              ]
            ]
        """,
}
