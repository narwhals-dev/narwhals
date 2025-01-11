from __future__ import annotations

EXAMPLES = {
    "is_into_series": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import numpy as np
        >>> import narwhals as nw

        >>> s_pd = pd.Series([1, 2, 3])
        >>> s_pl = pl.Series([1, 2, 3])
        >>> np_arr = np.array([1, 2, 3])

        >>> nw.dependencies.is_into_series(s_pd)
        True
        >>> nw.dependencies.is_into_series(s_pl)
        True
        >>> nw.dependencies.is_into_series(np_arr)
        False
    """,
    "is_into_dataframe": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import numpy as np
        >>> from narwhals.dependencies import is_into_dataframe

        >>> df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> df_pl = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> np_arr = np.array([[1, 4], [2, 5], [3, 6]])

        >>> is_into_dataframe(df_pd)
        True
        >>> is_into_dataframe(df_pl)
        True
        >>> is_into_dataframe(np_arr)
        False
    """,
}
