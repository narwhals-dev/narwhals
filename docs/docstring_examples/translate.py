from __future__ import annotations

EXAMPLES = {
    "get_native_namespace": """
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))
        >>> nw.get_native_namespace(df)
        <module 'pandas'...>
        >>> df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
        >>> nw.get_native_namespace(df)
        <module 'polars'...>
    """,
    "narwhalify": """
        Instead of writing

        >>> import narwhals as nw
        >>> def agnostic_group_by_sum(df):
        ...     df = nw.from_native(df, pass_through=True)
        ...     df = df.group_by("a").agg(nw.col("b").sum())
        ...     return nw.to_native(df)

        you can just write

        >>> @nw.narwhalify
        ... def agnostic_group_by_sum(df):
        ...     return df.group_by("a").agg(nw.col("b").sum())
    """,
    "to_py_scalar": """
        >>> import narwhals as nw
        >>> import pandas as pd
        >>> df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))
        >>> nw.to_py_scalar(df["a"].item(0))
        1
        >>> import pyarrow as pa
        >>> df = nw.from_native(pa.table({"a": [1, 2, 3]}))
        >>> nw.to_py_scalar(df["a"].item(0))
        1
        >>> nw.to_py_scalar(1)
        1
    """,
}
