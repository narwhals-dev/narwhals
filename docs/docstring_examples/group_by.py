from __future__ import annotations

EXAMPLES = {
    "agg": """
            Group by one column or by multiple columns and call `agg` to compute
            the grouped sum of another column.

            >>> import polars as pl
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "a": ["a", "b", "a", "b", "c"],
            ...         "b": [1, 2, 1, 3, 3],
            ...         "c": [5, 4, 3, 2, 1],
            ...     }
            ... )

            We define library agnostic functions:

            >>> def agnostic_func_one_col(lf_native: IntoFrameT) -> IntoFrameT:
            ...     lf = nw.from_native(lf_native)
            ...     return nw.to_native(lf.group_by("a").agg(nw.col("b").sum()).sort("a"))

            >>> def agnostic_func_mult_col(lf_native: IntoFrameT) -> IntoFrameT:
            ...     lf = nw.from_native(lf_native)
            ...     return nw.to_native(lf.group_by("a", "b").agg(nw.sum("c")).sort("a", "b"))

            We can then pass a lazy frame and materialise it with `collect`:

            >>> agnostic_func_one_col(lf_pl).collect()
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
            >>> agnostic_func_mult_col(lf_pl).collect()
            shape: (4, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ a   ┆ 1   ┆ 8   │
            │ b   ┆ 2   ┆ 4   │
            │ b   ┆ 3   ┆ 2   │
            │ c   ┆ 3   ┆ 1   │
            └─────┴─────┴─────┘
        """,
}
