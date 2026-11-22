from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from narwhals._expression_parsing import ExprKind, ExprNode

if TYPE_CHECKING:
    from narwhals.expr import Expr
    from narwhals.typing import NonNestedLiteral

ExprT = TypeVar("ExprT", bound="Expr")


class ExprMapNamespace(Generic[ExprT]):
    def __init__(self, expr: ExprT) -> None:
        self._expr = expr

    def len(self) -> ExprT:
        """Return the number of entries in each map.

        Null maps return null. Entries with null values count towards the total.
        The result has dtype `UInt32`. Supported on DuckDB, Ibis with DuckDB,
        PyArrow, Polars 2.0 or newer, pandas, and Modin. On pandas and Modin, this
        requires a PyArrow-backed `Map` column and pandas 2.2 or newer.

        Examples:
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> maps = pa.array(
            ...     [[("x", 1), ("y", 2)], [], None, [("x", None)]],
            ...     type=pa.map_(pa.string(), pa.int64()),
            ... )
            >>> df = nw.from_native(pa.table({"a": maps}))
            >>> df.select(nw.col("a").map.len()).get_column("a").to_list()
            [2, 0, None, 1]
        """
        return self._expr._append_node(ExprNode(ExprKind.ELEMENTWISE, "map.len"))

    def get(self, key: NonNestedLiteral) -> ExprT:
        """Return the value for `key` in each map.

        The result is null where the map is null or does not contain `key`.

        Arguments:
            key: Key to look up. Its type should match the map's key type.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame(
            ...     {
            ...         "a": pl.Series(
            ...             [{"x": 1, "y": 2}, {"y": 3}, None],
            ...             dtype=pl.Map(pl.String, pl.Int64),
            ...         )
            ...     }
            ... )
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(a_x=nw.col("a").map.get("x"))
            ┌───────────────────────────┐
            |    Narwhals DataFrame     |
            |---------------------------|
            |shape: (3, 2)              |
            |┌──────────────────┬──────┐|
            |│ a                ┆ a_x  │|
            |│ ---              ┆ ---  │|
            |│ map[str, i64]    ┆ i64  │|
            |╞══════════════════╪══════╡|
            |│ {"x": 1, "y": 2} ┆ 1    │|
            |│ {"y": 3}         ┆ null │|
            |│ null             ┆ null │|
            |└──────────────────┴──────┘|
            └───────────────────────────┘
        """
        return self._expr._append_node(ExprNode(ExprKind.ELEMENTWISE, "map.get", key=key))
