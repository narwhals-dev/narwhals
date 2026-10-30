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

    def get(self, key: NonNestedLiteral) -> ExprT:
        """Return the value for `key` in each map.

        The result is null where the map is null or does not contain `key`.

        Arguments:
            key: Key to look up.

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
