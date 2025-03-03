from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Generic
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.expr import Expr

ExprT = TypeVar("ExprT", bound="Expr")


class ExprStructNamespace(Generic[ExprT]):
    def __init__(self: Self, expr: ExprT) -> None:
        self._expr = expr

    def field(self: Self, *names: str) -> ExprT:
        r"""Retrieve one or multiple Struct field as a new expression.

        Arguments:
            names: Name of the struct field to retrieve.

        Returns:
            A new expression.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame(
            ...     {
            ...         "user": [
            ...             {"id": 0, "name": "john"},
            ...             {"id": 1, "name": "jane"},
            ...         ]
            ...     }
            ... )
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(name=nw.col("user").struct.field("name"))
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).struct.field(*names),
            self._expr._metadata,
        )
