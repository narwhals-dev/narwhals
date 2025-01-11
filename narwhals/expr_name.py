from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Generic
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.expr import Expr

ExprT = TypeVar("ExprT", bound="Expr")


class ExprNameNamespace(Generic[ExprT]):
    def __init__(self: Self, expr: ExprT) -> None:
        self._expr = expr

    def keep(self: Self) -> ExprT:
        r"""Keep the original root name of the expression.

        Returns:
            A new expression.

        Notes:
            This will undo any previous renaming operations on the expression.
            Due to implementation constraints, this method can only be called as the last
            expression in a chain. Only one name operation per expression will work.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).name.keep()
        )

    def map(self: Self, function: Callable[[str], str]) -> ExprT:
        r"""Rename the output of an expression by mapping a function over the root name.

        Arguments:
            function: Function that maps a root name to a new name.

        Returns:
            A new expression.

        Notes:
            This will undo any previous renaming operations on the expression.
            Due to implementation constraints, this method can only be called as the last
            expression in a chain. Only one name operation per expression will work.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).name.map(function)
        )

    def prefix(self: Self, prefix: str) -> ExprT:
        r"""Add a prefix to the root column name of the expression.

        Arguments:
            prefix: Prefix to add to the root column name.

        Returns:
            A new expression.

        Notes:
            This will undo any previous renaming operations on the expression.
            Due to implementation constraints, this method can only be called as the last
            expression in a chain. Only one name operation per expression will work.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).name.prefix(prefix)
        )

    def suffix(self: Self, suffix: str) -> ExprT:
        r"""Add a suffix to the root column name of the expression.

        Arguments:
            suffix: Suffix to add to the root column name.

        Returns:
            A new expression.

        Notes:
            This will undo any previous renaming operations on the expression.
            Due to implementation constraints, this method can only be called as the last
            expression in a chain. Only one name operation per expression will work.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).name.suffix(suffix)
        )

    def to_lowercase(self: Self) -> ExprT:
        r"""Make the root column name lowercase.

        Returns:
            A new expression.

        Notes:
            This will undo any previous renaming operations on the expression.
            Due to implementation constraints, this method can only be called as the last
            expression in a chain. Only one name operation per expression will work.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).name.to_lowercase()
        )

    def to_uppercase(self: Self) -> ExprT:
        r"""Make the root column name uppercase.

        Returns:
            A new expression.

        Notes:
            This will undo any previous renaming operations on the expression.
            Due to implementation constraints, this method can only be called as the last
            expression in a chain. Only one name operation per expression will work.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).name.to_uppercase()
        )
