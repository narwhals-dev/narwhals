from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprIR, ExprIRNamespace, Immutable

if TYPE_CHECKING:
    from narwhals._compliant.typing import AliasName


class KeepName(ExprIR):
    """Keep the original root name."""

    __slots__ = ("expr",)

    expr: ExprIR

    def __repr__(self) -> str:
        return f"{self.expr!r}.name.keep()"


class RenameAlias(ExprIR):
    __slots__ = ("expr", "function")

    expr: ExprIR
    function: AliasName

    def __repr__(self) -> str:
        return f".rename_alias({self.expr!r})"


class Prefix(Immutable):
    __slots__ = ("prefix",)

    prefix: str

    def __call__(self, name: str, /) -> str:
        return f"{self.prefix}{name}"


class Suffix(Immutable):
    __slots__ = ("suffix",)

    suffix: str

    def __call__(self, name: str, /) -> str:
        return f"{name}{self.suffix}"


class ExprIRNameNamespace(ExprIRNamespace):
    """Specialized expressions for modifying the name of existing expressions."""

    def keep(self) -> KeepName:
        return KeepName(expr=self._ir)

    def map(self, function: AliasName) -> RenameAlias:
        """Define an alias by mapping a function over the original root column name."""
        return RenameAlias(expr=self._ir, function=function)

    def prefix(self, prefix: str) -> RenameAlias:
        """Add a prefix to the root column name."""
        return self.map(Prefix(prefix=prefix))

    def suffix(self, suffix: str) -> RenameAlias:
        """Add a suffix to the root column name."""
        return self.map(Suffix(suffix=suffix))

    def to_lowercase(self) -> RenameAlias:
        """Update the root column name to use lowercase characters."""
        return self.map(str.lower)

    def to_uppercase(self) -> RenameAlias:
        """Update the root column name to use uppercase characters."""
        return self.map(str.upper)
