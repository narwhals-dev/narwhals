from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._expr_ir import ExprIR
from narwhals._plan._immutable import Immutable
from narwhals._plan._nodes import node
from narwhals._plan.expressions.namespace import IRNamespace

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._compliant.typing import AliasName


class KeepName(ExprIR, dispatch="no_dispatch"):
    """Keep the original root name of an expression.

    Arguments:
        expr: An expression with at least one `Col`.

    Important:
        All expressions that change the output name are
        resolved and removed following expression expansion.
        This means that you can do arbitrarily complex renames,
        **at the narwhals-level** but there is intentionally no support
        for them at the compliant-level.
    """

    __slots__ = ("expr",)
    expr: ExprIR = node()

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self

    def __repr__(self) -> str:
        return f"{self.expr!r}.name.keep()"

    def is_length_preserving(self) -> bool:
        return self.expr.is_length_preserving()


# TODO @dangotbanned: `RenameAlias` -> `MapAlias`?
# TODO @dangotbanned: Give `RenameAlias`, `Alias`, `KeepName` a common parent
class RenameAlias(ExprIR, dispatch="no_dispatch"):
    """Rename an expression by mapping a function over the root name.

    Arguments:
        expr: An expression or selector to rename.
        function: A function that maps a root name to a new name.

    Important:
        All expressions that change the output name are
        resolved and removed following expression expansion.
        This means that you can do arbitrarily complex renames,
        **at the narwhals-level** but there is intentionally no support
        for them at the compliant-level.
    """

    __slots__ = ("expr", "function")
    expr: ExprIR = node()
    function: AliasName

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self

    def __repr__(self) -> str:
        f = self.function
        if isinstance(f, (Prefix, Suffix)):
            s = repr(f)
        elif f in {str.upper, str.lower}:
            s = f"to_{f.__name__}case()"
        else:
            s = "map()"
        return f"{self.expr!r}.name.{s}"

    def is_length_preserving(self) -> bool:
        return self.expr.is_length_preserving()


class Prefix(Immutable):
    __slots__ = ("prefix",)
    prefix: str

    def __call__(self, name: str, /) -> str:
        return f"{self.prefix}{name}"

    def __repr__(self) -> str:
        return f"prefix({self.prefix!r})"


class Suffix(Immutable):
    __slots__ = ("suffix",)
    suffix: str

    def __call__(self, name: str, /) -> str:
        return f"{name}{self.suffix}"

    def __repr__(self) -> str:
        return f"suffix({self.suffix!r})"


class IRNameNamespace(IRNamespace):
    def keep(self) -> KeepName:
        return KeepName(expr=self._ir)

    def map(self, function: AliasName) -> RenameAlias:
        return RenameAlias(expr=self._ir, function=function)

    def prefix(self, prefix: str) -> RenameAlias:
        return self.map(Prefix(prefix=prefix))

    def suffix(self, suffix: str) -> RenameAlias:
        return self.map(Suffix(suffix=suffix))

    def to_lowercase(self) -> RenameAlias:
        return self.map(str.lower)

    def to_uppercase(self) -> RenameAlias:
        return self.map(str.upper)
