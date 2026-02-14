from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._expr_ir import ExprIR
from narwhals._plan._immutable import Immutable
from narwhals._plan.expressions.namespace import ExprNamespace, IRNamespace
from narwhals._plan.options import ExprIROptions

if TYPE_CHECKING:
    from narwhals._compliant.typing import AliasName
    from narwhals._plan.expr import Expr
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType


class KeepName(ExprIR, child=("expr",), config=ExprIROptions.no_dispatch()):
    __slots__ = ("expr",)
    expr: ExprIR

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.name.keep()"

    def _resolve_dtype(self, schema: FrozenSchema) -> DType:
        return self.expr._resolve_dtype(schema)


class RenameAlias(ExprIR, child=("expr",), config=ExprIROptions.no_dispatch()):
    __slots__ = ("expr", "function")
    expr: ExprIR
    function: AliasName

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        return f".rename_alias({self.expr!r})"

    def _resolve_dtype(self, schema: FrozenSchema) -> DType:
        return self.expr._resolve_dtype(schema)


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


class ExprNameNamespace(ExprNamespace[IRNameNamespace]):
    @property
    def _ir_namespace(self) -> type[IRNameNamespace]:
        return IRNameNamespace

    def keep(self) -> Expr:
        return self._to_narwhals(self._ir.keep())

    def map(self, function: AliasName) -> Expr:
        """Define an alias by mapping a function over the original root column name."""
        return self._to_narwhals(self._ir.map(function))

    def prefix(self, prefix: str) -> Expr:
        """Add a prefix to the root column name."""
        return self._to_narwhals(self._ir.prefix(prefix))

    def suffix(self, suffix: str) -> Expr:
        """Add a suffix to the root column name."""
        return self._to_narwhals(self._ir.suffix(suffix))

    def to_lowercase(self) -> Expr:
        """Update the root column name to use lowercase characters."""
        return self._to_narwhals(self._ir.to_lowercase())

    def to_uppercase(self) -> Expr:
        """Update the root column name to use uppercase characters."""
        return self._to_narwhals(self._ir.to_uppercase())
