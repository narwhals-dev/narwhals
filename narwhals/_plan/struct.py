from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprNamespace, Function, IRNamespace
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals._plan.dummy import DummyExpr


class StructFunction(Function): ...


class FieldByName(StructFunction):
    """https://github.com/pola-rs/polars/blob/62257860a43ec44a638e8492ed2cf98a49c05f2e/crates/polars-plan/src/dsl/function_expr/struct_.rs#L11."""

    __slots__ = ("name",)

    name: str

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return f"struct.field_by_name({self.name!r})"


class IRStructNamespace(IRNamespace):
    def field(self, name: str) -> FieldByName:
        return FieldByName(name=name)


class ExprStructNamespace(ExprNamespace[IRStructNamespace]):
    @property
    def _ir_namespace(self) -> type[IRStructNamespace]:
        return IRStructNamespace

    def field(self, name: str) -> DummyExpr:
        return self._to_narwhals(self._ir.field(name).to_function_expr(self._expr._ir))
