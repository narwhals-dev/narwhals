from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprNamespace, Function, IRNamespace
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals._plan.dummy import DummyExpr


class CategoricalFunction(Function): ...


class GetCategories(CategoricalFunction):
    """https://github.com/pola-rs/polars/blob/62257860a43ec44a638e8492ed2cf98a49c05f2e/crates/polars-plan/src/dsl/function_expr/cat.rs#L7."""

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.groupwise()

    def __repr__(self) -> str:
        return "cat.get_categories"


class IRCatNamespace(IRNamespace):
    def get_categories(self) -> GetCategories:
        return GetCategories()


class ExprCatNamespace(ExprNamespace[IRCatNamespace]):
    @property
    def _ir_namespace(self) -> type[IRCatNamespace]:
        return IRCatNamespace

    def get_categories(self) -> DummyExpr:
        return self._to_narwhals(
            self._ir.get_categories().to_function_expr(self._expr._ir)
        )
