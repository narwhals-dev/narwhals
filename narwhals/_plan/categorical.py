from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprNamespace, Function, IRNamespace
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals._plan.dummy import Expr


class CategoricalFunction(Function, accessor="cat"): ...


class GetCategories(CategoricalFunction, options=FunctionOptions.groupwise):
    """https://github.com/pola-rs/polars/blob/62257860a43ec44a638e8492ed2cf98a49c05f2e/crates/polars-plan/src/dsl/function_expr/cat.rs#L7."""


class IRCatNamespace(IRNamespace):
    def get_categories(self) -> GetCategories:
        return GetCategories()


class ExprCatNamespace(ExprNamespace[IRCatNamespace]):
    @property
    def _ir_namespace(self) -> type[IRCatNamespace]:
        return IRCatNamespace

    def get_categories(self) -> Expr:
        return self._with_unary(self._ir.get_categories())
