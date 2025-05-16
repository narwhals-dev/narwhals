from __future__ import annotations

from narwhals._plan.common import Function
from narwhals._plan.options import FunctionOptions


class CategoricalFunction(Function): ...


class GetCategories(CategoricalFunction):
    """https://github.com/pola-rs/polars/blob/62257860a43ec44a638e8492ed2cf98a49c05f2e/crates/polars-plan/src/dsl/function_expr/cat.rs#L7."""

    @property
    def function_options(self) -> FunctionOptions:
        """https://github.com/pola-rs/polars/blob/62257860a43ec44a638e8492ed2cf98a49c05f2e/crates/polars-plan/src/dsl/function_expr/cat.rs#L41."""
        return FunctionOptions.groupwise()

    def __repr__(self) -> str:
        return "cat.get_categories"
