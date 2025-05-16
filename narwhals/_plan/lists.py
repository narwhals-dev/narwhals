from __future__ import annotations

from narwhals._plan.common import Function
from narwhals._plan.options import FunctionOptions


class ListFunction(Function): ...


class Len(ListFunction):
    """https://github.com/pola-rs/polars/blob/62257860a43ec44a638e8492ed2cf98a49c05f2e/crates/polars-plan/src/dsl/function_expr/list.rs#L32."""

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "list.len"
