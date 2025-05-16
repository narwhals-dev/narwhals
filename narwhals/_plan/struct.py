from __future__ import annotations

from narwhals._plan.common import Function
from narwhals._plan.options import FunctionOptions


class StructFunction(Function): ...


class FieldByName(StructFunction):
    """https://github.com/pola-rs/polars/blob/62257860a43ec44a638e8492ed2cf98a49c05f2e/crates/polars-plan/src/dsl/function_expr/struct_.rs#L11."""

    __slots__ = ("name",)

    name: str

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()
