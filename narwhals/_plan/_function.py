from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._dispatch import dispatch_generate_function, dispatch_method_name
from narwhals._plan._immutable import Immutable
from narwhals._plan.common import replace
from narwhals._plan.options import FEOptions, FunctionOptions

if TYPE_CHECKING:
    from typing import Any, Callable, ClassVar

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.expressions import ExprIR, FunctionExpr
    from narwhals._plan.typing import Accessor

__all__ = ["Function", "HorizontalFunction"]

Incomplete: TypeAlias = "Any"


class Function(Immutable):
    """Shared by expr functions and namespace functions.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L114
    """

    _function_options: ClassVar[staticmethod[[], FunctionOptions]] = staticmethod(
        FunctionOptions.default
    )
    __expr_ir_config__: ClassVar[FEOptions] = FEOptions.default()
    __expr_ir_dispatch__: ClassVar[
        staticmethod[[Incomplete, FunctionExpr[Self], Incomplete, str], Incomplete]
    ]

    @property
    def function_options(self) -> FunctionOptions:
        return self._function_options()

    @property
    def is_scalar(self) -> bool:
        return self.function_options.returns_scalar()

    def to_function_expr(self, *inputs: ExprIR) -> FunctionExpr[Self]:
        from narwhals._plan.expressions.expr import FunctionExpr

        return FunctionExpr(input=inputs, function=self, options=self.function_options)

    def __init_subclass__(
        cls: type[Self],
        *args: Any,
        accessor: Accessor | None = None,
        options: Callable[[], FunctionOptions] | None = None,
        config: FEOptions | None = None,
        **kwds: Any,
    ) -> None:
        super().__init_subclass__(*args, **kwds)
        if accessor:
            config = replace(config or FEOptions.default(), accessor_name=accessor)
        if options:
            cls._function_options = staticmethod(options)
        if config:
            cls.__expr_ir_config__ = config
        cls.__expr_ir_dispatch__ = staticmethod(dispatch_generate_function(cls))

    def __repr__(self) -> str:
        return dispatch_method_name(type(self))


class HorizontalFunction(
    Function, options=FunctionOptions.horizontal, config=FEOptions.namespaced()
): ...
