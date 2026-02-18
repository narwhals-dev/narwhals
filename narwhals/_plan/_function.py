from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._dispatch import Dispatcher
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._immutable import Immutable
from narwhals._plan.common import replace
from narwhals._plan.options import FEOptions, FunctionOptions
from narwhals.dtypes import DType

if TYPE_CHECKING:
    from typing import Any, Callable, ClassVar

    from typing_extensions import Self

    from narwhals._plan.expressions import ExprIR, FunctionExpr
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import Accessor

__all__ = ["Function", "HorizontalFunction"]


class Function(Immutable):
    """Shared by expr functions and namespace functions.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L114
    """

    _function_options: ClassVar[staticmethod[[], FunctionOptions]] = staticmethod(
        FunctionOptions.default
    )
    __expr_ir_config__: ClassVar[FEOptions] = FEOptions.default()
    __expr_ir_dispatch__: ClassVar[Dispatcher[FunctionExpr[Self]]]
    __expr_ir_dtype__: ClassVar[ResolveDType[FunctionExpr[Self]]] = ResolveDType.default()

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
        dtype: DType | ResolveDType[Any] | Callable[[Self], DType] | None = None,
        **kwds: Any,
    ) -> None:
        super().__init_subclass__(*args, **kwds)
        if accessor_name := accessor or cls.__expr_ir_config__.accessor_name:
            config = replace(config or FEOptions.default(), accessor_name=accessor_name)
        if options:
            cls._function_options = staticmethod(options)
        if config:
            cls.__expr_ir_config__ = config
        cls.__expr_ir_dispatch__ = Dispatcher.from_function(cls)
        if dtype is not None:
            if isinstance(dtype, DType):
                dtype = ResolveDType.from_dtype(dtype)
            elif not isinstance(dtype, ResolveDType):
                dtype = ResolveDType.function_visitor(dtype)
            cls.__expr_ir_dtype__ = dtype

    def __repr__(self) -> str:
        return self.__expr_ir_dispatch__.name

    # TODO @dangotbanned: Try to avoid a contravariant (`Self` or `Function`)
    # Makes things complicated in the namespaces, which can only use
    # `FunctionExpr[FunctionT_co]` *because* it is a return type

    # TODO @dangotbanned: Flip `(schema, node)` -> `(node, schema)`
    # Will match the convention for `Dispatcher`
    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        # TODO @dangotbanned: Replace `_resolve_dtype` entirely with an identical pattern to `FunctionExpr.dispatch`?
        return self.__expr_ir_dtype__(node, schema)  # type: ignore[arg-type]


class HorizontalFunction(
    Function, options=FunctionOptions.horizontal, config=FEOptions.namespaced()
): ...
