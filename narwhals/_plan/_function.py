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
                dtype = ResolveDType.just_dtype(dtype)
            elif not isinstance(dtype, ResolveDType):
                dtype = ResolveDType.function.visitor(dtype)
            # TODO @dangotbanned: fix mypy
            # error: Incompatible types in assignment (expression has type "FunctionVisitor[Any] | ResolveDType[Any]",
            # variable has type "ResolveDType[FunctionExpr[Self]]"
            cls.__expr_ir_dtype__ = dtype  # type: ignore[assignment]

    def __repr__(self) -> str:
        return self.__expr_ir_dispatch__.name

    def resolve_dtype(self, node: FunctionExpr[Any], schema: FrozenSchema, /) -> DType:
        """Get the data type of an expanded expression.

        Arguments:
            node: The expanded expression, wrapping this function.
            schema: The same schema used to project `node`.
        """
        return self.__expr_ir_dtype__(node, schema)


class HorizontalFunction(
    Function, options=FunctionOptions.horizontal, config=FEOptions.namespaced()
): ...
