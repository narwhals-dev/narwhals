from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._function import Elementwise, Function, UnaryFunction
from narwhals._plan.common import into_dtype
from narwhals._plan.expressions.namespace import IRNamespace
from narwhals._utils import Version
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.expressions import FromStructExpr, FunctionExpr
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType

STRUCT = Version.MAIN.dtypes.Struct
# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
renamed = DispatcherOptions.renamed


class StructFunction(Function, dispatch=DispatcherOptions(accessor_name="struct")):
    @classmethod
    def __function_expr__(cls) -> type[FromStructExpr[Any]]:
        from narwhals._plan.expressions import FromStructExpr

        return FromStructExpr

    @property
    def needs_expansion(self) -> bool:
        msg = f"{type(self).__name__}.needs_expansion"
        raise NotImplementedError(msg)


class FieldByName(UnaryFunction, StructFunction, Elementwise, dispatch=renamed("field")):
    __slots__ = ("name",)
    name: str

    def __repr__(self) -> str:
        return f"{super().__repr__()}({self.name!r})"

    def __expr_ir_repr__(self, node: FunctionExpr[Any], /) -> str:
        return f"{node.args[0]!r}.{self!r}"

    @property
    def needs_expansion(self) -> bool:
        return True

    def resolve_dtype(self, node: FunctionExpr[Self], schema: FrozenSchema, /) -> DType:
        prev = node.args[0]
        name = self.name
        if (
            (
                (not schema and (dtype := prev.resolve_dtype(schema)))
                or (dtype := schema.get(prev.meta.output_name()))
            )
            and isinstance(dtype, STRUCT)
            and (field := next((f for f in dtype.fields if f.name == name), None))
        ):
            return into_dtype(field.dtype)
        raise not_found_error(name)


class IRStructNamespace(IRNamespace):
    field: ClassVar = FieldByName


def not_found_error(name: str, /) -> InvalidOperationError:
    msg = f"Struct field not found {name!r}"
    return InvalidOperationError(msg)
