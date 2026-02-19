from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from narwhals._plan._function import Function
from narwhals._plan.common import into_dtype
from narwhals._plan.expressions.namespace import ExprNamespace, IRNamespace
from narwhals._plan.options import FEOptions, FunctionOptions
from narwhals._utils import Version
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan._expr_ir import ExprIR
    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions.expr import FunctionExpr, StructExpr
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType, Field, Struct

STRUCT = Version.MAIN.dtypes.Struct
# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
elementwise = FunctionOptions.elementwise


class StructFunction(Function, accessor="struct"):
    def to_function_expr(self, *inputs: ExprIR) -> StructExpr[Self]:
        from narwhals._plan.expressions.expr import StructExpr

        return StructExpr(input=inputs, function=self, options=self.function_options)

    @property
    def needs_expansion(self) -> bool:
        msg = f"{type(self).__name__}.needs_expansion"
        raise NotImplementedError(msg)


class FieldByName(StructFunction, options=elementwise, config=FEOptions.renamed("field")):
    __slots__ = ("name",)
    name: str

    def __repr__(self) -> str:
        return f"{super().__repr__()}({self.name!r})"

    @property
    def needs_expansion(self) -> bool:
        return True

    def _field(self, dtype: Struct) -> Field:
        if field := next((f for f in dtype.fields if f.name == self.name), None):
            return field
        msg = f"Struct field not found {self.name!r}"
        raise InvalidOperationError(msg)

    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Self]) -> DType:
        if (
            (struct_name := node.input[0].meta.output_name(raise_if_undetermined=False))
            and (struct := schema.get(struct_name))
            and isinstance(struct, STRUCT)
        ):
            return into_dtype(self._field(struct).dtype)
        msg = f"Struct field not found {self.name!r}"
        raise InvalidOperationError(msg)


class IRStructNamespace(IRNamespace):
    field: ClassVar = FieldByName


class ExprStructNamespace(ExprNamespace[IRStructNamespace]):
    @property
    def _ir_namespace(self) -> type[IRStructNamespace]:
        return IRStructNamespace

    def field(self, name: str) -> Expr:
        return self._with_unary(self._ir.field(name=name))
