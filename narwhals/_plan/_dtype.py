from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol

from narwhals._plan._guards import is_function_expr
from narwhals._plan._meta import SlottedMeta
from narwhals._typing_compat import TypeVar
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from typing_extensions import Self, TypeAlias, TypeIs

    from narwhals._plan.options import ExprIROptions
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType


# NOTE: `Column` is the exception, which uses `schema[self.name]` and so it is manually defined
# rather than accepting 2 arguments here and ignoring in all other cases
T = TypeVar("T")
Visitor: TypeAlias = "Callable[[T], DType]"
"""A function requiring a single argument to derive the resolved `DType`."""

# fmt: off
# NOTE: Structural typing to prevent cycles
# See https://github.com/microsoft/pyright/issues/10661
_HasDTypeT = TypeVar("_HasDTypeT", bound="_ExprIRDType | _FunctionExpr[_FunctionDType]")
_HasParentExprIRT = TypeVar("_HasParentExprIRT", bound="_HasParentExprIR")
_FunctionExprT = TypeVar("_FunctionExprT", bound="_FunctionExpr[Any]")
_ExprIRT = TypeVar("_ExprIRT", bound="_ExprIR", default=Any)
_FunctionT = TypeVar("_FunctionT", bound="_Function", default=Any)
_FunctionT_co = TypeVar("_FunctionT_co", bound="_Function", covariant=True, default=Any)

class _ExprIR(Protocol):
    __expr_ir_config__: ClassVar[ExprIROptions]
    def resolve_dtype(self, schema: FrozenSchema, /) -> DType: ...
class _HasParentExprIR(_ExprIR, Protocol):
    @property
    def expr(self) -> _ExprIR: ...
class _Function(Protocol):
    def resolve_dtype(self, node: Any, schema: FrozenSchema, /) -> DType: ...
class _HasDType(Protocol):
    @property
    def dtype(self) -> DType: ...
class _ExprIRDType(_ExprIR, _HasDType, Protocol): ...
class _FunctionDType(_Function, _HasDType, Protocol): ...
class _FunctionExpr(_ExprIR, Protocol[_FunctionT_co]):
    @property
    def function(self) -> _FunctionT_co: ...
    @property
    def input(self) -> tuple[_ExprIR,...]: ...
# fmt: on


class _ClassAccessorDescriptor:
    """Namespace accessor that acts like if `@classmethod` and `@property` had a baby."""

    __slots__ = ()

    def __get__(self, instance: Any, owner: Any) -> Self:
        return self


class _FunctionAccessor(_ClassAccessorDescriptor):
    __slots__ = ()

    @staticmethod
    def same_dtype() -> FunctionSameDType[Any]:
        """Propagate the dtype of first function input."""
        return FunctionSameDType()

    @staticmethod
    def map_first(mapper: Visitor[DType], /) -> FunctionMapFirst[Any]:
        """Derive the dtype by calling `mapper` on the dtype of first function input."""
        return FunctionMapFirst(mapper)

    @staticmethod
    def map_all(mapper: Visitor[Iterable[DType]], /) -> FunctionMapAll[Any]:
        """Derive the dtype by calling `mapper` on the dtypes of all function inputs."""
        return FunctionMapAll(mapper)

    @staticmethod
    def visitor(visitor: Visitor[_FunctionT], /) -> FunctionVisitor[_FunctionT]:
        """Derive the dtype by calling `visitor` on an instance of `FunctionT`."""
        return FunctionVisitor(visitor)


class _ExprIRAccessor(_ClassAccessorDescriptor):
    __slots__ = ()

    @staticmethod
    def same_dtype() -> ExprIRSameDType:
        """Propagate the dtype of first expression input."""
        return ExprIRSameDType()

    @staticmethod
    def map_first(mapper: Visitor[DType], /) -> ExprIRMapFirst[Any]:
        """Derive the dtype by calling `mapper` on the dtype of first expression input."""
        return ExprIRMapFirst(mapper)

    @staticmethod
    def visitor(visitor: Visitor[_ExprIRT], /) -> ExprIRVisitor[_ExprIRT]:
        """Derive the dtype by calling `visitor` on an instance of `ExprIRT`."""
        return ExprIRVisitor(visitor)


class ResolveDType(Generic[_ExprIRT], metaclass=SlottedMeta):
    """Resolve the data type of an expanded expression.

    An `ExprIR` or `Function` can use this to define how the node derives
    a `DType` and (optionally) how the incoming `DType` should be transformed.

    By default, this operation is unsupported and when called `ResolveDType` will raise an
    appropriate error.

    `ResolveDType` provides constructors (`@staticmethod`s) targeting patterns observed in:
    - [`AExpr.to_field_impl`]
    - [`IRFunctionExpr.get_field`]
    - [`FieldsMapper`]

    <!---TODO @dangotbanned: Add examples after finalizing the naming -->

    [`AExpr.to_field_impl`]: https://github.com/pola-rs/polars/blob/375fdc81c846c2c35e1b96677d0b483b33a6c3d1/crates/polars-plan/src/plans/aexpr/schema.rs#L45-L877
    [`IRFunctionExpr.get_field`]: https://github.com/pola-rs/polars/blob/375fdc81c846c2c35e1b96677d0b483b33a6c3d1/crates/polars-plan/src/plans/aexpr/function_expr/schema.rsL6-L463
    [`FieldsMapper`]: https://github.com/pola-rs/polars/blob/375fdc81c846c2c35e1b96677d0b483b33a6c3d1/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L476-L838
    """

    __slots__ = ()

    def __call__(self, node: _ExprIRT, schema: FrozenSchema, /) -> DType:
        node_name = type(node).__name__
        if is_function_expr(node):
            generic_name = f"{node_name}[{type(node.function).__name__}]"
        elif not node.__expr_ir_config__.allow_dispatch:
            msg = f"`resolve_dtype` is not supported for {node_name!r}.\n"
            "This method should only be called as `NamedIR.resolve_dtype(...)`,\n"
            f"to ensure all expressions have been expanded, got:\n{node!r}"
            raise InvalidOperationError(msg)
        else:
            generic_name = node_name
        msg = f"`NamedIR[{generic_name}].resolve_dtype()` is not yet implemented, got:\n{node!r}"
        raise NotImplementedError(msg)

    # TODO @dangotbanned: Look into caching these on `dtype`
    @staticmethod
    def just_dtype(dtype: DType, /) -> JustDType:
        """Always returns exactly `dtype`, disregarding any prior context."""
        return JustDType(dtype)

    @staticmethod
    def get_dtype() -> GetDType[Any]:
        """Propagate a `dtype` attribute from the `ExprIR` or `Function` instance."""
        return GetDType()

    expr_ir = _ExprIRAccessor()
    """`ExprIR`-based constructors."""

    function = _FunctionAccessor()
    """`Function`-based constructors."""


class _Singleton(ResolveDType[_ExprIRT], Generic[_ExprIRT]):
    __instance: ClassVar[Any | None] = None

    def __new__(cls) -> Self:
        if not isinstance(cls.__instance, cls):
            cls.__instance = object.__new__(cls)
        return cls.__instance


class GetDType(_Singleton[_HasDTypeT]):
    def __call__(self, node: _HasDTypeT, _: FrozenSchema, /) -> DType:
        if _is_function_expr_dtype(node):
            return node.function.dtype
        return node.dtype


class JustDType(ResolveDType[Any]):
    __slots__ = ("_dtype",)

    def __init__(self, dtype: DType, /) -> None:
        self._dtype: DType = dtype

    def __call__(self, _: Any, __: FrozenSchema, /) -> DType:
        return self._dtype


class ExprIRSameDType(_Singleton[_HasParentExprIR]):
    def __call__(self, node: _HasParentExprIR, schema: FrozenSchema, /) -> DType:
        return node.expr.resolve_dtype(schema)


class ExprIRMapFirst(ResolveDType[_HasParentExprIRT]):
    __slots__ = ("_mapper",)

    def __init__(self, mapper: Visitor[DType], /) -> None:
        self._mapper = mapper

    def __call__(self, node: _HasParentExprIRT, schema: FrozenSchema, /) -> DType:
        return self._mapper(node.expr.resolve_dtype(schema))


class ExprIRVisitor(ResolveDType[_ExprIRT], Generic[_ExprIRT]):
    __slots__ = ("_visitor",)

    def __init__(self, visitor: Visitor[_ExprIRT], /) -> None:
        self._visitor = visitor

    def __call__(self, node: _ExprIRT, _: FrozenSchema, /) -> DType:
        return self._visitor(node)


class FunctionVisitor(ResolveDType[_FunctionExpr[_FunctionT]], Generic[_FunctionT]):
    __slots__ = ("_visitor",)

    def __init__(self, visitor: Visitor[_FunctionT], /) -> None:
        self._visitor = visitor

    def __call__(self, node: _FunctionExpr[_FunctionT], _: FrozenSchema, /) -> DType:
        return self._visitor(node.function)


class FunctionSameDType(_Singleton[_FunctionExprT]):
    def __call__(self, node: _FunctionExprT, schema: FrozenSchema, /) -> DType:
        return node.input[0].resolve_dtype(schema)


class FunctionMapFirst(ResolveDType[_FunctionExprT]):
    __slots__ = ("_mapper",)

    def __init__(self, mapper: Visitor[DType], /) -> None:
        self._mapper = mapper

    def __call__(self, node: _FunctionExprT, schema: FrozenSchema, /) -> DType:
        return self._mapper(node.input[0].resolve_dtype(schema))


class FunctionMapAll(ResolveDType[_FunctionExprT]):
    __slots__ = ("_mapper",)

    def __init__(self, mapper: Visitor[Iterable[DType]], /) -> None:
        self._mapper = mapper

    def __call__(self, node: _FunctionExprT, schema: FrozenSchema, /) -> DType:
        return self._mapper(e.resolve_dtype(schema) for e in node.input)


def _is_function_expr_dtype(node: Any) -> TypeIs[_FunctionExpr[_FunctionDType]]:
    return is_function_expr(node)
