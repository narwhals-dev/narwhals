"""`resolve_dtype` equivalent to `Dispatcher`."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Generic, Protocol, final

from narwhals._plan._guards import is_function_expr
from narwhals._typing_compat import TypeVar
from narwhals.dtypes import DType
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan.dtypes_mapper import _HasChildExpr
    from narwhals._plan.expressions import ExprIR, Function, FunctionExpr as FExpr
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import ExprIRT, FunctionT


T = TypeVar("T")
Node = TypeVar("Node", bound="ExprIR | FExpr[Any] | _HasChildExpr")
Node_contra = TypeVar(
    "Node_contra",
    bound="ExprIR | FExpr[Any] | _HasChildExpr",
    contravariant=True,
    default=Any,
)
FunctionExprT = TypeVar("FunctionExprT", bound="FExpr[Function]")


# NOTE: `Column` is the exception, which uses `schema[self.name]` and so it is manually defined
# rather than accepting 2 arguments here and ignoring in all other cases
Visitor: TypeAlias = Callable[[T], DType]
"""A function requiring a single argument to derive the resolved `DType`."""

MapDType: TypeAlias = Visitor[DType]
ReduceDTypes: TypeAlias = Visitor[Iterable[DType]]


class _Resolver(Protocol[Node_contra]):
    def __call__(self, node: Node_contra, schema: FrozenSchema, /) -> DType: ...


# TODO @dangotbanned: Un-`@final` this and use subclasses for the static methods
@final
class ResolveDType(Generic[Node]):
    __slots__ = ("_dtype",)
    _dtype: _Resolver[Node]

    def __call__(self, node: Node, schema: FrozenSchema, /) -> DType:
        """Note how (node, schema) are flipped to match `Dispatcher`."""
        return self._dtype(node, schema)

    @staticmethod
    def default() -> ResolveDType[Any]:
        """By default, the only appropriate behavior is to raise as an unsupported operation."""
        return ResolveDType(_no_default_error)

    @staticmethod
    def from_dtype(dtype: DType, /) -> ResolveDType[Any]:
        """Always returns exactly `dtype`."""
        return ResolveDType(_from_dtype(dtype))

    @staticmethod
    def get_dtype() -> ResolveDType[Any]:
        """Propagate a `dtype` attribute from the `ExprIR` or `Function` instance."""
        return ResolveDType(_get_dtype)

    # NOTE: Poor man's namespace (1)

    @staticmethod
    def expr_ir_visitor(visitor: Visitor[ExprIRT], /) -> ResolveDType[ExprIRT]:
        """Derive the `DType` by calling `visitor` on an instance of `ExprIRT`."""
        return ResolveDType(_expr_ir_visitor(visitor))

    # NOTE: I think I mixed up child/parent
    @staticmethod
    def expr_ir_root() -> ResolveDType[_HasChildExpr]:
        """Propagate the dtype of first expression input."""
        return ResolveDType(_expr_ir_root)

    # NOTE: Poor man's namespace (2)

    @staticmethod
    def function_same_dtype() -> ResolveDType[FExpr[Any]]:
        """Propagate the dtype of first function input."""
        return ResolveDType(_function_same_dtype)

    @staticmethod
    def function_map_first(mapper: MapDType, /) -> ResolveDType[FExpr[Any]]:
        """Derive the `DType` by calling `mapper` on the dtype of first function input."""
        return ResolveDType(_function_map_first(mapper))

    # TODO @dangotbanned: This isn't what "horizontal" or "reduce" mean
    # You *can* implement reduce with it, but I'm not asking for the binary function
    @staticmethod
    def function_map_horizontal(mapper: ReduceDTypes, /) -> ResolveDType[FExpr[Any]]:
        """Derive the `DType` by calling `mapper` on the dtypes of all function inputs."""
        return ResolveDType(_function_map_horizontal(mapper))

    @staticmethod
    def function_visitor(
        visitor: Visitor[FunctionT], /
    ) -> ResolveDType[FExpr[FunctionT]]:
        """Derive the `DType` by calling `visitor` on an instance of `FunctionT`."""
        return ResolveDType(_function_visitor(visitor))

    # TODO @dangotbanned: Maybe just replace with an `__init__`?
    def __init__(self, resolver: _Resolver[Node], /) -> None:
        self._dtype = resolver


def _from_dtype(dtype: DType, /) -> _Resolver:
    # TODO @dangotbanned: Look into caching these on `dtype`
    def inner(_: Any, __: FrozenSchema, /) -> DType:
        return dtype

    return inner


_dtype_getter = operator.attrgetter("dtype")


def _get_dtype(node: ExprIR, _: FrozenSchema, /) -> DType:
    if is_function_expr(node):
        return _dtype_getter(node.function)
    return _dtype_getter(node)


def _expr_ir_root(node: _HasChildExpr, schema: FrozenSchema, /) -> DType:
    return node.expr._resolve_dtype(schema)


def _no_default_error(node: ExprIR, _: FrozenSchema, /) -> DType:
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


def _expr_ir_visitor(visitor: Visitor[ExprIRT], /) -> _Resolver[ExprIRT]:
    def inner(node: ExprIRT, _: FrozenSchema, /) -> DType:
        return visitor(node)

    return inner


def _function_visitor(visitor: Visitor[FunctionT], /) -> _Resolver[FExpr[FunctionT]]:
    def inner(node: FExpr[FunctionT], _: FrozenSchema, /) -> DType:
        return visitor(node.function)

    return inner


def _function_same_dtype(node: FExpr[Any], schema: FrozenSchema, /) -> DType:
    return node.input[0]._resolve_dtype(schema)


def _function_map_first(mapper: MapDType, /) -> _Resolver[FExpr[Any]]:
    def inner(node: FExpr[Any], schema: FrozenSchema, /) -> DType:
        return mapper(node.input[0]._resolve_dtype(schema))

    return inner


def _function_map_horizontal(mapper: ReduceDTypes, /) -> _Resolver[FExpr[Any]]:
    def inner(node: FExpr[Any], schema: FrozenSchema, /) -> DType:
        return mapper(e._resolve_dtype(schema) for e in node.input)

    return inner
