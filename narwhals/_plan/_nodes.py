"""Traversal of `ExprIR`, maybe `*Plan` eventually too.

There's a couple *big ideas* in here, but they could broadly fit into:

> Try to encode as much as possible **into the classes**

## Misc Notes
- Minimize the amount of `isinstance` calls within methods
  - ATOW, all expressions can be fully traversed **and** transformed without a single runtime type check
- Implements [Field specifiers] from [PEP 681]


[Field specifiers]: https://typing.python.org/en/latest/spec/dataclasses.html#field-specifiers
[PEP 681]: https://peps.python.org/pep-0681/#field-specifiers
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, SupportsIndex, TypeVar

from narwhals._utils import qualified_type_name

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from typing_extensions import Self, TypeAlias

    from narwhals._plan._expr_ir import ExprIR
    from narwhals._plan.typing import MapIR, Seq

    class HasExprTraverser(Protocol):
        @property
        def __expr_ir_nodes__(self) -> ExprTraverser: ...


NodeT = TypeVar("NodeT")
Get = TypeVar("Get", covariant=True)  # noqa: PLC0105


class IsScalar(enum.Enum):
    """Annoying workaround for union simplification.

    Originally wanted something like this:

        bool | Literal[False]

    But that simplifies to:

        bool
    """

    OBSERVE = enum.auto()
    SKIP = enum.auto()


_OBSERVE: Final = IsScalar.OBSERVE
_SKIP: Final = IsScalar.SKIP
_Observe: TypeAlias = Literal[IsScalar.OBSERVE]
_Skip: TypeAlias = Literal[IsScalar.SKIP]
IsScalarT = TypeVar(  # noqa: PLC0105
    "IsScalarT", _Observe, _Skip, covariant=True
)


# TODO @dangotbanned: Doc needs to focus on `node(...)`
def node(*, observe_scalar: bool = True) -> Any:
    """Singular `ExprIR` field specifier.

    Part of the spec for [`@dataclass_transform`], and is similar to [`dataclasses.field`]
    but for more niche use cases.

    Arguments:
        observe_scalar: Should `is_scalar` look here?

    [`@dataclass_transform`]: https://typing.python.org/en/latest/spec/dataclasses.html#field-specifiers
    [`dataclasses.field`]: https://docs.python.org/3/library/dataclasses.html#dataclasses.field
    """
    return SingleExpr(_OBSERVE) if observe_scalar else SingleExpr(_SKIP)


# TODO @dangotbanned: Doc needs to focus on `nodes()`
def nodes() -> Any:
    """Multiple `ExprIR` field specifier.

    Part of the spec for [`@dataclass_transform`], and is similar to [`dataclasses.field`]
    but for more niche use cases.

    [`@dataclass_transform`]: https://typing.python.org/en/latest/spec/dataclasses.html#field-specifiers
    [`dataclasses.field`]: https://docs.python.org/3/library/dataclasses.html#dataclasses.field
    """
    return MultipleExpr()


# TODO @dangotbanned: Class doc, focus on why a convention for traversal matters
class Node(Protocol[NodeT, Get]):
    # Hoping to make it easier to document this way
    # overlaps with https://github.com/narwhals-dev/narwhals/blob/c5a624885b845a1a9fcab849296424879197f6e5/narwhals/_plan/plans/_base.py#L13-L27
    __slots__ = ("name",)
    # TODO @dangotbanned: Doc
    name: str

    def iter_left(self, instance: NodeT, /) -> Iterator[NodeT]:
        """Yield nodes recursively from root->leaf."""
        ...

    def iter_right(self, instance: NodeT, /) -> Iterator[NodeT]:
        """Yield nodes recursively from leaf->root."""
        ...

    # TODO @dangotbanned: Doc (maybe)
    def with_name(self, name: str, /) -> Self:
        # - Called during `type.__new__`, when we have the `name`
        # - If we used a descriptor with `__set_name__`, that would be too late to move
        #   the instance from the class namespace
        self.name = name
        return self

    def get(self, instance: NodeT, /) -> Get:
        """Get the implementation of this node from `instance`."""
        node: Get = getattr(instance, self.name)
        return node

    def map_nodes(
        self, instance: NodeT, function: Callable[[NodeT], NodeT], /
    ) -> Get | None:
        """Call `function` on this node and return the result *iff* changed.

        The idea is to limit the size of `Immutable.__replace__(**changes)`, or skip it
        entirely if we didn't need it.
        """
        ...


class ExprNode(Node["ExprIR", Get], Protocol[Get, IsScalarT]):
    __slots__ = ()

    @property
    def observe_scalar(self) -> IsScalarT: ...
    def iter_output_name(self, instance: ExprIR, /) -> Iterator[ExprIR]: ...


class SingleExpr(ExprNode["ExprIR", IsScalarT]):
    __slots__ = ("_observe_scalar",)
    _observe_scalar: IsScalarT

    def __init__(self, observe_scalar: IsScalarT) -> None:
        # Called immediately after:
        #     `<field-name> = <field-specifier-function>(**kwds)`
        self._observe_scalar = observe_scalar

    def __repr__(self) -> str:
        if not hasattr(self, "name"):
            return f"{type(self).__name__}(observe_scalar={self.observe_scalar})"
        args = "observe_scalar=False" if self.observe_scalar is _SKIP else ""
        return f"{self.name}: ExprIR = {node.__name__}({args})"

    @property
    def observe_scalar(self) -> IsScalarT:
        return self._observe_scalar

    def iter_left(self, instance: ExprIR) -> Iterator[ExprIR]:
        yield from self.get(instance).iter_left()

    def iter_right(self, instance: ExprIR) -> Iterator[ExprIR]:
        yield from self.get(instance).iter_right()

    def iter_output_name(self, instance: ExprIR) -> Iterator[ExprIR]:
        yield from self.get(instance).iter_output_name()

    def is_scalar(self, instance: ExprIR) -> bool:
        return self.get(instance).is_scalar()

    def map_nodes(self, instance: ExprIR, function: MapIR, /) -> ExprIR | None:
        before = self.get(instance)
        after = before.map_ir(function)
        return None if after == before else after


class MultipleExpr(ExprNode["Seq[ExprIR]", _Skip]):
    __slots__ = ()

    def __repr__(self) -> str:
        if not hasattr(self, "name"):
            return f"{type(self).__name__}()"
        return f"{self.name}: Seq[ExprIR] = {nodes.__name__}()"

    @property
    def observe_scalar(self) -> _Skip:
        return _SKIP

    def iter_left(self, instance: ExprIR) -> Iterator[ExprIR]:
        for expr in self.get(instance):
            yield from expr.iter_left()

    def iter_right(self, instance: ExprIR) -> Iterator[ExprIR]:
        if exprs := self.get(instance):
            for expr in reversed(exprs):
                yield from expr.iter_right()

    def iter_output_name(self, instance: ExprIR) -> Iterator[ExprIR]:
        for lhs in self.get(instance)[:1]:
            yield from lhs.iter_output_name()

    def map_nodes(self, instance: ExprIR, function: MapIR, /) -> Seq[ExprIR] | None:
        if before := self.get(instance):
            after = tuple(e.map_ir(function) for e in before)
            return None if after == before else after
        return None


_ExprNode: TypeAlias = "SingleExpr[_Observe] | SingleExpr[_Skip] | MultipleExpr"
ExprNodes: TypeAlias = "Seq[_ExprNode]"
IntoExprNodes: TypeAlias = "Iterable[_ExprNode]"

_EXPR_NODE_TYPES: Final = (SingleExpr, MultipleExpr)
_EXPR_FIELD_SPECIFIER_NAMES: Final = (node.__name__, nodes.__name__)


def _into_expr_node_error(assigned_name: str, node: Any, cls_name: str) -> TypeError:
    name = assigned_name
    value = repr(node)
    value = value[:10] + "..." if len(value) > 10 else value
    value = f"`{value}`"
    hints = (
        f"if {name!r} is a class variable,\n  remove it from __slots__",
        f"if {value} is a default, for `{cls_name}({name}=...)`,"
        "\n  define the it in a classmethod, staticmethod or function instead of the class body",
        f"if {value} came from a new fancy field specifier,\n  fix the check that raised this!",
    )
    node_examples = ", ".join(f"`{nm}()`" for nm in _EXPR_FIELD_SPECIFIER_NAMES)
    what = f"Incompatible assignment in ExprIR subclass {cls_name!r}."
    cause = f"The class body tried to assign a {qualified_type_name(node)!r} to {name!r}, while also declaring {name!r} in __slots__."
    nl = "\n"
    return TypeError(
        f"{what}\n\n{cause}\n"
        f"This syntax is reserved for field specifiers: {node_examples}\n\n"
        f"Hints:\n{nl.join(f'- {hint}' for hint in hints)}"
    )


def into_expr_node(assigned_name: str, node: Any, cls_name: str) -> _ExprNode:
    if not isinstance(node, _EXPR_NODE_TYPES):
        raise _into_expr_node_error(assigned_name, node, cls_name)
    return node.with_name(assigned_name)


class ExprTraverser:
    """Field specifier-based iteration backend for `ExprIR`.

    ## Notes
    - Methods *accepting* an `instance` correspond to those in `ExprIR` with the same name
    - The rest expose a subset of `Sequence[_ExprNode]`
        - Excluding the `value`-based methods
    """

    __slots__ = ("_nodes",)
    _nodes: ExprNodes

    def __init__(self, nodes: ExprNodes, /) -> None:
        self._nodes = nodes

    def __repr__(self) -> str:
        return self._repr_with("\n", " ")

    def _repr_html_(self) -> str:
        return self._repr_with("<br>", "&nbsp;")

    def _repr_with(
        self, new_line: Literal["\n", "<br>"], indent: Literal[" ", "&nbsp;"], /
    ) -> str:
        tp_name = type(self).__name__
        if self:
            members = new_line.join(f"{indent * 4}{n!r}" for n in self)
            return f"{tp_name}[{len(self)}]{new_line}{members}"
        return f"{tp_name}[]"

    @staticmethod
    def inherit_from(parent: HasExprTraverser, nodes: IntoExprNodes) -> ExprTraverser:
        return ExprTraverser((*parent.__expr_ir_nodes__, *nodes))

    # NOTE: `ExprIR` API
    def iter_left(self, instance: ExprIR, /) -> Iterator[ExprIR]:
        """Yield from nodes recursively from root->leaf."""
        for node in self:
            yield from node.iter_left(instance)
        yield instance

    def iter_right(self, instance: ExprIR, /) -> Iterator[ExprIR]:
        """Yield from nodes recursively from leaf->root."""
        yield instance
        for node in reversed(self):
            yield from node.iter_right(instance)

    def iter_output_name(self, instance: ExprIR, /) -> Iterator[ExprIR]:
        """Follow the **left-hand-side** of the graph until we can derive an output name.

        See `ExprIR.iter_output_name` for examples.
        """
        if self:
            yield from self[0].iter_output_name(instance)

    def is_scalar(self, instance: ExprIR, /) -> bool:
        # NOTE: Acrobatics because `all(...)` returns True on an empty iterable,
        # and there's 2 ways we can get there
        it = (
            node.is_scalar(instance) for node in self if node.observe_scalar is _OBSERVE
        )
        return (next(it, False)) and all(it)

    def map_ir(self, instance: ExprIR, function: MapIR, /) -> ExprIR:
        if self and (
            changes := {
                node.name: change
                for node in self
                if (change := node.map_nodes(instance, function))
            }
        ):
            instance = instance.__replace__(**changes)
        return function(instance)

    # NOTE: `Sequence[_ExprNode]` API
    def __getitem__(self, key: SupportsIndex, /) -> _ExprNode:
        return self._nodes.__getitem__(key)

    def __iter__(self) -> Iterator[_ExprNode]:
        yield from self._nodes

    def __len__(self) -> int:
        return self._nodes.__len__()

    def __reversed__(self) -> Iterator[_ExprNode]:
        if self:
            yield from reversed(self._nodes)
