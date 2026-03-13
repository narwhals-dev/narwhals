"""Experimental replacement for `ExprIR`.

Planning an (internally) large change to how the classes are defined.

Main difference to `ExprIR` itself, is that this:

    class ExprIR(Immutable):
        _child: ClassVar[tuple[str, ...]] = ()

        def __init_subclass__(cls, *, child: tuple[str, ...] = ()) -> None:
            if child:
                cls._child = child

        @property
        def is_scalar(self) -> bool:
            return False


    class Over(ExprIR, child=("expr", "partition_by")):
        __slots__ = ("expr", "partition_by")
        expr: ExprIR
        partition_by: Seq[ExprIR]


    class OverOrdered(Over, child=("expr", "partition_by", "order_by")):
        __slots__ = ("order_by", "sort_options")
        expr: ExprIR
        partition_by: Seq[ExprIR]
        order_by: Seq[ExprIR]
        sort_options: SortOptions


Is now expressed as:

    class ExprIR(Immutable, metaclass=ExprIRMeta):
        __expr_ir_nodes__: ClassVar[ExprTraverser]

        def is_scalar(self) -> bool:
            return self.__expr_ir_nodes__.is_scalar(self)


    class Over(ExprIR):
        __slots__ = ("expr", "partition_by")
        expr: ExprIR = node(observe_scalar=False)
        partition_by: Seq[ExprIR] = nodes()


    class OverOrdered(Over):
        __slots__ = ("order_by", "sort_options")
        order_by: Seq[ExprIR] = nodes()
        sort_options: SortOptions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic

from narwhals._plan._immutable import Immutable
from narwhals._plan._meta import ExprIRMeta
from narwhals._plan._nodes import ExprTraverser, MapIR, node, nodes
from narwhals._plan.options import SortMultipleOptions, SortOptions
from narwhals._plan.typing import OperatorT, SelectorOperatorT
from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan._dispatch import Dispatcher
    from narwhals._plan._dtype import ResolveDType
    from narwhals._plan.typing import Seq


class ExprIR(Immutable, metaclass=ExprIRMeta):
    __expr_ir_dispatch__: ClassVar[Dispatcher[Any]]
    __expr_ir_dtype__: ClassVar[ResolveDType[Any]]

    __expr_ir_nodes__: ClassVar[ExprTraverser]
    """Brand new fancy boi"""

    def iter_left(self) -> Iterator[ExprIR]:
        yield from self.__expr_ir_nodes__.iter_left(self)

    def iter_right(self) -> Iterator[ExprIR]:
        yield from self.__expr_ir_nodes__.iter_right(self)

    # TODO @dangotbanned: *Do* need to override on
    # - `Len`, `Literal.value`, `FunctionExpr.function`
    # NOTE: Don't need to override on False anymore (0 observing scalar == False)
    # - `Column`, `Filter`, `Over`, `OverOrdered`,`SelectorIR`
    def is_scalar(self) -> bool:
        return self.__expr_ir_nodes__.is_scalar(self)

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return self.__expr_ir_nodes__.map_ir(self, function)

    def needs_expansion(self) -> bool:
        return any(isinstance(e, SelectorIR) for e in self.iter_left())


class SelectorIR(ExprIR):
    def needs_expansion(self) -> bool:
        return True


class Column(ExprIR):
    __slots__ = ("name",)
    name: str


class Alias(ExprIR):
    __slots__ = ("expr", "name")
    expr: ExprIR = node()
    name: str


class SortBy(ExprIR):
    __slots__ = ("by", "expr", "options")
    expr: ExprIR = node()
    by: Seq[ExprIR] = nodes()
    options: SortMultipleOptions


class Over(ExprIR):
    __slots__ = ("expr", "partition_by")
    expr: ExprIR = node(observe_scalar=False)
    partition_by: Seq[ExprIR] = nodes()


class OverOrdered(Over):
    # very nice, this plays well with inheritance!
    __slots__ = ("order_by", "sort_options")
    order_by: Seq[ExprIR] = nodes()
    sort_options: SortOptions


LeftT = TypeVar("LeftT", bound="ExprIR", default="ExprIR")
RightT = TypeVar("RightT", bound="ExprIR", default="ExprIR")


class BinaryExpr(ExprIR, Generic[LeftT, OperatorT, RightT]):
    __slots__ = ("left", "op", "right")
    left: LeftT = node()
    op: OperatorT
    right: RightT = node()

    def __repr__(self) -> str:
        return f"[({self.left!r}) {self.op!r} ({self.right!r})]"


LeftSelectorT = TypeVar("LeftSelectorT", bound="SelectorIR", default="SelectorIR")
RightSelectorT = TypeVar("RightSelectorT", bound="SelectorIR", default="SelectorIR")


class BinarySelector(
    SelectorIR, Generic[LeftSelectorT, SelectorOperatorT, RightSelectorT]
):
    """Needs to deviate from using `_BinaryOp`.

    - `BinaryExpr(_BinaryOp): ...`
       - Slots defined in parent, but wants to use them for 2x `node`
    - `BinarySelector(SelectorIR, _BinaryOp): ...`
        - Slots defined in parent, and wants them to be ignored for traversal

    Sadly this means a tiny bit of code duplication
    """

    __slots__ = ("left", "op", "right")
    left: LeftSelectorT
    op: SelectorOperatorT
    right: RightSelectorT

    def __repr__(self) -> str:
        return f"[({self.left!r}) {self.op!r} ({self.right!r})]"


# ruff: noqa: F841
def what_does_type_checker_say() -> None:  # noqa: PLR0914
    """All looking like pyright understands the typing correctly!"""
    no_args_good = ExprIR()
    no_args_bad = Column()  # type: ignore[call-arg]
    one_arg_good = Column(name="hello")

    no_args_node_bad_0 = Alias()  # type: ignore[call-arg]
    no_args_node_bad_1_1 = Alias(expr=one_arg_good)  # type: ignore[call-arg]
    no_args_node_bad_1_2 = Alias(name="hello")  # type: ignore[call-arg]
    mixed_good = Alias(expr=Column(name="hello"), name="goodbye")
    mixed_non_node_bad = Alias(expr="hello", name="goodbye")  # type: ignore[arg-type]

    opts = SortMultipleOptions.default()
    _ = SortBy()  # type: ignore[call-arg]
    _ = SortBy(options=opts)  # type: ignore[call-arg]
    _ = SortBy(expr=one_arg_good, options=opts)  # type: ignore[call-arg]
    _ = SortBy(expr=one_arg_good, by=one_arg_good, options=opts)  # type: ignore[arg-type]
    _ = SortBy(expr=one_arg_good, by=(one_arg_good,), options=opts)

    expr_ir_nodes = SortBy.__expr_ir_nodes__
    a = Column(name="hello")
    b = Alias(expr=a, name="...")
    c = SortBy(expr=a, by=(a, b), options=opts)
    over = Over(expr=a, partition_by=(b, a))

    over_ordered = OverOrdered(
        expr=a,
        partition_by=(),
        order_by=(c, a),
        sort_options=SortOptions(descending=False, nulls_last=False),
    )
