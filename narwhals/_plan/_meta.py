"""Metaclasses and other unholy metaprogramming nonsense."""

from __future__ import annotations

# ruff: noqa: N806
from itertools import chain
from typing import TYPE_CHECKING, Any

from narwhals._plan import _nodes
from narwhals._plan._nodes import _EXPR_NODE_TYPES

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Final, TypeVar

    import _typeshed
    from typing_extensions import TypeAlias, dataclass_transform

    from narwhals._plan import _expr_ir2
    from narwhals._plan.typing import Seq

    T = TypeVar("T")

else:
    # https://docs.python.org/3/library/typing.html#typing.dataclass_transform
    def dataclass_transform(
        *,
        eq_default: bool = True,
        order_default: bool = False,
        kw_only_default: bool = False,
        frozen_default: bool = False,
        field_specifiers: tuple[type[Any] | Callable[..., Any], ...] = (),
        **kwargs: Any,
    ) -> Callable[[T], T]:
        def decorator(cls_or_fn: T) -> T:
            cls_or_fn.__dataclass_transform__ = {
                "eq_default": eq_default,
                "order_default": order_default,
                "kw_only_default": kw_only_default,
                "frozen_default": frozen_default,
                "field_specifiers": field_specifiers,
                "kwargs": kwargs,
            }
            return cls_or_fn

        return decorator


__all__ = ["ImmutableMeta", "SlottedMeta", "dataclass_transform"]

Ns: TypeAlias = dict[str, Any]
"""The class `namespace`.

Marked so that mutating it is visible.
"""

flatten = chain.from_iterable
_KEYS_NAME: Final = "__immutable_keys__"
_HASH_NAME: Final = "__immutable_hash_value__"


class SlottedMeta(type):
    """Ensure [`__slots__`] are always defined to prevent `__dict__` creation.

    [`__slots__`]: https://docs.python.org/3/reference/datamodel.html#object.__slots__
    """

    # https://github.com/python/typeshed/blob/776508741d76b58f9dcb2aaf42f7d4596a48d580/stdlib/abc.pyi#L13-L19
    # https://github.com/python/typeshed/blob/776508741d76b58f9dcb2aaf42f7d4596a48d580/stdlib/_typeshed/__init__.pyi#L36-L40
    # https://github.com/astral-sh/ruff/issues/8353#issuecomment-1786238311
    # https://docs.python.org/3/reference/datamodel.html#creating-the-class-object
    def __new__(
        metacls: type[_typeshed.Self],
        cls_name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwds: Any,
    ) -> _typeshed.Self:
        namespace.setdefault("__slots__", ())
        return super().__new__(metacls, cls_name, bases, namespace, **kwds)  # type: ignore[no-any-return, misc]


@dataclass_transform(kw_only_default=True, frozen_default=True)
class ImmutableMeta(SlottedMeta):
    def __new__(
        metacls: type[_typeshed.Self],
        cls_name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwds: Any,
    ) -> _typeshed.Self:
        KEYS, HASH = _KEYS_NAME, _HASH_NAME
        getattr_: Callable[..., Seq[str]] = getattr
        it_bases = (getattr_(b, KEYS, ()) for b in bases)
        it_all = chain(
            flatten(it_bases), namespace.get(KEYS, namespace.get("__slots__", ()))
        )
        namespace[KEYS] = tuple(key for key in it_all if key != HASH)
        return super().__new__(metacls, cls_name, bases, namespace, **kwds)  # type: ignore[no-any-return, misc]


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(_nodes.node, _nodes.nodes),
)
class ExprIRMeta(ImmutableMeta):
    """Metaclass for `ExprIR`.

    ## Important
    `ExprIRMeta` is intended to work within the following (narrow) constraints:
    - It is the [most derived metaclass]
      - You can use `Generic`, but cannot use `Protocol`
    - It is used as the `metaclass` hint for a single base class (`ExprIR`), which all other
      callers of `ExprIRMeta.__new__` inherit from
    - Only the types returned by `field_specifiers` may be present in both `__slots__` and `namespace`

    [most derived metaclass]: https://docs.python.org/3/reference/datamodel.html#determining-the-appropriate-metaclass
    """

    def __new__(
        metacls: type[_typeshed.Self],
        cls_name: str,
        bases: tuple[type, ...],
        namespace: Ns,
        /,
        **kwds: Any,
    ) -> _typeshed.Self:
        namespace, nodes = ExprIRMeta._pop_nodes(namespace)
        tp = super().__new__(metacls, cls_name, bases, namespace, **kwds)  # type: ignore[misc]
        if not TYPE_CHECKING:  # noqa: SIM102
            # `pyright` is quite unhappy w/ this, thinks that `Self` means `object` but it is a `type`!
            # `mypy` forgets how `TYPE_CHECKING` works when multiple conditions appear
            if nodes:
                _inherit_traverser(metacls, tp, nodes)
        return tp  # type: ignore[no-any-return]

    @staticmethod
    def _pop_nodes(namespace: Ns) -> tuple[Ns, _nodes.IntoExprNodes]:
        """Extract any field specifiers from the class namespace.

        ## What?
        These steps are needed *iff* the new class used the following syntax:

            class Subclass(ExprIR):
                __slots__ = ("expr", "non_node")
                #             ^^^^
                expr: ExprIR = node()
                #            ^ ^^^^^^
                non_node: str

        Because `expr` appears both in `__slots__` and in `__dict__`, we would normally see:

            ValueError: 'expr' in __slots__ conflicts with class variable

        However, what we actually want is to transform the definition into this:

            class Subclass(ExprIR):
                __slots__ = ("expr", "non_node")
                #             ^^^^
                #             The slot is for the `ExprIR`
                expr: ExprIR
                #            ^ ^^^^^^
                #            Poof! The assignment has vanished. We make the connection
                #            between `"expr"` and the field specifier `node()` and
                #            move it to `__expr_ir_nodes__` ...
                non_node: str

                __expr_ir_nodes__: ClassVar[ExprTraverser] = ...

        Our end result has a nice repr that looks a lot like our source code:

            Subclass.__expr_ir_nodes__
            ExprTraverser[1]
                expr: ExprIR = node()

        ## How?
        A simple, unoptimized version might look like this:

            def pop_nodes(namespace: Ns) -> tuple[Ns, list[Any]]:
                slots = namespace.get("__slots__", ())
                pop = namespace.pop
                nodes = [
                    pop(name).with_name(name)
                    for name in tuple(namespace)
                    if name in slots
                ]
                return namespace, nodes

        *But*, since this code will be [executed for each new subclass] - we take 4 steps
        to avoid hogging cycles at import-time.

        If either of these are true, we have no work to do:
        1. we *don't* have new `__slots__`
        2. we do, but they *don't* intersect with `namespace`

        If we have some work, we can avoid iterating over `namespace` *iff*:

        3. Our intersection is a single name

        Surprisingly, that covers 87% of cases (34/39) *already*.

        For everything else (4), we do a `namespace`-ordered pass
        but work destructively on the intersection to stop as early as we can.

        [executed for each new subclass]: https://docs.python.org/3/reference/datamodel.html#class-object-creation
        """
        slots: tuple[str, ...]
        nodes = []
        # Steps 1, 2
        if (slots := namespace.get("__slots__", ())) and (
            intersection := namespace.keys() & slots
        ):
            # Step 3
            if len(intersection) == 1:
                name = intersection.pop()
                nodes.append(_ensure_node(name, namespace.pop(name)))
                return namespace, nodes

            # Step 4
            for name in tuple(namespace):
                if name in intersection:
                    nodes.append(_ensure_node(name, namespace.pop(name)))
                    intersection.remove(name)
                    if not intersection:
                        break
        return namespace, nodes


def _ensure_node(name: str, node: Any) -> _nodes._ExprNode:
    if not isinstance(node, _EXPR_NODE_TYPES):
        msg = f"Expected field specifier of type {_EXPR_NODE_TYPES!r}, got:\n`{name}={node!r}`"
        raise TypeError(msg)
    return node.with_name(name)


def _inherit_traverser(
    metacls: type[ExprIRMeta], cls: type[_expr_ir2.ExprIR], extra: _nodes.IntoExprNodes
) -> None:
    traverser = _nodes.ExprTraverser.inherit_from(cls.__expr_ir_nodes__, extra)
    metacls.__setattr__(cls, "__expr_ir_nodes__", traverser)
