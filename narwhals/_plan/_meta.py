"""Metaclasses and other unholy metaprogramming nonsense."""

from __future__ import annotations

# ruff: noqa: N806
from itertools import chain
from typing import TYPE_CHECKING

from narwhals._plan import _nodes
from narwhals._plan._nodes import _EXPR_NODE_TYPES

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Final, TypeVar

    import _typeshed
    from typing_extensions import dataclass_transform

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


# towards https://github.com/narwhals-dev/narwhals/commit/4b0431a234808450a61d8b5260c8769f8cebff7b
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
    - It is used as the `metaclass` hint for a single base class named `ExprIR`, which all other
      callers of `ExprIRMeta.__new__` inherit from
    - Only the types returned by `field_specifiers` may be present in both `__slots__` and `namespace`


    [most derived metaclass]: https://docs.python.org/3/reference/datamodel.html#determining-the-appropriate-metaclass
    """

    def __new__(
        metacls: type[_typeshed.Self],
        cls_name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwds: Any,
    ) -> _typeshed.Self:
        nodes, namespace = _expr_meta_massage_namespace(cls_name, namespace)
        tp = super().__new__(metacls, cls_name, bases, namespace, **kwds)  # type: ignore[misc]
        if not TYPE_CHECKING:  # noqa: SIM102
            # `pyright` is quite unhappy w/ this, thinks that `Self` means `object` but it is a `type`!
            # `mypy` forgets how `TYPE_CHECKING` works when multiple conditions appear
            if nodes:
                _inherit_traverser(metacls, tp, nodes)
        return tp  # type: ignore[no-any-return]


def _expr_meta_massage_namespace(
    cls_name: str, namespace: dict[str, Any]
) -> tuple[_nodes.IntoExprNodes, dict[str, Any]]:
    """What on earth is this?

    The steps after this are needed if the new class used the following syntax:

        class Subclass(ExprIR):
            __slots__ = ("expr", "non_node")
            #             ^^^^    (1)
            expr: ExprIR = node()
            #            ^ ^^^^^^ (2)
            non_node: str

    Because `expr` appears both in `__slots__` (1) and in `__dict__` (2),
    we would normally see a runtime error:

        # TODO @dangotbanned: Add error output

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
            # ^                                         | ExprTraverser[1]
            # |                                         |     expr: ExprIR = node()
            # has a repr that looks like the source definition
    """
    slots: tuple[str, ...]
    nodes = []
    if cls_name == "ExprIR":
        namespace["__expr_ir_nodes__"] = _nodes.ExprTraverser(())

    # - The order defined in `namespace` is significant and must be preserved
    #   when moving to `__expr_ir_nodes__`
    # - The order in `__slots__` is probably different ([RUF023]), so we *may*
    #   need to do a (relatively) slow `namespace`-ordered search
    # - Since this code will run at import-time, there are **4** guards to stop as early as possible
    # [RUF023]: https://docs.astral.sh/ruff/rules/unsorted-dunder-slots/

    # (1): If there were no new slots, or
    # (2): All new slots did not *also* assign to `__dict__` then we're done
    elif (slots := namespace.get("__slots__", ())) and (
        intersection := namespace.keys() & slots
    ):
        # NOTE: ^^^ These acrobatics gave us the right names, but no guarantee on order ...

        # (3): ... but who needs order anyway?
        if len(intersection) == 1:
            name = intersection.pop()
            nodes.append(_ensure_node(name, namespace.pop(name)))
            return nodes, namespace

        # NOTE: ... I do!
        # We need ownership for the iterator as `namespace` is going to be mutated at least twice
        for name in tuple(namespace):
            if name in intersection:
                nodes.append(_ensure_node(name, namespace.pop(name)))
                # (4): In most cases `len(intersection) == 1`, but here will be <= 3`
                #      Whereas these classes have `len(namespace) >= 8`
                intersection.remove(name)
                if not intersection:
                    break
    return nodes, namespace


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
