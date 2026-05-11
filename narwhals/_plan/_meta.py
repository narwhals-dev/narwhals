"""Metaclasses and other unholy metaprogramming nonsense."""

# ruff: noqa: N806
from __future__ import annotations

from collections import deque
from itertools import chain
from typing import TYPE_CHECKING, Any

from narwhals._plan import _nodes

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Final, TypeVar

    from typing_extensions import TypeAlias, dataclass_transform

    from narwhals._plan._nodes import ExprTraverser
    from narwhals._plan.typing import Seq

    T = TypeVar("T")

    # NOTE: Similar to `_typeshed.Self`
    # https://github.com/astral-sh/ruff/issues/8353#issuecomment-1786238311
    # https://github.com/python/typeshed/blob/f8f0794d0fe249c06dc9f31a004d85be6cca6ced/stdlib/_typeshed/__init__.pyi#L36-L40
    # https://github.com/python/typeshed/blob/f8f0794d0fe249c06dc9f31a004d85be6cca6ced/stdlib/abc.pyi#L13-L23
    S = TypeVar("S", bound="SlottedMeta")
    I = TypeVar("I", bound="ImmutableMeta")  # noqa: E741
    E = TypeVar("E", bound="ExprIRMeta")

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


__all__ = ("ExprIRMeta", "ImmutableMeta", "SlottedMeta")

Ns: TypeAlias = dict[str, Any]
"""The class `namespace`.

Marked so that mutating it is visible.
"""

flatten = chain.from_iterable
_KEYS_NAME: Final = "__immutable_keys__"
_HASH_NAME: Final = "__immutable_hash_value__"
_NODES_NAME: Final = "__expr_ir_nodes__"


class SlottedMeta(type):
    """Metaclass ensuring [`__slots__`] are always defined.

    Consider using this metaclass if you have something like:
    >>> class Base:
    ...     __slots__ = ("a", "b")
    ...     a: int
    ...     b: int
    >>> class Sub1(Base):
    ...     __slots__ = ()
    >>> class Sub2(Base):
    ...     __slots__ = ("c",)
    ...     c: int
    >>> class Sub21(Sub2):
    ...     __slots__ = ()

    But you'd prefer to write this:
    >>> class Base(metaclass=SlottedMeta):
    ...     __slots__ = ("a", "b")
    ...     a: int
    ...     b: int
    >>> class Sub1(Base): ...
    >>> class Sub2(Base):
    ...     __slots__ = ("c",)
    ...     c: int
    >>> class Sub21(Sub2): ...

    And still [avoid these issues]:
    >>> any(hasattr(tp(), "__dict__") for tp in (Base, Sub1, Sub2, Sub21))
    False

    [`__slots__`]: https://docs.python.org/3/reference/datamodel.html#object.__slots__
    [avoid these issues]: https://dev.arie.bovenberg.net/blog/finding-broken-slots-in-popular-python-libraries/
    """

    def __new__(
        metacls: type[S],
        cls_name: str,
        bases: tuple[type, ...],
        namespace: Ns,
        /,
        **kwds: Any,
    ) -> S:
        namespace.setdefault("__slots__", ())
        return super().__new__(metacls, cls_name, bases, namespace, **kwds)


@dataclass_transform(kw_only_default=True, frozen_default=True)
class ImmutableMeta(SlottedMeta):
    """Metaclass for `Immutable`.

    We do two things here ...

    ## 1. Introduce [`@dataclass_transform`]
    This let's type checkers understand what subclasses of `Immutable` can (and can't!) do.

    Note that subclasses of `ImmutableMeta` must repeat the decorator, even if it uses
    the same arguments.

    ## 2. Track what is a *"field"*
    The rest is some housekeeping of `__slots__` -> `__immutable_keys__`, which avoids needing
    to inspect `__annotations__` for field names.

    [`@dataclass_transform`]: https://typing.python.org/en/latest/spec/dataclasses.html#the-dataclass-transform-decorator
    """

    def __new__(
        metacls: type[I],
        cls_name: str,
        bases: tuple[type, ...],
        namespace: Ns,
        /,
        **kwds: Any,
    ) -> I:
        KEYS, HASH = _KEYS_NAME, _HASH_NAME
        getattr_: Callable[..., Seq[str]] = getattr
        it_bases = (getattr_(b, KEYS, ()) for b in bases)
        it_all = chain(
            flatten(it_bases),
            namespace.get(KEYS, namespace.get("__slots__", ())),  # type: ignore[arg-type]
        )
        namespace[KEYS] = tuple(key for key in it_all if key != HASH)
        return super().__new__(metacls, cls_name, bases, namespace, **kwds)


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

    if TYPE_CHECKING:
        # NOTE: Refers to `ExprIR.__expr_ir_nodes__: ClassVar[ExprTraverser]`
        __expr_ir_nodes__: ExprTraverser

    def __new__(
        metacls: type[E],
        cls_name: str,
        bases: tuple[type, ...],
        namespace: Ns,
        /,
        **kwds: Any,
    ) -> E:
        namespace, nodes = metacls._pop_nodes(cls_name, namespace)
        tp = super().__new__(metacls, cls_name, bases, namespace, **kwds)
        if nodes:
            metacls.__setattr__(tp, _NODES_NAME, tp.__expr_ir_nodes__.extend_with(nodes))
        return tp

    # TODO @dangotbanned: Move out of metaclass, since `ExprIR` also gets this as a `staticmethod`
    @staticmethod
    def _pop_nodes(cls_name: str, namespace: Ns) -> tuple[Ns, _nodes.IntoTraverser]:
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
            ns_pop = namespace.pop
            # Step 3
            if len(intersection) == 1:
                name = intersection.pop()
                nodes.append(_nodes.into_expr_node(name, ns_pop(name), cls_name))
                return namespace, nodes

            # Step 4
            names = deque(namespace)
            while intersection:
                name = names.popleft()
                if name in intersection:
                    nodes.append(_nodes.into_expr_node(name, ns_pop(name), cls_name))
                    intersection.remove(name)
        return namespace, nodes
