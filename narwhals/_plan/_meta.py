"""Metaclasses and other unholy metaprogramming nonsense."""

from __future__ import annotations

# ruff: noqa: N806
from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Final, TypeVar

    import _typeshed
    from typing_extensions import dataclass_transform

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
