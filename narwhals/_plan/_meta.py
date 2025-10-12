"""Metaclasses and other unholy metaprogramming nonsense."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import _typeshed


class SlottedMeta(type):
    # https://github.com/python/typeshed/blob/776508741d76b58f9dcb2aaf42f7d4596a48d580/stdlib/abc.pyi#L13-L19
    # https://github.com/python/typeshed/blob/776508741d76b58f9dcb2aaf42f7d4596a48d580/stdlib/_typeshed/__init__.pyi#L36-L40
    # https://github.com/astral-sh/ruff/issues/8353#issuecomment-1786238311
    def __new__(
        mcls: type[_typeshed.Self],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwds: Any,
    ) -> _typeshed.Self:
        namespace.setdefault("__slots__", ())
        return super().__new__(mcls, name, bases, namespace, **kwds)  # type: ignore[no-any-return, misc]
