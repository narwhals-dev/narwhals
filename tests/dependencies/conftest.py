from __future__ import annotations

from typing import Any

import pytest


class DynamicAttrOnly:
    """An object whose attributes only exist dynamically, never statically.

    Every attribute access is fabricated on the fly via `__getattr__`, so
    `hasattr` reports `True` for any name while [`inspect.getattr_static`]
    (and therefore `_hasattr_static`) correctly reports `False`:

        obj = DynamicAttrOnly()
        hasattr(obj, "i_dont_exist")          # True   (false positive)
        _hasattr_static(obj, "i_dont_exist")  # False  (correct)

    Use it in tests to prove that code under test relies on `_hasattr_static`
    rather than `hasattr` when probing unknown objects: any check that accepts
    a `DynamicAttrOnly` instance is, by construction, using the unsafe path.

    [`inspect.getattr_static`]: https://docs.python.org/3/library/inspect.html#inspect.getattr_static
    """

    def __getattr__(self, name: str) -> Any:
        return self


@pytest.fixture(scope="session")
def dynamic_attr_only() -> DynamicAttrOnly:
    """An object that triggers `hasattr` false positives for every attribute name."""
    return DynamicAttrOnly()
