from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TypeVar

from narwhals.dtypes import DType, Field

if TYPE_CHECKING:
    from collections.abc import Iterator


T_co = TypeVar("T_co", covariant=True)


def _iter_descendants(*bases: type[T_co]) -> Iterator[type[T_co]]:
    for base in bases:
        if children := base.__subclasses__():
            yield from _iter_descendants(*children)
        else:
            yield base


def iter_unslotted_classes(*bases: type[T_co]) -> Iterator[str]:
    """Find classes in that inherit from `bases` but don't define `__slots__`."""
    for tp in sorted(set(_iter_descendants(*bases)), key=repr):
        if "__slots__" not in tp.__dict__:
            yield f"{tp.__module__}.{tp.__name__}"


ret = 0
unslotted_classes = tuple(iter_unslotted_classes(DType, Field))

if unslotted_classes:
    ret = 1
    msg = "The following classes are expected to define `__slots__` but they don't:\n"
    cls_list = "\n".join(f"  * {name}" for name in unslotted_classes)
    print(f"{msg}{cls_list}")

sys.exit(ret)
