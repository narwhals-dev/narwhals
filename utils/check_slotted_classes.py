from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TypeVar

from narwhals._utils import Version, qualified_type_name
from narwhals.dtypes import DType, Field

if TYPE_CHECKING:
    from collections.abc import Iterator


T_co = TypeVar("T_co", covariant=True)

# NOTE: For `__subclasses__` to work, all modules that descendants are defined in must be imported
_ = Version.MAIN.dtypes
_ = Version.V1.dtypes
_ = Version.V2.dtypes


def _iter_descendants(*bases: type[T_co]) -> Iterator[type[T_co]]:
    seen = set[T_co]()
    for base in bases:
        yield base
        if (children := (base.__subclasses__())) and (
            unseen := set(children).difference(seen)
        ):
            yield from _iter_descendants(*unseen)


def iter_unslotted_classes(*bases: type[T_co]) -> Iterator[str]:
    """Find classes in that inherit from `bases` but don't define `__slots__`."""
    for tp in sorted(set(_iter_descendants(*bases)), key=qualified_type_name):
        if "__slots__" not in tp.__dict__:
            yield qualified_type_name(tp)


ret = 0
unslotted_classes = tuple(iter_unslotted_classes(DType, Field))

if unslotted_classes:
    ret = 1
    msg = "The following classes are expected to define `__slots__` but they don't:\n"
    cls_list = "\n".join(f"  * {name}" for name in unslotted_classes)
    url = "https://docs.python.org/3/reference/datamodel.html#slots"
    hint = f"Hint: See for detail {url!r}"
    print(f"{msg}{cls_list}")

sys.exit(ret)
