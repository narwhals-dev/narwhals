from __future__ import annotations

import inspect
import sys
from itertools import chain
from typing import TYPE_CHECKING

import narwhals.dtypes as dtypes_main
import narwhals.stable.v1.dtypes as v1_dtypes
import narwhals.stable.v2.dtypes as v2_dtypes

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType, UnionType

    from typing_extensions import TypeAlias

    if sys.version_info >= (3, 10):
        _ClassInfo: TypeAlias = type | UnionType | tuple["_ClassInfo", ...]
    else:
        _ClassInfo: TypeAlias = type | tuple["_ClassInfo", ...]


base_dtype = dtypes_main.DType
field_type = dtypes_main.Field


def get_unslotted_classes(
    module: ModuleType, bases: _ClassInfo
) -> Iterator[tuple[str, ModuleType]]:
    """Find classes in a `module` that inherit from `bases` but don't define `__slots__`."""
    return (
        (name, module)
        for name, cls in inspect.getmembers(module)
        if isinstance(cls, type)
        and issubclass(cls, bases)
        and "__slots__" not in cls.__dict__
    )


ret = 0
unslotted_classes = tuple(
    chain.from_iterable(
        get_unslotted_classes(mod, bases=(base_dtype, field_type))
        for mod in (dtypes_main, v1_dtypes, v2_dtypes)
    )
)

if unslotted_classes:
    ret = 1
    msg = "The following classes are expected to define `__slots__` but they don't:\n"
    cls_list = "\n".join(f"  * {name} from {mod}" for name, mod in unslotted_classes)
    print(f"{msg}{cls_list}")

sys.exit(ret)
