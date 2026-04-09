from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._utils import isinstance_or_issubclass, qualified_type_name
from narwhals.dtypes._classes import DType, DTypeClass, NestedType

if TYPE_CHECKING:
    from typing import Any

    from typing_extensions import TypeIs

    from narwhals.typing import IntoDType


def validate_dtype(dtype: DType | type[DType]) -> None:
    if not isinstance_or_issubclass(dtype, DType):
        msg = (
            f"Expected Narwhals dtype, got: {type(dtype)}.\n\n"
            "Hint: if you were trying to cast to a type, use e.g. nw.Int64 instead of 'int64'."
        )
        raise TypeError(msg)


def is_into_dtype(obj: Any) -> TypeIs[IntoDType]:
    return isinstance(obj, DType) or (
        isinstance(obj, DTypeClass) and not issubclass(obj, NestedType)
    )


def is_nested_type(obj: Any) -> TypeIs[type[NestedType]]:
    return isinstance(obj, DTypeClass) and issubclass(obj, NestedType)


def validate_into_dtype(dtype: Any) -> None:
    if not is_into_dtype(dtype):
        if is_nested_type(dtype):
            name = f"nw.{dtype.__name__}"
            msg = (
                f"{name!r} is not valid in this context.\n\n"
                f"Hint: instead of:\n\n"
                f"    {name}\n\n"
                "use:\n\n"
                f"    {name}(...)"
            )
        else:
            msg = f"Expected Narwhals dtype, got: {qualified_type_name(dtype)!r}."
        raise TypeError(msg)
