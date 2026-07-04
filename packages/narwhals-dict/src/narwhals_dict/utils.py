from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from narwhals._utils import Version
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType
    from narwhals_dict.typing import NativeSeries

__all__ = [
    "binary_op",
    "infer_dtype",
    "is_native_column",
    "is_native_frame",
    "py_type_for_dtype",
]


def is_native_column(obj: Any) -> bool:
    """A native column is any non-string sequence, `None` marks a null."""
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def is_native_frame(obj: Any) -> bool:
    return isinstance(obj, dict) and all(
        isinstance(name, str) and is_native_column(column) for name, column in obj.items()
    )


# NOTE: `bool` before `int`, `datetime` before `date` (subclass relationships).
_PY_TYPE_TO_DTYPE_NAME: tuple[tuple[type[Any], str], ...] = (
    (bool, "Boolean"),
    (int, "Int64"),
    (float, "Float64"),
    (str, "String"),
    (datetime, "Datetime"),
    (date, "Date"),
    (time, "Time"),
    (timedelta, "Duration"),
    (bytes, "Binary"),
)


def infer_dtype(values: Iterable[Any], version: Version) -> DType:
    """Infer the Narwhals dtype from the first non-null Python value."""
    dtypes = version.dtypes
    for value in values:
        if value is None:
            continue
        for py_type, dtype_name in _PY_TYPE_TO_DTYPE_NAME:
            if isinstance(value, py_type):
                return getattr(dtypes, dtype_name)()
        if is_native_column(value):
            return dtypes.List(infer_dtype(value, version))
        return dtypes.Unknown()
    return dtypes.Unknown()


def py_type_for_dtype(dtype: IntoDType, version: Version) -> type[Any] | None:
    """Map a Narwhals dtype to the Python type used to cast values, if supported."""
    dtypes = version.dtypes
    dtype = dtype() if isinstance(dtype, type) else dtype
    if dtype.is_integer():
        return int
    if dtype.is_float():
        return float
    if dtype == dtypes.String:
        return str
    if dtype == dtypes.Boolean:
        return bool
    return None


def binary_op(
    op: Callable[[Any, Any], Any], left: NativeSeries, right: Any, *, is_scalar: bool
) -> list[Any]:
    """Elementwise binary operation with null (`None`) propagation."""
    if is_scalar:
        if right is None:
            return [None] * len(left)
        return [None if lhs is None else op(lhs, right) for lhs in left]
    if len(left) != len(right):
        from narwhals.exceptions import ShapeError

        msg = f"Expected object of length {len(left)}, got: {len(right)}."
        raise ShapeError(msg)
    return [
        None if (lhs is None or rhs is None) else op(lhs, rhs)
        for lhs, rhs in zip(left, right, strict=True)
    ]
