from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import narwhals_to_native_dtype as _dtype_native

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from typing_extensions import TypeIs

    from narwhals._plan.arrow.typing import (
        ChunkedArray,
        ChunkedOrScalar,
        DataType,
        DataTypeRemap,
        DataTypeT,
        Scalar,
        StringType,
    )
    from narwhals._utils import Version
    from narwhals.typing import IntoArrowSchema, IntoDType

__all__ = [
    "BOOL",
    "DATE",
    "F64",
    "I32",
    "I64",
    "U32",
    "cast",
    "cast_table",
    "dtype_native",
    "string_type",
]

# NOTE: Common data type instances to share.
# Names use an uppercase equivalent to [short repr codes]
# (https://github.com/pola-rs/polars/blob/5deaf7e9074fdc8f7f0082974cc956acf645af62/crates/polars-core/src/datatypes/dtype.rs#L1127-L1187)
U32: Final = pa.uint32()
I32: Final = pa.int32()
I64: Final = pa.int64()
F64: Final = pa.float64()
BOOL: Final = pa.bool_()
DATE: Final = pa.date32()


@overload
def dtype_native(dtype: IntoDType, version: Version) -> DataType: ...
@overload
def dtype_native(dtype: None, version: Version) -> None: ...
@overload
def dtype_native(dtype: IntoDType | None, version: Version) -> DataType | None: ...
def dtype_native(dtype: IntoDType | None, version: Version) -> DataType | None:
    return dtype if dtype is None else _dtype_native(dtype, version)


@overload
def cast(
    native: Scalar[Any], target_type: DataTypeT, *, safe: bool | None = ...
) -> Scalar[DataTypeT]: ...
@overload
def cast(
    native: ChunkedArray[Any], target_type: DataTypeT, *, safe: bool | None = ...
) -> ChunkedArray[Scalar[DataTypeT]]: ...
@overload
def cast(
    native: ChunkedOrScalar[Scalar[Any]],
    target_type: DataTypeT,
    *,
    safe: bool | None = ...,
) -> ChunkedArray[Scalar[DataTypeT]] | Scalar[DataTypeT]: ...
def cast(
    native: ChunkedOrScalar[Scalar[Any]],
    target_type: DataTypeT,
    *,
    safe: bool | None = None,
) -> ChunkedArray[Scalar[DataTypeT]] | Scalar[DataTypeT]:
    return pc.cast(native, target_type, safe=safe)


def cast_table(
    native: pa.Table, target: DataType | IntoArrowSchema | DataTypeRemap
) -> pa.Table:
    s = target if isinstance(target, pa.Schema) else _cast_schema(native.schema, target)
    return native.cast(s)


def string_type(data_types: Iterable[DataType] = (), /) -> StringType:
    """Return a native string type, compatible with `data_types`.

    Until [apache/arrow#45717] is resolved, we need to upcast `string` to `large_string` when joining.

    [apache/arrow#45717]: https://github.com/apache/arrow/issues/45717
    """
    return (
        pa.large_string()
        if any(pa.types.is_large_string(tp) for tp in data_types)
        else pa.string()
    )


def _cast_schema(
    native: pa.Schema, target_types: DataType | Mapping[str, DataType] | DataTypeRemap
) -> pa.Schema:
    if isinstance(target_types, pa.DataType):
        return pa.schema((name, target_types) for name in native.names)
    if _is_into_pyarrow_schema(target_types):
        new_schema = native
        for name, dtype in target_types.items():
            index = native.get_field_index(name)
            new_schema.set(index, native.field(index).with_type(dtype))
        return new_schema
    return pa.schema((fld.name, target_types.get(fld.type, fld.type)) for fld in native)


def _is_into_pyarrow_schema(obj: Mapping[Any, Any]) -> TypeIs[Mapping[str, DataType]]:
    return (
        (first := next(iter(obj.items())), None)
        and isinstance(first[0], str)
        and isinstance(first[1], pa.DataType)
    )
