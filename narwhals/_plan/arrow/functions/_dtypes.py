"""Native data types, conversion and casting."""

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

__all__ = [  # noqa: RUF022
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
    # Not to be exported to `functions.__all__`
    "is_integer",
    "is_large_string",
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

is_integer: Final = pa.types.is_integer
is_large_string: Final = pa.types.is_large_string


@overload
def dtype_native(dtype: IntoDType, /, version: Version) -> DataType: ...
@overload
def dtype_native(dtype: None, /, version: Version) -> None: ...
@overload
def dtype_native(dtype: IntoDType | None, /, version: Version) -> DataType | None: ...
def dtype_native(dtype: IntoDType | None, /, version: Version) -> DataType | None:
    """Convert a Narwhals `DType` to a `pyarrow.DataType`, or passthrough `None`."""
    return dtype if dtype is None else _dtype_native(dtype, version)


@overload
def cast(native: Scalar[Any], dtype: DataTypeT, /) -> Scalar[DataTypeT]: ...
@overload
def cast(
    native: ChunkedArray[Any], dtype: DataTypeT, /
) -> ChunkedArray[Scalar[DataTypeT]]: ...
@overload
def cast(
    native: ChunkedOrScalar[Scalar[Any]], dtype: DataTypeT, /
) -> ChunkedArray[Scalar[DataTypeT]] | Scalar[DataTypeT]: ...
def cast(
    native: ChunkedOrScalar[Scalar[Any]], dtype: DataTypeT, /
) -> ChunkedArray[Scalar[DataTypeT]] | Scalar[DataTypeT]:
    """Cast arrow data to the specified dtype."""
    return pc.cast(native, dtype)


def cast_table(
    native: pa.Table, dtypes: DataType | IntoArrowSchema | DataTypeRemap, /
) -> pa.Table:
    """Cast Table column(s) to the specified dtype(s).

    Similar to [`pl.DataFrame.cast`].

    Arguments:
        native: An arrow table.
        dtypes: Mapping of column names (or dtypes) to dtypes, or a single dtype
            to which all columns will be cast.

    [`pl.DataFrame.cast`]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.cast.html#polars.DataFrame.cast
    """
    s = dtypes if isinstance(dtypes, pa.Schema) else _cast_schema(native.schema, dtypes)
    return native.cast(s)


def string_type(dtypes: Iterable[DataType] = (), /) -> StringType:
    """Return a native string type, compatible with `dtypes`.

    Until [apache/arrow#45717] is resolved, we need to upcast `string` to `large_string` when joining.

    [apache/arrow#45717]: https://github.com/apache/arrow/issues/45717
    """
    return pa.large_string() if any(is_large_string(tp) for tp in dtypes) else pa.string()


def _cast_schema(
    native: pa.Schema, dtypes: DataType | Mapping[str, DataType] | DataTypeRemap
) -> pa.Schema:
    if isinstance(dtypes, pa.DataType):
        return pa.schema((name, dtypes) for name in native.names)
    if _is_into_pyarrow_schema(dtypes):
        new_schema = native
        for name, dtype in dtypes.items():
            index = native.get_field_index(name)
            new_schema.set(index, native.field(index).with_type(dtype))
        return new_schema
    return pa.schema((fld.name, dtypes.get(fld.type, fld.type)) for fld in native)


def _is_into_pyarrow_schema(obj: Mapping[Any, Any]) -> TypeIs[Mapping[str, DataType]]:
    return (
        (first := next(iter(obj.items())), None)
        and isinstance(first[0], str)
        and isinstance(first[1], pa.DataType)
    )
