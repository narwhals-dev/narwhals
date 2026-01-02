"""Struct function namespace, and some helpers."""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING, Any, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan import common
from narwhals._plan.arrow import compat
from narwhals._plan.arrow.guards import is_arrow

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from narwhals._arrow.typing import Incomplete
    from narwhals._plan.arrow.acero import Field
    from narwhals._plan.arrow.typing import (
        ArrayAny,
        Arrow,
        ArrowAny,
        ChunkedArrayAny,
        ChunkedOrScalarAny,
        ChunkedStruct,
        SameArrowT,
        ScalarAny,
        StructArray,
    )
    from narwhals._plan.typing import Seq
    from narwhals.typing import NonNestedLiteral

__all__ = ["field", "field_names", "fields", "into_struct", "schema"]


@overload
def into_struct(
    names: Iterable[str], columns: Iterable[ChunkedArrayAny]
) -> ChunkedStruct: ...
@overload
def into_struct(names: Iterable[str], columns: Iterable[ArrayAny]) -> pa.StructArray: ...
# NOTE: `mypy` isn't happy, but this broadcasting behavior is worth documenting
@overload
def into_struct(  # type: ignore[overload-overlap]
    names: Iterable[str], columns: Iterable[ScalarAny] | Iterable[NonNestedLiteral]
) -> pa.StructScalar: ...
@overload
def into_struct(  # type: ignore[overload-overlap]
    names: Iterable[str], columns: Iterable[ChunkedArrayAny | NonNestedLiteral]
) -> ChunkedStruct: ...
@overload
def into_struct(
    names: Iterable[str], columns: Iterable[ArrayAny | NonNestedLiteral]
) -> pa.StructArray: ...
@overload
def into_struct(names: Iterable[str], columns: Iterable[ArrowAny]) -> Incomplete: ...
def into_struct(names: Iterable[str], columns: Iterable[Incomplete]) -> Incomplete:
    """Collect columns into a struct.

    Arguments:
        names: Name(s) to assign to each struct field.
        columns: Value(s) to collect into a struct. Scalars will will be broadcast unless all
            inputs are scalar.

    Note:
        Roughly [`polars.struct`] but `names` must be resolved ahead of time.

    [`polars.struct`]: https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.struct.html
    """
    return pc.make_struct(
        *columns, options=pc.MakeStructOptions(common.ensure_seq_str(names))
    )


def schema(native: Arrow[pa.StructScalar] | pa.StructType, /) -> pa.Schema:
    """Get the struct definition as a schema.

    Arguments:
        native: Struct-typed arrow data, or a `StructType` *itself*.
    """
    tp = native.type if is_arrow(native) else native
    fields = tp.fields if compat.HAS_STRUCT_TYPE_FIELDS else list(tp)
    return pa.schema(fields)


def field_names(native: Arrow[pa.StructScalar] | pa.StructType, /) -> list[str]:
    """Get the names of each field in a struct.

    Arguments:
        native: Struct-typed arrow data, or a `StructType` *itself*.
    """
    tp = native.type if is_arrow(native) else native
    return tp.names if compat.HAS_STRUCT_TYPE_FIELDS else [f.name for f in tp]


@overload
def field(native: ChunkedStruct, name: Field, /) -> ChunkedArrayAny: ...
@overload
def field(native: StructArray, name: Field, /) -> ArrayAny: ...
@overload
def field(native: pa.StructScalar, name: Field, /) -> ScalarAny: ...
@overload
def field(native: SameArrowT, name: Field, /) -> SameArrowT: ...
@overload
def field(native: ChunkedOrScalarAny, name: Field, /) -> ChunkedOrScalarAny: ...
def field(native: ArrowAny, name: Field, /) -> ArrowAny:
    """Retrieve a single field from a struct as a new array/scalar.

    Arguments:
        native: Struct-typed arrow data.
        name: Name of the struct field to retrieve.
    """
    func = t.cast("Callable[[Any,Any], ArrowAny]", pc.struct_field)
    return func(native, name)


@overload
def fields(native: ChunkedStruct, *names: Field) -> Seq[ChunkedArrayAny]: ...
@overload
def fields(native: StructArray, *names: Field) -> Seq[ArrayAny]: ...
@overload
def fields(native: pa.StructScalar, *names: Field) -> Seq[ScalarAny]: ...
@overload
def fields(native: SameArrowT, *names: Field) -> Seq[SameArrowT]: ...
def fields(native: ArrowAny, *names: Field) -> Seq[ArrowAny]:
    """Retrieve multiple fields from a struct as new array/scalar(s).

    Arguments:
        native: Struct-typed arrow data.
        names: Names of the struct fields to retrieve.
    """
    func = t.cast("Callable[[Any,Any], ArrowAny]", pc.struct_field)
    return tuple(func(native, name) for name in names)
