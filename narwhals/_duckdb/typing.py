from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, Union, overload

import duckdb
from duckdb import Expression

from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    import uuid

    import numpy as np
    import pandas as pd
    from duckdb import DuckDBPyConnection
    from typing_extensions import TypeAlias, TypeIs

    from narwhals.typing import Into1DArray, PythonLiteral


__all__ = [
    "BaseType",
    "IntoColumnExpr",
    "WindowExpressionKwargs",
    "has_children",
    "is_dtype",
]

IntoDuckDBLiteral: TypeAlias = """
    PythonLiteral
    | dict[Any, Any]
    | uuid.UUID
    | bytearray
    | memoryview
    | Into1DArray
    | pd.api.typing.NaTType
    | pd.api.typing.NAType
    | np.ma.MaskedArray
    | duckdb.Value
    """


class WindowExpressionKwargs(TypedDict, total=False):
    partition_by: Sequence[str | Expression]
    order_by: Sequence[str | Expression]
    rows_start: int | None
    rows_end: int | None
    descending: Sequence[bool]
    nulls_last: Sequence[bool]
    ignore_nulls: bool


_Children_co = TypeVar(
    "_Children_co",
    covariant=True,
    bound=Sequence[tuple[str, Any]],
    default=Sequence[tuple[str, Any]],
)
DTypeT_co = TypeVar("DTypeT_co", covariant=True, bound="BaseType", default="BaseType")
_Child: TypeAlias = tuple[Literal["child"], DTypeT_co]
_Size: TypeAlias = tuple[Literal["size"], int]
_ID_co = TypeVar("_ID_co", bound=str, default=str, covariant=True)
_Array: TypeAlias = Literal["array"]
_Struct: TypeAlias = Literal["struct"]
_List: TypeAlias = Literal["list"]
_Enum: TypeAlias = Literal["enum"]
_Decimal: TypeAlias = Literal["decimal"]
_TimestampTZ: TypeAlias = Literal["timestamp with time zone"]
IntoColumnExpr: TypeAlias = Union[str, Expression]
"""A column name, or the result of calling `duckdb.ColumnExpression`."""


class BaseType(Protocol[_ID_co]):
    """Structural equivalent to [`DuckDBPyType`].

    Excludes attributes which are unsafe to use on most types.

    [`DuckDBPyType`]: https://github.com/duckdb/duckdb-python/blob/df7789cbd31b2d2b8d03d012f14331bc3297fb2d/_duckdb-stubs/_sqltypes.pyi#L35-L75
    """

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @overload
    def __init__(self, type_str: str, connection: DuckDBPyConnection) -> None: ...
    @overload
    def __init__(self, obj: object) -> None: ...
    @property
    def id(self) -> _ID_co: ...


def has_children(
    dtype: BaseType | _ParentType[_ID_co, _Children_co],
) -> TypeIs[_ParentType[_ID_co, _Children_co]]:
    """Return True if `dtype.children` can be accessed safely.

    `_hasattr_static` returns True on *any* [`DuckDBPyType`], so the only way to be sure is by forcing an exception.

    [`DuckDBPyType`]: https://github.com/duckdb/duckdb-python/blob/df7789cbd31b2d2b8d03d012f14331bc3297fb2d/_duckdb-stubs/_sqltypes.pyi#L35-L75
    """
    try:
        return hasattr(dtype, "children")
    except duckdb.InvalidInputException:
        return False


@overload
def is_dtype(obj: BaseType, type_id: _Array, /) -> TypeIs[ArrayType]: ...
@overload
def is_dtype(obj: BaseType, type_id: _Struct, /) -> TypeIs[StructType]: ...
@overload
def is_dtype(obj: BaseType, type_id: _List, /) -> TypeIs[ListType]: ...
@overload
def is_dtype(obj: BaseType, type_id: _Enum, /) -> TypeIs[EnumType]: ...
@overload
def is_dtype(obj: BaseType, type_id: _Decimal, /) -> TypeIs[DecimalType]: ...
@overload
def is_dtype(
    obj: BaseType, type_id: _TimestampTZ, /
) -> TypeIs[BaseType[_TimestampTZ]]: ...
def is_dtype(
    obj: BaseType, type_id: _Array | _Struct | _List | _Enum | _Decimal | _TimestampTZ, /
) -> bool:
    """Return True if `obj` is the [`DuckDBPyType`] corresponding with `type_id`.

    [`DuckDBPyType`]: https://github.com/duckdb/duckdb-python/blob/df7789cbd31b2d2b8d03d012f14331bc3297fb2d/_duckdb-stubs/_sqltypes.pyi#L35-L75
    """
    return obj.id == type_id


class _ParentType(BaseType[_ID_co], Protocol[_ID_co, _Children_co]):
    @property
    def children(self) -> _Children_co: ...


ArrayType: TypeAlias = _ParentType[_Array, tuple[_Child[DTypeT_co], _Size]]
EnumType: TypeAlias = _ParentType[_Enum, tuple[tuple[Literal["values"], list[str]]]]
DecimalType: TypeAlias = _ParentType[
    _Decimal, tuple[tuple[Literal["precision"], int], tuple[Literal["scale"], int]]
]


class ListType(_ParentType[_List, tuple[_Child[DTypeT_co]]], Protocol[DTypeT_co]):
    @property
    def child(self) -> DTypeT_co: ...


class StructType(_ParentType[_Struct, Sequence[tuple[str, BaseType]]], Protocol):
    def __getattr__(self, name: str) -> BaseType: ...
    def __getitem__(self, name: str) -> BaseType: ...
