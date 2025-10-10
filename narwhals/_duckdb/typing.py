from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, overload

from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection, Expression
    from typing_extensions import TypeAlias, TypeIs


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


class BaseType(Protocol[_ID_co]):
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @overload
    def __init__(self, type_str: str, connection: DuckDBPyConnection) -> None: ...
    @overload
    def __init__(self, obj: object) -> None: ...
    @property
    def id(self) -> _ID_co: ...


class _ParentType(BaseType[_ID_co], Protocol[_ID_co, _Children_co]):
    @property
    def children(self) -> _Children_co: ...


class ArrayType(
    _ParentType[Literal["array"], tuple[_Child[DTypeT_co], _Size]], Protocol[DTypeT_co]
): ...


class EnumType(
    _ParentType[Literal["enum"], tuple[tuple[Literal["values"], list[str]]]], Protocol
): ...


class ListType(
    _ParentType[Literal["list"], tuple[_Child[DTypeT_co]]], Protocol[DTypeT_co]
):
    @property
    def child(self) -> DTypeT_co: ...


class StructType(
    _ParentType[Literal["struct"], Sequence[tuple[str, BaseType]]], Protocol
):
    def __getattr__(self, name: str) -> BaseType: ...
    def __getitem__(self, name: str) -> BaseType: ...


class DecimalType(
    _ParentType[
        Literal["decimal"],
        tuple[tuple[Literal["precision"], int], tuple[Literal["scale"], int]],
    ],
    Protocol,
): ...


def has_children(
    dtype: BaseType | _ParentType[_ID_co, _Children_co],
) -> TypeIs[_ParentType[_ID_co, _Children_co]]:
    """Using `_hasattr_static` returns `True` on any `DuckDBPyType.

    The only way to be sure is forcing an exception.
    """
    import duckdb

    try:
        return hasattr(dtype, "children")
    except duckdb.InvalidInputException:
        return False


def is_dtype_array(obj: BaseType) -> TypeIs[ArrayType]:
    return obj.id == "array"


def is_dtype_struct(obj: BaseType) -> TypeIs[StructType]:
    return obj.id == "struct"


def is_dtype_list(obj: BaseType) -> TypeIs[ListType]:
    return obj.id == "list"


def is_dtype_enum(obj: BaseType) -> TypeIs[EnumType]:
    return obj.id == "enum"


def is_dtype_timestamp_with_time_zone(
    obj: BaseType,
) -> TypeIs[BaseType[Literal["timestamp with time zone"]]]:
    return obj.id == "timestamp with time zone"


def is_dtype_decimal(obj: BaseType) -> TypeIs[DecimalType]:
    return obj.id == "decimal"
