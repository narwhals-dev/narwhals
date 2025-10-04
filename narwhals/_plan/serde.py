"""(De)serialization support for `ExprIR` & `Immutable`.

Planned for use by:
- `Expr.meta.serialize`
- `Expr.deserialize`
"""

from __future__ import annotations

import re
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, Union, overload

from narwhals._plan import expressions as ir
from narwhals._plan._immutable import Immutable
from narwhals._plan.options import FunctionOptions
from narwhals._plan.series import Series
from narwhals._utils import Version, isinstance_or_issubclass, qualified_type_name
from narwhals.dtypes import DType, _is_into_dtype, _validate_into_dtype

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import TypeAlias

    from narwhals._plan.expr import Expr
    from narwhals.typing import FileSource

__all__ = ["deserialize", "from_dict", "serialize", "to_dict"]

dtypes = Version.MAIN.dtypes

JSONLiteral: TypeAlias = (
    "Union[str, int, float, list[JSONLiteral], Mapping[str, JSONLiteral], None]"
)
SerializationFormat: TypeAlias = Literal["binary", "json"]

# e.g. `tuple[tuple[...]]` is not allowed
SUPPORTED_NESTED = (str, int, float, type(None), Immutable, DType, type)
# NOTE: Intentionally omits `list`, `dict`
SUPPORTED = (*SUPPORTED_NESTED, Series, tuple)

SELECTORS_ONLY = frozenset, re.Pattern

# udf + `.name` functions
UNLIKELY = (callable,)

# Not sure how to handle
PLANNED = SELECTORS_ONLY

UNSUPPORTED_DTYPES = (
    dtypes.Datetime,
    dtypes.Duration,
    dtypes.Enum,
    dtypes.Struct,
    dtypes.List,
    dtypes.Array,
)


def _not_yet_impl_error(
    obj: Immutable, field: str, value: Any, origin: Literal["to_dict"]
) -> NotImplementedError:
    msg = (
        f"Found an expected type in {field!r}: {qualified_type_name(value)!r}\n"
        f"but serde (`{origin}`) support has not yet been implemented.\n\n{obj!s}"
    )
    return NotImplementedError(msg)


def _unrecognized_type_error(obj: Immutable, field: str, value: Any) -> TypeError:
    msg = f"Found an unrecognized type in {field!r}: {qualified_type_name(value)!r}\n\n{obj!s}"
    return TypeError(msg)


def _to_list_iter(
    field: str, obj: tuple[object, ...], /, *, qualify_type_name: bool, owner: Immutable
) -> Iterator[JSONLiteral]:
    for element in obj:
        if isinstance(element, SUPPORTED_NESTED):
            yield _to_values(
                field, element, qualify_type_name=qualify_type_name, owner=owner
            )
        elif isinstance(element, PLANNED):
            raise _not_yet_impl_error(owner, field, element, "to_dict")
        else:
            raise _unrecognized_type_error(owner, field, element)


def _to_values(
    field: str, v: object, /, *, qualify_type_name: bool, owner: Immutable
) -> JSONLiteral:
    # Handled by json, not nested
    if isinstance(v, (str, int, float, type(None))):
        return v
    # Literal / is_in
    if isinstance(v, Series):
        return v.to_list()
    # Primary cases
    if isinstance(v, Immutable):
        if isinstance(v, FunctionOptions):
            return v.to_dict(qualify_type_name=qualify_type_name)
        return _to_dict(v, qualify_type_name=qualify_type_name)
    # Primary nesting case
    if isinstance(v, (tuple,)):
        return list(
            _to_list_iter(field, v, qualify_type_name=qualify_type_name, owner=owner)
        )
    if isinstance_or_issubclass(v, DType):
        return _dtype_to_path_str(v)
    raise _not_yet_impl_error(owner, field, v, "to_dict")


def _to_dict_children_iter(
    obj: Immutable, /, *, qualify_type_name: bool
) -> Iterator[tuple[str, JSONLiteral]]:
    for k, v in obj.__immutable_items__:
        # Handled by json, not nested
        if isinstance(v, SUPPORTED):
            yield k, _to_values(k, v, qualify_type_name=qualify_type_name, owner=obj)
        elif isinstance(v, PLANNED) or callable(v):
            raise _not_yet_impl_error(obj, k, v, "to_dict")
        else:  # FunctionFlags Enum
            raise _unrecognized_type_error(obj, k, v)


def _dtype_to_path_str(obj: DType | type[DType]) -> str:
    if isinstance_or_issubclass(obj, UNSUPPORTED_DTYPES):
        msg = f"DType serialization is not yet supported for {qualified_type_name(obj)!r}"
        raise NotImplementedError(msg)
    return qualified_type_name(obj)


def _dtype_from_path_str(path_str: str) -> DType:
    parts = path_str.rsplit(".", maxsplit=1)
    imported_module = import_module(parts[0])
    tp = getattr(imported_module, parts[1])
    if not _is_into_dtype(tp):
        _validate_into_dtype(tp)
        return dtypes.Unknown()
    return tp.base_type()()


def _to_dict(
    obj: ir.ExprIR | Immutable, /, *, qualify_type_name: bool = False
) -> dict[str, dict[str, JSONLiteral]]:
    leaf_name = qualified_type_name(obj) if qualify_type_name else type(obj).__name__
    return {
        leaf_name: dict(_to_dict_children_iter(obj, qualify_type_name=qualify_type_name))
    }


def to_dict(expr: Expr | ir.ExprIR, /) -> dict[str, dict[str, JSONLiteral]]:
    return _to_dict(expr if isinstance(expr, ir.ExprIR) else expr._ir)


def from_dict(mapping: Mapping[str, Mapping[str, JSONLiteral]], /) -> Any:
    msg = "`serde.from_dict` is not yet implemented"
    raise NotImplementedError(msg)


@overload
def serialize(
    expr: Expr | ir.ExprIR, /, file: None = ..., *, format: Literal["binary"] = ...
) -> bytes: ...


@overload
def serialize(
    expr: Expr | ir.ExprIR, /, file: None = ..., *, format: Literal["json"]
) -> str: ...


@overload
def serialize(
    expr: Expr | ir.ExprIR, /, file: FileSource, *, format: SerializationFormat = ...
) -> None: ...


def serialize(
    expr: Expr | ir.ExprIR,
    /,
    file: FileSource | None = None,
    *,
    format: SerializationFormat = "binary",
) -> bytes | str | None:
    msg = "`serde.serialize` is not yet implemented"
    raise NotImplementedError(msg)


def deserialize(source: FileSource, *, format: SerializationFormat = "binary") -> Expr:
    msg = "`serde.deserialize` is not yet implemented"
    raise NotImplementedError(msg)
