from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Protocol

_MIN_TYPING_EXTENSIONS = 4, 4, 0
_TYPING_EXTENSIONS = "typing_extensions"


def _typing_extensions_has_pep_696() -> bool:  # pragma: no cover
    from importlib.metadata import version
    from importlib.util import find_spec

    from narwhals.utils import parse_version

    if find_spec(_TYPING_EXTENSIONS):
        return parse_version(version(_TYPING_EXTENSIONS)) >= _MIN_TYPING_EXTENSIONS
    return False


if TYPE_CHECKING:
    import pyarrow as pa
    from typing_extensions import Self
    from typing_extensions import TypeAlias
    from typing_extensions import TypeIs
    from typing_extensions import TypeVar


else:  # pragma: no cover
    import sys

    if sys.version_info >= (3, 13):
        from typing import TypeVar
    elif _typing_extensions_has_pep_696():
        from typing_extensions import TypeVar
    else:
        from typing import TypeVar as _TypeVar

        def TypeVar(  # noqa: ANN202, N802
            name: str,
            *constraints: Any,
            bound: Any | None = None,
            covariant: bool = False,
            contravariant: bool = False,
            **kwds: Any,  # noqa: ARG001
        ):
            return _TypeVar(
                name,
                *constraints,
                bound=bound,
                covariant=covariant,
                contravariant=contravariant,
            )


class ArrowStreamExportable(Protocol):
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...


ToNumpyT_co = TypeVar("ToNumpyT_co", covariant=True)
FromNumpyDT_contra = TypeVar(
    "FromNumpyDT_contra", contravariant=True, default=ToNumpyT_co
)
FromNumpyT_contra = TypeVar("FromNumpyT_contra", contravariant=True)


class ToNumpy(Protocol[ToNumpyT_co]):
    def to_numpy(self, *args: Any, **kwds: Any) -> ToNumpyT_co: ...


class FromNumpy(Protocol[FromNumpyT_contra]):
    @classmethod
    def from_numpy(cls, data: FromNumpyT_contra, *args: Any, **kwds: Any) -> Self: ...


class NumpyConvertible(
    ToNumpy[ToNumpyT_co],
    FromNumpy[FromNumpyDT_contra],
    Protocol[ToNumpyT_co, FromNumpyDT_contra],
):
    def to_numpy(self, dtype: Any, *, copy: bool | None) -> ToNumpyT_co: ...


FromIterableT_contra = TypeVar("FromIterableT_contra", contravariant=True, default=Any)


class FromIterable(Protocol[FromIterableT_contra]):
    @classmethod
    def from_iterable(
        cls, data: Iterable[FromIterableT_contra], *args: Any, **kwds: Any
    ) -> Self: ...


ToDictDT_co = TypeVar(
    "ToDictDT_co", bound=Mapping[str, Any], covariant=True, default="dict[str, Any]"
)
FromDictDT_contra = TypeVar(
    "FromDictDT_contra",
    bound=Mapping[str, Any],
    contravariant=True,
    default=Mapping[str, Any],
)


class ToDict(Protocol[ToDictDT_co]):
    def to_dict(self, *args: Any, **kwds: Any) -> ToDictDT_co: ...


class FromDict(Protocol[FromDictDT_contra]):
    @classmethod
    def from_dict(cls, data: FromDictDT_contra, *args: Any, **kwds: Any) -> Self: ...


class DictConvertible(
    ToDict[ToDictDT_co],
    FromDict[FromDictDT_contra],
    Protocol[ToDictDT_co, FromDictDT_contra],
): ...


IntoArrowTable: TypeAlias = "ArrowStreamExportable | pa.Table"
"""An object supporting the [Arrow PyCapsule Interface], or a native [`pyarrow.Table`].

[Arrow PyCapsule Interface]: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html#arrowstream-export
[`pyarrow.Table`]: https://arrow.apache.org/docs/python/generated/pyarrow.Table.html
"""
ToArrowT_co = TypeVar("ToArrowT_co", covariant=True)
FromArrowDT_contra = TypeVar(
    "FromArrowDT_contra", contravariant=True, default=IntoArrowTable
)


class ToArrow(Protocol[ToArrowT_co]):
    def to_arrow(self, *args: Any, **kwds: Any) -> ToArrowT_co: ...


class FromArrow(Protocol[FromArrowDT_contra]):
    @classmethod
    def from_arrow(cls, data: FromArrowDT_contra, *args: Any, **kwds: Any) -> Self: ...


class ArrowConvertible(
    ToArrow[ToArrowT_co],
    FromArrow[FromArrowDT_contra],
    Protocol[ToArrowT_co, FromArrowDT_contra],
): ...


FromNativeT = TypeVar("FromNativeT")


class FromNative(Protocol[FromNativeT]):
    @classmethod
    def from_native(cls, data: FromNativeT, *args: Any, **kwds: Any) -> Self: ...
    @staticmethod
    def _is_native(obj: FromNativeT | Any, /) -> TypeIs[FromNativeT]: ...
