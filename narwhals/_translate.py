from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Protocol

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing_extensions import TypeVar


else:  # pragma: no cover
    import sys
    from importlib.util import find_spec

    if sys.version_info >= (3, 13):
        from typing import TypeVar
    elif find_spec("typing_extensions"):
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
