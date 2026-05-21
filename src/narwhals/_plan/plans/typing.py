from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from narwhals._utils import Implementation, Version


# fmt: off
class _FrameClassVar(Protocol):
    implementation: ClassVar[Implementation]
    version: ClassVar[Version]
    def to_native(self) -> Any: ...
class _FrameProperty(Protocol):
    @property
    def implementation(self) -> Implementation: ...
    @property
    def version(self) -> Version: ...
    def to_native(self) -> Any: ...
# fmt: on

FrameT_co = TypeVar("FrameT_co", bound="_FrameClassVar | _FrameProperty", covariant=True)
