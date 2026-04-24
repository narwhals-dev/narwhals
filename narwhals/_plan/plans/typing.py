from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from narwhals._utils import Implementation


# fmt: off
class _FrameClassVar(Protocol):
    implementation: ClassVar[Implementation]
    def to_native(self) -> Any: ...
class _FrameProperty(Protocol):
    @property
    def implementation(self) -> Implementation: ...
    def to_native(self) -> Any: ...
# fmt: on

FrameT_co = TypeVar("FrameT_co", bound="_FrameClassVar | _FrameProperty", covariant=True)
