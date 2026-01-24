from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol

import narwhals as nw

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


DataLoader = Callable[[str], tuple[nw.LazyFrame[Any], ...]]


class QueryModule(Protocol):
    def query(
        self, *args: nw.LazyFrame[Any], **kwds: nw.LazyFrame[Any]
    ) -> nw.LazyFrame[Any]: ...
