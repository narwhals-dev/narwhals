from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals.typing import DataFrameLike, NativeFrame, NativeLazyFrame

Data: TypeAlias = "dict[str, Any]"

Constructor: TypeAlias = Callable[[Any], "NativeLazyFrame | NativeFrame | DataFrameLike"]
ConstructorEager: TypeAlias = Callable[[Any], "NativeFrame | DataFrameLike"]
ConstructorLazy: TypeAlias = Callable[[Any], "NativeLazyFrame"]
