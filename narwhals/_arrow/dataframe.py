from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.dependencies import get_pyarrow

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.namespace import ArrowNamespace


class ArrowDataFrame:
    # --- not in the spec ---
    def __init__(
        self,
        dataframe: Any,
    ) -> None:
        self._dataframe = dataframe

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._arrow.namespace import ArrowNamespace

        return ArrowNamespace()

    def __native_namespace__(self) -> Any:
        return get_pyarrow()

    def __len__(self) -> int:
        return len(self._dataframe)
