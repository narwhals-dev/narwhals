from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from narwhals._arrow.utils import translate_dtype
from narwhals._pandas_like.dataframe import PandasDataFrame
from narwhals._pandas_like.utils import evaluate_into_exprs
from narwhals.dependencies import get_pyarrow

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals.dtypes import DType


class ArrowDataFrame(PandasDataFrame):
    # --- not in the spec ---
    def __init__(
        self,
        dataframe: Any,
        *,
        implementation: str,
    ) -> None:
        super().__init__(dataframe, implementation=implementation)

    def _validate_columns(self, columns: Sequence[str]) -> None:
        return None

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._arrow.namespace import ArrowNamespace

        return ArrowNamespace()

    def __native_namespace__(self) -> Any:
        return get_pyarrow()

    @property
    def schema(self) -> dict[str, DType]:
        schema = self._dataframe.schema
        return {
            name: translate_dtype(dtype)
            for name, dtype in zip(schema.names, schema.types)
        }

    @property
    def columns(self) -> list[str]:
        return self._dataframe.schema.names

    def with_columns(
        self,
        *exprs: IntoPandasExpr,
        **named_exprs: IntoPandasExpr,
    ) -> Self:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        new_names = {s.name: s for s in new_series}
        to_concat = []
        # Make sure to preserve column order
        for s in self.columns:
            if s in new_names:
                to_concat.append(validate_dataframe_comparand(index, new_names.pop(s)))
            else:
                to_concat.append(self._dataframe[s])
        to_concat.extend(
            validate_dataframe_comparand(index, new_names[s]) for s in new_names
        )

        df = horizontal_concat(
            to_concat,
            implementation=self._implementation,
        )
        return self._from_dataframe(df)
