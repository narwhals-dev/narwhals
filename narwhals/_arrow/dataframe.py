from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence
from typing import overload

from narwhals._arrow.utils import translate_dtype
from narwhals._pandas_like.dataframe import PandasDataFrame
from narwhals._pandas_like.utils import evaluate_into_exprs
from narwhals.dependencies import get_pyarrow

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import IntoArrowExpr
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

    @overload
    def __getitem__(self, item: str) -> ArrowSeries: ...

    @overload
    def __getitem__(self, item: range | slice) -> ArrowDataFrame: ...

    def __getitem__(self, item: str | range | slice) -> ArrowSeries | ArrowDataFrame:
        if isinstance(item, str):
            from narwhals._arrow.series import ArrowSeries

            name = self._dataframe.schema.names.index(item)
            return ArrowSeries(
                self._dataframe[name],
                implementation=self._implementation,
                name=name,
            )

        elif isinstance(item, (range, slice)):
            from narwhals._arrow.dataframe import ArrowDataFrame

            return ArrowDataFrame(
                self._dataframe.iloc[item], implementation=self._implementation
            )

        else:  # pragma: no cover
            msg = f"Expected str, range or slice, got: {type(item)}"
            raise TypeError(msg)

    @property
    def schema(self) -> dict[str, DType]:
        schema = self._dataframe.schema
        return {
            name: translate_dtype(dtype)
            for name, dtype in zip(schema.names, schema.types)
        }

    @property
    def columns(self) -> list[str]:
        return self._dataframe.schema.names  # type: ignore[no-any-return]

    def with_columns(
        self,
        *exprs: IntoArrowExpr,  # type: ignore[override]
        **named_exprs: IntoArrowExpr,  # type: ignore[override]
    ) -> Self:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        new_names = {s.name: s for s in new_series}
        result_names = []
        to_concat = []
        # Make sure to preserve column order
        for s in self.columns:
            if s in new_names:
                to_concat.append(new_names.pop(s)._series)
                # to_concat.append(validate_dataframe_comparand(index, new_names.pop(s)))
            else:
                to_concat.append(self._dataframe[s])
            result_names.append(s)
        for s in new_names:
            to_concat.append(new_names[s]._series)
            result_names.append(s)
        pa = get_pyarrow()
        df = pa.Table.from_arrays(to_concat, names=result_names)
        return self._from_dataframe(df)

    def select(
        self,
        *exprs: IntoArrowExpr,  # type: ignore[override]
        **named_exprs: IntoArrowExpr,  # type: ignore[override]
    ) -> Self:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        if not new_series:
            # return empty dataframe, like Polars does
            return self._from_dataframe(self._dataframe.__class__())
        names = [s.name for s in new_series]
        pa = get_pyarrow()
        df = pa.Table.from_arrays([s._series for s in new_series], names=names)
        return self._from_dataframe(df)
