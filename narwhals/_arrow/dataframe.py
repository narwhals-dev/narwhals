from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import overload

from narwhals._arrow.utils import translate_dtype
from narwhals._pandas_like.utils import evaluate_into_exprs
from narwhals.dependencies import get_pyarrow

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import IntoArrowExpr
    from narwhals.dtypes import DType


class ArrowDataFrame:
    # --- not in the spec ---
    def __init__(self, dataframe: Any) -> None:
        self._dataframe = dataframe
        self._implementation = "arrow"  # for compatibility with PandasDataFrame

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._arrow.namespace import ArrowNamespace

        return ArrowNamespace()

    def __native_namespace__(self) -> Any:
        return get_pyarrow()

    def __narwhals_dataframe__(self) -> Self:
        return self

    def _from_dataframe(self, df: Any) -> Self:
        return self.__class__(df)

    @property
    def shape(self) -> tuple[int, int]:
        return self._dataframe.shape  # type: ignore[no-any-return]

    def __len__(self) -> int:
        return len(self._dataframe)

    def rows(
        self, *, named: bool = False
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        if not named:
            msg = "Unnamed rows are not yet supported on PyArrow tables"
            raise NotImplementedError(msg)
        return self._dataframe.to_pylist()  # type: ignore[no-any-return]

    @overload
    def __getitem__(self, item: str) -> ArrowSeries: ...

    @overload
    def __getitem__(self, item: slice) -> ArrowDataFrame: ...

    def __getitem__(self, item: str | slice) -> ArrowSeries | ArrowDataFrame:
        if isinstance(item, str):
            from narwhals._arrow.series import ArrowSeries

            return ArrowSeries(self._dataframe[item], name=item)

        elif isinstance(item, slice):
            from narwhals._arrow.dataframe import ArrowDataFrame

            if item.step is not None and item.step != 1:
                msg = "Slicing with step is not supported on PyArrow tables"
                raise NotImplementedError(msg)
            start = item.start or 0
            stop = item.stop or len(self._dataframe)
            return ArrowDataFrame(
                self._dataframe.slice(item.start, stop - start),
            )

        else:  # pragma: no cover
            msg = f"Expected str or slice, got: {type(item)}"
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

    def select(
        self,
        *exprs: IntoArrowExpr,
        **named_exprs: IntoArrowExpr,
    ) -> Self:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)  # type: ignore[arg-type]
        if not new_series:
            # return empty dataframe, like Polars does
            return self._from_dataframe(self._dataframe.__class__.from_arrays([]))
        names = [s.name for s in new_series]
        pa = get_pyarrow()
        df = pa.Table.from_arrays([s._series for s in new_series], names=names)
        return self._from_dataframe(df)
