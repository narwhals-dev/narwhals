from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence
from typing import overload

from narwhals._arrow.utils import translate_dtype
from narwhals._pandas_like.utils import evaluate_into_exprs
from narwhals.dependencies import get_numpy
from narwhals.dependencies import get_pyarrow
from narwhals.utils import flatten

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

    def __narwhals_lazyframe__(self) -> Self:
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
            if item.step is not None and item.step != 1:
                msg = "Slicing with step is not supported on PyArrow tables"
                raise NotImplementedError(msg)
            start = item.start or 0
            stop = item.stop or len(self._dataframe)
            return self._from_dataframe(
                self._dataframe.slice(item.start, stop - start),
            )

        elif isinstance(item, Sequence) or (
            (np := get_numpy()) is not None
            and isinstance(item, np.ndarray)
            and item.ndim == 1
        ):
            return self._from_dataframe(self._dataframe.take(item))

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

    def drop(self, *columns: str | Iterable[str]) -> Self:
        return self._from_dataframe(self._dataframe.drop(list(flatten(columns))))

    def drop_nulls(self) -> Self:
        return self._from_dataframe(self._dataframe.drop_null())

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        flat_keys = flatten([*flatten([by]), *more_by])
        df = self._dataframe

        if isinstance(descending, bool):
            order = "descending" if descending else "ascending"
            sorting = [(key, order) for key in flat_keys]
        else:
            sorting = [
                (key, "descending" if is_descending else "ascending")
                for key, is_descending in zip(flat_keys, descending)
            ]
        return self._from_dataframe(df.sort_by(sorting=sorting))

    def to_pandas(self) -> Any:
        return self._dataframe.to_pandas()

    def lazy(self) -> Self:
        return self

    def collect(self) -> ArrowDataFrame:
        return ArrowDataFrame(self._dataframe)

    def clone(self) -> Self:
        raise NotImplementedError("clone is not yet supported on PyArrow tables")
