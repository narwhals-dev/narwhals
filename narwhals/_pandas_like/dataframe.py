from __future__ import annotations

import collections
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal

from narwhals._pandas_like.utils import evaluate_into_exprs
from narwhals._pandas_like.utils import horizontal_concat
from narwhals._pandas_like.utils import translate_dtype
from narwhals._pandas_like.utils import validate_dataframe_comparand
from narwhals._pandas_like.utils import validate_indices
from narwhals.utils import flatten

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from narwhals._pandas_like.group_by import PandasGroupBy
    from narwhals._pandas_like.namespace import PandasNamespace
    from narwhals._pandas_like.series import PandasSeries
    from narwhals._pandas_like.typing import IntoPandasExpr
    from narwhals.dtypes import DType


class PandasDataFrame:
    # --- not in the spec ---
    def __init__(
        self,
        dataframe: Any,
        *,
        implementation: str,
    ) -> None:
        self._validate_columns(dataframe.columns)
        self._dataframe = dataframe
        self._implementation = implementation

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PandasNamespace:
        from narwhals._pandas_like.namespace import PandasNamespace

        return PandasNamespace(self._implementation)

    def _validate_columns(self, columns: Sequence[str]) -> None:
        if len(columns) != len(set(columns)):
            counter = collections.Counter(columns)
            for col, count in counter.items():
                if count > 1:
                    msg = f"Expected unique column names, got {col!r} {count} time(s)"
                    raise ValueError(
                        msg,
                    )
            raise AssertionError("Pls report bug")

    def _from_dataframe(self, df: Any) -> Self:
        return self.__class__(
            df,
            implementation=self._implementation,
        )

    def __getitem__(self, column_name: str) -> PandasSeries:
        from narwhals._pandas_like.series import PandasSeries

        return PandasSeries(
            self._dataframe.loc[:, column_name],
            implementation=self._implementation,
        )

    # --- properties ---
    @property
    def columns(self) -> list[str]:
        return self._dataframe.columns.tolist()  # type: ignore[no-any-return]

    @property
    def schema(self) -> dict[str, DType]:
        return {
            col: translate_dtype(dtype) for col, dtype in self._dataframe.dtypes.items()
        }

    # --- reshape ---
    def select(
        self,
        *exprs: IntoPandasExpr | Iterable[IntoPandasExpr],
        **named_exprs: IntoPandasExpr,
    ) -> Self:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        new_series = validate_indices(new_series)
        df = horizontal_concat(
            new_series,
            implementation=self._implementation,
        )
        return self._from_dataframe(df)

    def filter(
        self,
        *predicates: IntoPandasExpr | Iterable[IntoPandasExpr],
    ) -> Self:
        from narwhals._pandas_like.namespace import PandasNamespace

        plx = PandasNamespace(self._implementation)
        expr = plx.all_horizontal(*predicates)
        # Safety: all_horizontal's expression only returns a single column.
        mask = expr._call(self)[0]
        _mask = validate_dataframe_comparand(self._dataframe.index, mask)
        return self._from_dataframe(self._dataframe.loc[_mask])

    def with_columns(
        self,
        *exprs: IntoPandasExpr | Iterable[IntoPandasExpr],
        **named_exprs: IntoPandasExpr,
    ) -> Self:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        df = self._dataframe.assign(
            **{
                series.name: validate_dataframe_comparand(self._dataframe.index, series)
                for series in new_series
            }
        )
        return self._from_dataframe(df)

    def rename(self, mapping: dict[str, str]) -> Self:
        return self._from_dataframe(self._dataframe.rename(columns=mapping))

    def drop(self, columns: str | Iterable[str]) -> Self:
        return self._from_dataframe(self._dataframe.drop(columns=flatten(columns)))

    # --- transform ---
    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        flat_keys = flatten([*flatten([by]), *more_by])
        df = self._dataframe
        if isinstance(descending, bool):
            ascending: bool | list[bool] = not descending
        else:
            ascending = [not d for d in descending]
        return self._from_dataframe(df.sort_values(flat_keys, ascending=ascending))

    # --- convert ---
    def collect(self) -> PandasDataFrame:
        return PandasDataFrame(
            self._dataframe,
            implementation=self._implementation,
        )

    # --- actions ---
    def group_by(self, *keys: str | Iterable[str]) -> PandasGroupBy:
        from narwhals._pandas_like.group_by import PandasGroupBy

        return PandasGroupBy(
            self,
            flatten(keys),
        )

    def join(
        self,
        other: Self,
        *,
        how: Literal["left", "inner", "outer"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> Self:
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        return self._from_dataframe(
            self._dataframe.merge(
                other._dataframe,
                left_on=left_on,
                right_on=right_on,
                how=how,
                suffixes=("", "_right"),
            ),
        )

    # --- partial reduction ---

    def head(self, n: int) -> Self:
        return self._from_dataframe(self._dataframe.head(n))

    def unique(self, subset: str | list[str]) -> Self:
        subset = flatten(subset)
        return self._from_dataframe(self._dataframe.drop_duplicates(subset=subset))

    # --- lazy-only ---
    def lazy(self) -> Self:
        return self.__class__(
            self._dataframe,
            implementation=self._implementation,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self._dataframe.shape  # type: ignore[no-any-return]

    def to_dict(self, *, as_series: bool = False) -> dict[str, Any]:
        if as_series:
            # todo: should this return narwhals series?
            return {col: self._dataframe.loc[:, col] for col in self.columns}
        return self._dataframe.to_dict(orient="list")  # type: ignore[no-any-return]

    def to_numpy(self) -> Any:
        from narwhals._pandas_like.series import PANDAS_TO_NUMPY_DTYPE_MISSING

        # pandas return `object` dtype for nullable dtypes, so we cast each
        # Series to numpy and let numpy find a common dtype.
        # If there aren't any dtypes where `to_numpy()` is "broken" (i.e. it
        # returns Object) then we just call `to_numpy()` on the DataFrame.
        for dtype in self._dataframe.dtypes:
            if str(dtype) in PANDAS_TO_NUMPY_DTYPE_MISSING:
                import numpy as np

                return np.hstack([self[col].to_numpy()[:, None] for col in self.columns])
        return self._dataframe.to_numpy()

    def to_pandas(self) -> Any:
        if self._implementation == "pandas":
            return self._dataframe
        if self._implementation == "modin":  # pragma: no cover
            return self._dataframe._to_pandas()
        return self._dataframe.to_pandas()  # pragma: no cover
