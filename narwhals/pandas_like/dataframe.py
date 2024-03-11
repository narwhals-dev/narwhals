from __future__ import annotations

import collections
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal

from narwhals.pandas_like.utils import evaluate_into_exprs
from narwhals.pandas_like.utils import horizontal_concat
from narwhals.pandas_like.utils import reset_index
from narwhals.pandas_like.utils import translate_dtype
from narwhals.pandas_like.utils import validate_dataframe_comparand
from narwhals.utils import flatten_str

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.pandas_like.group_by import PandasGroupBy
    from narwhals.pandas_like.series import PandasSeries
    from narwhals.pandas_like.typing import IntoExpr


class PandasDataFrame:
    # --- not in the spec ---
    def __init__(
        self,
        dataframe: Any,
        *,
        implementation: str,
        is_eager: bool,
        is_lazy: bool,
    ) -> None:
        self._validate_columns(dataframe.columns)
        self._dataframe = reset_index(dataframe)
        self._implementation = implementation
        self._is_eager = is_eager
        self._is_lazy = is_lazy

    def _validate_columns(self, columns: Sequence[str]) -> None:
        counter = collections.Counter(columns)
        for col, count in counter.items():
            if count > 1:
                msg = f"Expected unique column names, got {col!r} {count} time(s)"
                raise ValueError(
                    msg,
                )

    def _validate_booleanness(self) -> None:
        if not (
            (self._dataframe.dtypes == "bool") | (self._dataframe.dtypes == "boolean")
        ).all():
            msg = "'any' can only be called on DataFrame where all dtypes are 'bool'"
            raise TypeError(
                msg,
            )

    def _from_dataframe(self, df: Any) -> Self:
        return self.__class__(
            df,
            implementation=self._implementation,
            is_eager=self._is_eager,
            is_lazy=self._is_lazy,
        )

    def __getitem__(self, column_name: str) -> PandasSeries:
        from narwhals.pandas_like.series import PandasSeries

        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.__getitem__ can only be called when it was instantiated with `is_eager=True`"
            )
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
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        df = horizontal_concat(
            [series._series for series in new_series],
            implementation=self._implementation,
        )
        return self._from_dataframe(df)

    def filter(
        self,
        *predicates: IntoExpr | Iterable[IntoExpr],
    ) -> Self:
        from narwhals.pandas_like.namespace import Namespace

        plx = Namespace(self._implementation)
        expr = plx.all_horizontal(*predicates)
        # Safety: all_horizontal's expression only returns a single column.
        mask = expr._call(self)[0]
        _mask = validate_dataframe_comparand(mask)
        return self._from_dataframe(self._dataframe.loc[_mask])

    def with_columns(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        df = self._dataframe.assign(
            **{series.name: validate_dataframe_comparand(series) for series in new_series}
        )
        return self._from_dataframe(df)

    def rename(self, mapping: dict[str, str]) -> Self:
        return self._from_dataframe(self._dataframe.rename(columns=mapping))

    # --- transform ---
    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        flat_keys = flatten_str([*flatten_str(by), *more_by])
        if not flat_keys:
            flat_keys = self._dataframe.columns.tolist()
        df = self._dataframe
        if isinstance(descending, bool):
            ascending: bool | list[bool] = not descending
        else:
            ascending = [not d for d in descending]
        return self._from_dataframe(
            df.sort_values(flat_keys, ascending=ascending),
        )

    # --- convert ---
    def collect(self) -> PandasDataFrame:
        if not self._is_lazy:
            raise RuntimeError(
                "DataFrame.collect can only be called when it was instantiated with `is_lazy=True`"
            )
        return PandasDataFrame(
            self._dataframe,
            implementation=self._implementation,
            is_eager=True,
            is_lazy=False,
        )

    # --- actions ---
    def group_by(self, *keys: str | Iterable[str]) -> PandasGroupBy:
        from narwhals.pandas_like.group_by import PandasGroupBy

        return PandasGroupBy(
            self,
            flatten_str(*keys),
            is_eager=self._is_eager,
            is_lazy=self._is_lazy,
        )

    def join(
        self,
        other: Self,
        *,
        how: Literal["left", "inner", "outer"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> Self:
        if how not in ["inner"]:
            msg = "Only inner join supported for now, others coming soon"
            raise ValueError(msg)

        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        if overlap := (set(self.columns) - set(left_on)).intersection(
            set(other.columns) - set(right_on),
        ):
            msg = f"Found overlapping columns in join: {overlap}. Please rename columns to avoid this."
            raise ValueError(msg)

        return self._from_dataframe(
            self._dataframe.merge(
                other._dataframe,
                left_on=left_on,
                right_on=right_on,
                how=how,
            ),
        )

    # --- partial reduction ---

    def head(self, n: int) -> Self:
        return self._from_dataframe(self._dataframe.head(n))

    def unique(self, subset: list[str]) -> Self:
        return self._from_dataframe(self._dataframe.drop_duplicates(subset=subset))

    # --- lazy-only ---
    def cache(self) -> Self:
        return self

    def lazy(self) -> Self:
        return self.__class__(
            self._dataframe,
            is_eager=False,
            is_lazy=True,
            implementation=self._implementation,
        )

    @property
    def shape(self) -> tuple[int, int]:
        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.shape can only be called when it was instantiated with `is_eager=True`"
            )
        return self._dataframe.shape  # type: ignore[no-any-return]

    def iter_columns(self) -> Iterable[PandasSeries]:
        from narwhals.pandas_like.series import PandasSeries

        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.iter_columns can only be called when it was instantiated with `is_eager=True`"
            )
        return (
            PandasSeries(self._dataframe[col], implementation=self._implementation)
            for col in self.columns
        )

    def to_dict(self, *, as_series: bool = False) -> dict[str, Any]:
        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.to_dict can only be called when it was instantiated with `is_eager=True`"
            )
        if as_series:
            # todo: should this return narwhals series?
            return {col: self._dataframe.loc[:, col] for col in self.columns}
        return self._dataframe.to_dict(orient="list")  # type: ignore[no-any-return]

    def to_numpy(self) -> Any:
        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.to_numpy can only be called when it was instantiated with `is_eager=True`"
            )
        return self._dataframe.to_numpy()

    def to_pandas(self) -> Any:
        if not self._is_eager:
            raise RuntimeError(
                "DataFrame.to_pandas can only be called when it was instantiated with `is_eager=True`"
            )
        return self._dataframe

    # --- public, non-Polars ---
    @property
    def is_eager(self) -> bool:
        return self._is_eager

    @property
    def is_lazy(self) -> bool:
        return self._is_lazy
