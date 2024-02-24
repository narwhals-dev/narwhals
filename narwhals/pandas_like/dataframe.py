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
from narwhals.spec import DataFrame as DataFrameProtocol
from narwhals.spec import LazyFrame as LazyFrameProtocol
from narwhals.utils import flatten_str

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from narwhals.pandas_like.group_by import GroupBy
    from narwhals.pandas_like.group_by import LazyGroupBy
    from narwhals.spec import DType
    from narwhals.spec import IntoExpr


class DataFrame(DataFrameProtocol):
    # --- not in the spec ---
    def __init__(
        self,
        dataframe: Any,
        *,
        implementation: str,
    ) -> None:
        self._validate_columns(dataframe.columns)
        self._dataframe = reset_index(dataframe)
        self._implementation = implementation

    def _dispatch_to_lazy(self, method: str, *args: Any, **kwargs: Any) -> Self:
        return getattr(self.lazy(), method)(*args, **kwargs).collect()  # type: ignore[no-any-return]

    def __repr__(self) -> str:  # pragma: no cover
        header = " Narwhals DataFrame                      "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Use `.to_native()` to see native output |\n"
            + "└"
            + "─" * length
            + "┘\n"
        )

    def _validate_columns(self, columns: Sequence[str]) -> None:
        counter = collections.Counter(columns)
        for col, count in counter.items():
            if count > 1:
                msg = f"Expected unique column names, got {col} {count} time(s)"
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

    # --- properties ---
    @property
    def columns(self) -> list[str]:
        return self._dataframe.columns.tolist()  # type: ignore[no-any-return]

    @property
    def schema(self) -> dict[str, DType]:
        return self.lazy().schema

    @property
    def shape(self) -> tuple[int, int]:
        return self._dataframe.shape  # type: ignore[no-any-return]

    # --- reshape ---
    def with_columns(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        return self._dispatch_to_lazy("with_columns", *exprs, **named_exprs)

    def select(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        return self._dispatch_to_lazy("select", *exprs, **named_exprs)

    def filter(
        self,
        *predicates: IntoExpr | Iterable[IntoExpr],
    ) -> Self:
        return self._dispatch_to_lazy("filter", *predicates)

    def rename(self, mapping: dict[str, str]) -> Self:
        return self._dispatch_to_lazy("rename", mapping)

    # --- transform ---
    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        return self._dispatch_to_lazy("sort", by, *more_by, descending=descending)

    # --- convert ---
    def lazy(self) -> LazyFrame:
        return LazyFrame(
            self._dataframe,
            implementation=self._implementation,
        )

    def to_numpy(self) -> Any:
        return self._dataframe.to_numpy()

    def to_pandas(self) -> Any:
        if self._implementation == "pandas":
            return self._dataframe
        elif self._implementation == "cudf":
            return self._dataframe.to_pandas()
        elif self._implementation == "modin":
            return self._dataframe._to_pandas()
        msg = f"Unknown implementation: {self._implementation}"
        raise TypeError(msg)

    def to_dict(self, *, as_series: bool = True) -> dict[str, Any]:
        if as_series:
            return {col: self._dataframe[col] for col in self._dataframe.columns}
        return self._dataframe.to_dict(orient="list")  # type: ignore[no-any-return]

    # --- actions ---
    def group_by(self, *keys: str | Iterable[str]) -> GroupBy:
        from narwhals.pandas_like.group_by import GroupBy

        return GroupBy(self, flatten_str(*keys))

    def join(
        self,
        other: Self,
        *,
        how: Literal["left", "inner", "outer"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> Self:
        return self._dispatch_to_lazy(
            "join", other.lazy(), how=how, left_on=left_on, right_on=right_on
        )

    # --- partial reduction ---
    def head(self, n: int) -> Self:
        return self._dispatch_to_lazy("head", n)

    def unique(self, subset: list[str]) -> Self:
        return self._dispatch_to_lazy("unique", subset)

    # --- public, non-Polars ---
    def to_native(self) -> Any:
        return self._dataframe


class LazyFrame(LazyFrameProtocol):
    # --- not in the spec ---
    def __init__(
        self,
        dataframe: Any,
        *,
        implementation: str,
    ) -> None:
        self._validate_columns(dataframe.columns)
        self._dataframe = reset_index(dataframe)
        self._implementation = implementation

    def __repr__(self) -> str:  # pragma: no cover
        header = " Narwhals LazyFrame                      "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Add `._dataframe` to see native output |\n"
            + "└"
            + "─" * length
            + "┘\n"
        )

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
            [series.series for series in new_series],
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
            **{series.name: series.series for series in new_series}
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
    def collect(self) -> DataFrame:
        return DataFrame(
            self._dataframe,
            implementation=self._implementation,
        )

    # --- actions ---
    def group_by(self, *keys: str | Iterable[str]) -> LazyGroupBy:
        from narwhals.pandas_like.group_by import LazyGroupBy

        return LazyGroupBy(self, flatten_str(*keys))

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

    # --- public, non-Polars ---
    def to_native(self) -> Any:
        return self._dataframe
