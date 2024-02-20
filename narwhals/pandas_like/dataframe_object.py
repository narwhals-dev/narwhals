from __future__ import annotations

import collections
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal

import narwhals
from narwhals.spec import DataFrame as DataFrameT
from narwhals.spec import GroupBy as GroupByT
from narwhals.spec import IntoExpr
from narwhals.spec import LazyFrame as LazyFrameT
from narwhals.spec import LazyGroupBy as LazyGroupByT
from narwhals.spec import Namespace as NamespaceT
from narwhals.utils import evaluate_into_exprs
from narwhals.utils import flatten_str
from narwhals.utils import horizontal_concat
from narwhals.utils import validate_dataframe_comparand

if TYPE_CHECKING:
    from collections.abc import Sequence


class DataFrame(DataFrameT):
    """dataframe object"""

    def __init__(
        self,
        dataframe: Any,
        *,
        api_version: str,
        implementation: str,
    ) -> None:
        self._validate_columns(dataframe.columns)
        self._dataframe = dataframe.reset_index(drop=True)
        self.api_version = api_version
        self._implementation = implementation

    @property
    def columns(self) -> list[str]:
        return self.dataframe.columns.tolist()

    def __repr__(self) -> str:  # pragma: no cover
        header = f" Standard DataFrame (api_version={self.api_version}) "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Add `.dataframe` to see native output         |\n"
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
            (self.dataframe.dtypes == "bool") | (self.dataframe.dtypes == "boolean")
        ).all():
            msg = "'any' can only be called on DataFrame where all dtypes are 'bool'"
            raise TypeError(
                msg,
            )

    @property
    def dataframe(self) -> Any:
        return self._dataframe

    def __dataframe_namespace__(
        self,
    ) -> NamespaceT:
        return narwhals.pandas_like.Namespace(
            api_version=self.api_version,
            implementation=self._implementation,  # type: ignore[attr-defined]
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.dataframe.shape  # type: ignore[no-any-return]

    def group_by(self, *keys: str | Iterable[str]) -> GroupByT:
        from narwhals.pandas_like.group_by_object import GroupBy

        return GroupBy(self, flatten_str(*keys), api_version=self.api_version)

    def select(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> DataFrameT:
        return self.lazy().select(*exprs, **named_exprs).collect()

    def filter(
        self,
        *predicates: IntoExpr | Iterable[IntoExpr],
    ) -> DataFrameT:
        return self.lazy().filter(*predicates).collect()

    def with_columns(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> DataFrameT:
        return self.lazy().with_columns(*exprs, **named_exprs).collect()

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Iterable[bool] = False,
    ) -> DataFrameT:
        return self.lazy().sort(by, *more_by, descending=descending).collect()

    def join(
        self,
        other: DataFrameT,
        *,
        how: Literal["left", "inner", "outer"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> DataFrameT:
        return (
            self.lazy()
            .join(other.lazy(), how=how, left_on=left_on, right_on=right_on)
            .collect()
        )

    def lazy(self) -> LazyFrameT:
        return LazyFrame(
            self.dataframe,
            api_version=self.api_version,
            implementation=self._implementation,
        )

    def head(self, n: int) -> DataFrameT:
        return self.lazy().head(n).collect()

    def unique(self, subset: list[str]) -> DataFrameT:
        return self.lazy().unique(subset).collect()

    def rename(self, mapping: dict[str, str]) -> DataFrameT:
        return self.lazy().rename(mapping).collect()

    def to_numpy(self) -> Any:
        return self.dataframe.to_numpy()

    def to_pandas(self) -> Any:
        if self._implementation == "pandas":
            return self.dataframe
        elif self._implementation == "cudf":
            return self.dataframe.to_pandas()
        elif self._implementation == "modin":
            return self.dataframe._to_pandas()
        msg = f"Unknown implementation: {self._implementation}"
        raise TypeError(msg)


class LazyFrame(LazyFrameT):
    """dataframe object"""

    def __init__(
        self,
        dataframe: Any,
        *,
        api_version: str,
        implementation: str,
    ) -> None:
        self._validate_columns(dataframe.columns)
        self._df = dataframe.reset_index(drop=True)
        self.api_version = api_version
        self._implementation = implementation

    @property
    def columns(self) -> list[str]:
        return self.dataframe.columns.tolist()

    def __repr__(self) -> str:  # pragma: no cover
        header = f" Standard DataFrame (api_version={self.api_version}) "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Add `.dataframe` to see native output         |\n"
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
            (self.dataframe.dtypes == "bool") | (self.dataframe.dtypes == "boolean")
        ).all():
            msg = "'any' can only be called on DataFrame where all dtypes are 'bool'"
            raise TypeError(
                msg,
            )

    def _from_dataframe(self, df: Any) -> LazyFrameT:
        return LazyFrame(
            df,
            api_version=self.api_version,
            implementation=self._implementation,
        )

    @property
    def dataframe(self) -> Any:
        return self._df

    def __lazyframe_namespace__(
        self,
    ) -> NamespaceT:
        return narwhals.pandas_like.Namespace(
            api_version=self.api_version,
            implementation=self._implementation,  # type: ignore[attr-defined]
        )

    def group_by(self, *keys: str | Iterable[str]) -> LazyGroupByT:
        from narwhals.pandas_like.group_by_object import LazyGroupBy

        return LazyGroupBy(self, flatten_str(*keys), api_version=self.api_version)

    def select(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> LazyFrameT:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        df = horizontal_concat(
            [series.series for series in new_series],  # type: ignore[attr-defined]
            implementation=self._implementation,
        )
        return self._from_dataframe(df)

    def filter(
        self,
        *predicates: IntoExpr | Iterable[IntoExpr],
    ) -> LazyFrameT:
        plx = self.__lazyframe_namespace__()
        expr = plx.all_horizontal(*predicates)
        # Safety: all_horizontal's expression only returns a single column.
        mask = expr.call(self)[0]  # type: ignore[attr-defined]
        _mask = validate_dataframe_comparand(mask)
        return self._from_dataframe(self.dataframe.loc[_mask])

    def with_columns(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> LazyFrameT:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        df = self.dataframe.assign(
            **{
                series.name: series.series  # type: ignore[attr-defined]
                for series in new_series
            }
        )
        return self._from_dataframe(df)

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Iterable[bool] = False,
    ) -> LazyFrameT:
        flat_keys = flatten_str([*flatten_str(by), *more_by])
        if not flat_keys:
            flat_keys = self.dataframe.columns.tolist()
        df = self.dataframe
        if isinstance(descending, bool):
            ascending: bool | list[bool] = not descending
        else:
            ascending = [not d for d in descending]
        return self._from_dataframe(
            df.sort_values(flat_keys, ascending=ascending),
        )

    # Other
    def join(
        self,
        other: LazyFrameT,
        *,
        how: Literal["left", "inner", "outer"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> LazyFrameT:
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
            self.dataframe.merge(
                other.dataframe,  # type: ignore[attr-defined]
                left_on=left_on,
                right_on=right_on,
                how=how,
            ),
        )

    # Conversion
    def collect(self) -> DataFrameT:
        return DataFrame(
            self.dataframe,
            api_version=self.api_version,
            implementation=self._implementation,
        )

    def cache(self) -> LazyFrameT:
        return self

    def head(self, n: int) -> LazyFrameT:
        return self._from_dataframe(self.dataframe.head(n))

    def unique(self, subset: list[str]) -> LazyFrameT:
        return self._from_dataframe(self.dataframe.drop_duplicates(subset=subset))

    def rename(self, mapping: dict[str, str]) -> LazyFrameT:
        return self._from_dataframe(self.dataframe.rename(columns=mapping))
