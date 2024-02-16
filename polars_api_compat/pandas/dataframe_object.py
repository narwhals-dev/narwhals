from __future__ import annotations

from polars_api_compat.utils import (
    evaluate_into_exprs,
)

from polars_api_compat.utils import flatten_into_expr, flatten_str
import collections
from typing import TYPE_CHECKING, Iterable
from typing import Any
from typing import Literal

import pandas as pd

import polars_api_compat
from polars_api_compat.utils import validate_dataframe_comparand

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars_api_compat.pandas.group_by_object import GroupBy

    from polars_api_compat.spec import (
        DataFrame as DataFrameT,
        IntoExpr,
        Namespace as NamespaceT,
    )
else:
    DataFrameT = object


class DataFrame(DataFrameT):
    """dataframe object"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        api_version: str,
    ) -> None:
        self._validate_columns(dataframe.columns)
        self._dataframe = dataframe
        self.api_version = api_version

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

    def _from_dataframe(self, df: pd.DataFrame) -> DataFrame:
        return DataFrame(
            df,
            api_version=self.api_version,
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @property
    def columns(self) -> list[str]:
        return self.dataframe.columns.tolist()  # type: ignore[no-any-return]

    def __dataframe_namespace__(
        self,
    ) -> NamespaceT:
        return polars_api_compat.pandas.Namespace(
            api_version=self.api_version,
        )

    def shape(self) -> tuple[int, int]:
        return df.shape  # type: ignore[no-any-return]

    def group_by(self, *keys: str) -> GroupBy:
        from polars_api_compat.pandas.group_by_object import GroupBy

        # todo: do this properly
        out = []
        for key in keys:
            if isinstance(key, str):
                out.append(key)
            elif isinstance(key, (list, tuple)):
                out.extend(key)
            elif key not in self.columns:
                msg = f"key {key} not present in DataFrame's columns"
                raise KeyError(msg)
        return GroupBy(self, out, api_version=self.api_version)

    def select(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> DataFrame:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        df = pd.concat(
            {series.name: series.series for series in new_series}, axis=1, copy=False
        )
        return self._from_dataframe(df)

    def filter(
        self,
        *predicates: IntoExpr | Iterable[IntoExpr],
    ) -> DataFrame:
        plx = self.__dataframe_namespace__()
        # Safety: all_horizontal's expression only returns a single column.
        filter = plx.all_horizontal(*predicates).call(self)[0]
        _mask = validate_dataframe_comparand(self, filter)
        return self._from_dataframe(self.dataframe.loc[_mask])

    def with_columns(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> DataFrame:
        new_series = evaluate_into_exprs(self, *exprs, **named_exprs)
        df = self.dataframe.assign(
            **{series.name: series.series for series in new_series}
        )
        return self._from_dataframe(df)

    def sort(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
    ) -> DataFrame:
        keys = flatten_into_expr(*keys)
        if not keys:
            keys = self.dataframe.columns.tolist()
        df = self.dataframe
        return self._from_dataframe(
            df.sort_values(keys, ascending=ascending),
        )

    # Other
    def join(
        self,
        other: DataFrame,
        *,
        how: Literal["left", "inner", "outer"],
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> DataFrame:
        if how not in ["left", "inner", "outer"]:
            msg = f"Expected 'left', 'inner', 'outer', got: {how}"
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
                other.dataframe,
                left_on=left_on,
                right_on=right_on,
                how=how,
            ),
        )

    # Conversion

    def to_numpy(self, dtype: DType | None = None) -> Any:
        return self.dataframe.to_numpy()


class LazyFrame(DataFrameT):
    """dataframe object"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        api_version: str,
    ) -> None:
        self._validate_columns(dataframe.columns)
        self._dataframe = dataframe
        self.api_version = api_version

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

    def _from_dataframe(self, df: pd.DataFrame) -> DataFrame:
        return DataFrame(
            df,
            api_version=self.api_version,
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @property
    def columns(self) -> list[str]:
        return self.dataframe.columns.tolist()  # type: ignore[no-any-return]

    def __dataframe_namespace__(
        self,
    ) -> polars_api_compat.pandas.Namespace:
        return polars_api_compat.pandas.Namespace(
            api_version=self.api_version,
        )

    def group_by(self, *keys: str) -> GroupBy:
        from polars_api_compat.pandas.group_by_object import GroupBy

        # todo: do this properly
        out = []
        for key in keys:
            if isinstance(key, str):
                out.append(key)
            elif isinstance(key, (list, tuple)):
                out.extend(key)
            elif key not in self.columns:
                msg = f"key {key} not present in DataFrame's columns"
                raise KeyError(msg)
        return GroupBy(self, out, api_version=self.api_version)

    def select(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> DataFrame:
        new_cols = evaluate_exprs(self, *exprs, **named_exprs)
        df = pd.concat(
            {column.name: column.column for column in new_cols}, axis=1, copy=False
        )
        return self._from_dataframe(df)

    def filter(
        self,
        *mask: Column,
    ) -> DataFrame:
        plx = self.__dataframe_namespace__()
        # Safety: all_horizontal's expression only returns a single column.
        filter = evaluate_exprs(self, plx.all_horizontal(*mask))[0]
        _mask = validate_dataframe_comparand(self, filter)
        df = self.dataframe
        df = df.loc[_mask]
        return self._from_dataframe(df)

    def with_columns(
        self,
        *exprs,
        **named_exprs,
    ) -> DataFrame:
        new_cols = evaluate_exprs(self, *exprs, **named_exprs)
        df = self.dataframe.assign(
            **{column.name: column.column for column in new_cols}
        )
        return self._from_dataframe(df)

    def sort(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
    ) -> DataFrame:
        keys = flatten_str(*keys)
        if not keys:
            keys = self.dataframe.columns.tolist()
        df = self.dataframe
        return self._from_dataframe(
            df.sort_values(keys, ascending=ascending),
        )

    # Other
    def join(
        self,
        other: DataFrame,
        *,
        how: Literal["left", "inner", "outer"],
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> DataFrame:
        if how not in ["left", "inner", "outer"]:
            msg = f"Expected 'left', 'inner', 'outer', got: {how}"
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
                other.dataframe,
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
        )
