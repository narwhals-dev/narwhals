from __future__ import annotations

from polars_api_compat.utils import parse_exprs

from polars_api_compat.utils import flatten_strings
import collections
import warnings
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Literal
from typing import NoReturn

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype

import polars_api_compat
from polars_api_compat.utils import validate_column_comparand
from polars_api_compat.utils import validate_dataframe_comparand

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from dataframe_api import DataFrame as DataFrameT
    from dataframe_api.typing import AnyScalar
    from dataframe_api.typing import Column
    from dataframe_api.typing import DType
    from dataframe_api.typing import NullType
    from dataframe_api.typing import Scalar

    from polars_api_compat.pandas import Expr
    from polars_api_compat.pandas.group_by_object import GroupBy
else:
    DataFrameT = object


class DataFrame(DataFrameT):
    """dataframe object"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        api_version: str,
        is_persisted: bool = False,
    ) -> None:
        self._is_persisted = is_persisted
        self._validate_columns(dataframe.columns)
        self._dataframe = dataframe
        self._api_version = api_version

    # Validation helper methods

    def _validate_is_persisted(self) -> pd.DataFrame:
        if not self._is_persisted:
            msg = "Method requires you to call `.persist` first.\n\nNote: `.persist` forces materialisation in lazy libraries and so should be called as late as possible in your pipeline. Use with care."
            raise ValueError(
                msg,
            )
        return self.dataframe

    def __repr__(self) -> str:  # pragma: no cover
        header = f" Standard DataFrame (api_version={self._api_version}) "
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
            api_version=self._api_version,
            is_persisted=self._is_persisted,
        )

    # Properties
    @property
    def schema(self) -> dict[str, DType]:
        return {
            column_name: polars_api_compat.pandas.map_pandas_dtype_to_standard_dtype(
                dtype.name,
            )
            for column_name, dtype in self.dataframe.dtypes.items()
        }

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @property
    def columns(self) -> list[str]:
        return self.dataframe.columns.tolist()  # type: ignore[no-any-return]

    # In the Standard

    def __dataframe_namespace__(
        self,
    ) -> polars_api_compat.pandas.Namespace:
        return polars_api_compat.pandas.Namespace(
            api_version=self._api_version,
        )

    def iter_columns(self) -> Iterator[Column]:
        return (self.get_column(col_name) for col_name in self.column_names)

    def get_column(self, name: str) -> Column:
        if not self._is_persisted:
            msg = "`get_column` can only be called on persisted DataFrame."
            raise ValueError(msg)
        from polars_api_compat.pandas.column_object import Series

        return Series(
            self.dataframe.loc[:, name],
            api_version=self._api_version,
        )

    def shape(self) -> tuple[int, int]:
        df = self._validate_is_persisted()
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
            elif key not in self.column_names:
                msg = f"key {key} not present in DataFrame's columns"
                raise KeyError(msg)
        return GroupBy(self, out, api_version=self._api_version)

    def select(
        self,
        *exprs: str | Expr,
        **named_exprs,
    ) -> DataFrame:
        new_cols = parse_exprs(self, *exprs, **named_exprs)
        df = pd.concat(
            {column.name: column.column for column in new_cols}, axis=1, copy=False
        )
        return self._from_dataframe(df)

    def gather(
        self,
        indices: Column,
    ) -> DataFrame:
        _indices = validate_column_comparand(self, indices)
        return self._from_dataframe(
            self.dataframe.iloc[_indices.to_list(), :],
        )

    def slice_rows(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
    ) -> DataFrame:
        return self._from_dataframe(self.dataframe.iloc[start:stop:step])

    def filter(
        self,
        *mask: Column,
    ) -> DataFrame:
        plx = self.__dataframe_namespace__()
        # Safety: all_horizontal's expression only returns a single column.
        filter = parse_exprs(self, plx.all_horizontal(*mask))[0]
        _mask = validate_dataframe_comparand(self, filter)
        df = self.dataframe
        df = df.loc[_mask]
        return self._from_dataframe(df)

    def with_columns(
        self,
        *exprs,
        **named_exprs,
    ) -> DataFrame:
        new_cols = parse_exprs(self, *exprs, **named_exprs)
        df = self.dataframe.assign(
            **{column.name: column.column for column in new_cols}
        )
        return self._from_dataframe(df)

    def drop(self, *labels: str) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.drop(list(labels), axis=1),
        )

    def rename(self, mapping: Mapping[str, str]) -> DataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            msg = f"Expected Mapping, got: {type(mapping)}"
            raise TypeError(msg)
        return self._from_dataframe(
            self.dataframe.rename(columns=mapping),
        )

    def sort(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
    ) -> DataFrame:
        keys = flatten_strings(*keys)
        if not keys:
            keys = self.dataframe.columns.tolist()
        df = self.dataframe
        return self._from_dataframe(
            df.sort_values(keys, ascending=ascending),
        )

    # Binary operations

    def __eq__(self, other: AnyScalar) -> DataFrame:  # type: ignore[override]
        return self._from_dataframe(self.dataframe.__eq__(other))

    def __ne__(self, other: AnyScalar) -> DataFrame:  # type: ignore[override]
        return self._from_dataframe(self.dataframe.__ne__(other))

    def __ge__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(self.dataframe.__ge__(other))

    def __gt__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(self.dataframe.__gt__(other))

    def __le__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(self.dataframe.__le__(other))

    def __lt__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(self.dataframe.__lt__(other))

    def __and__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.__and__(other),
        )

    def __rand__(self, other: Column | AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self.__and__(_other)

    def __or__(self, other: AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self._from_dataframe(self.dataframe.__or__(_other))

    def __ror__(self, other: Column | AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self.__or__(_other)

    def __add__(self, other: AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__add__(_other),
        )

    def __radd__(self, other: Column | AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self.__add__(_other)

    def __sub__(self, other: AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__sub__(_other),
        )

    def __rsub__(self, other: Column | AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return -1 * self.__sub__(_other)

    def __mul__(self, other: AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__mul__(_other),
        )

    def __rmul__(self, other: Column | AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self.__mul__(_other)

    def __truediv__(self, other: AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__truediv__(_other),
        )

    def __rtruediv__(self, other: Column | AnyScalar) -> DataFrame:  # pragma: no cover
        _other = validate_column_comparand(self, other)
        raise NotImplementedError

    def __floordiv__(self, other: AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__floordiv__(_other),
        )

    def __rfloordiv__(self, other: Column | AnyScalar) -> DataFrame:  # pragma: no cover
        _other = validate_column_comparand(self, other)
        raise NotImplementedError

    def __pow__(self, other: AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__pow__(_other),
        )

    def __rpow__(self, other: Column | AnyScalar) -> DataFrame:  # pragma: no cover
        _other = validate_column_comparand(self, other)
        raise NotImplementedError

    def __mod__(self, other: AnyScalar) -> DataFrame:
        _other = validate_column_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__mod__(other),
        )

    def __rmod__(self, other: Column | AnyScalar) -> DataFrame:  # type: ignore[misc]  # pragma: no cover
        _other = validate_column_comparand(self, other)
        raise NotImplementedError

    def __divmod__(
        self,
        other: DataFrame | AnyScalar,
    ) -> tuple[DataFrame, DataFrame]:
        _other = validate_column_comparand(self, other)
        quotient, remainder = self.dataframe.__divmod__(_other)
        return self._from_dataframe(quotient), self._from_dataframe(
            remainder,
        )

    # Unary

    def __invert__(self) -> DataFrame:
        self._validate_booleanness()
        return self._from_dataframe(self.dataframe.__invert__())

    def __iter__(self) -> NoReturn:
        raise NotImplementedError

    # Reductions

    def any(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        self._validate_booleanness()
        return self._from_dataframe(
            self.dataframe.any().to_frame().T,
        )

    def all(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        self._validate_booleanness()
        return self._from_dataframe(
            self.dataframe.all().to_frame().T,
        )

    def min(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.min().to_frame().T,
        )

    def max(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.max().to_frame().T,
        )

    def sum(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.sum().to_frame().T,
        )

    def prod(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.prod().to_frame().T,
        )

    def median(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.median().to_frame().T,
        )

    def mean(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.mean().to_frame().T,
        )

    def std(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.std().to_frame().T,
        )

    def var(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.var().to_frame().T,
        )

    # Transformations

    def is_null(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result: list[pd.Series] = []
        for column in self.dataframe.columns:
            result.append(self.dataframe[column].isna())
        return self._from_dataframe(pd.concat(result, axis=1))

    def is_nan(self) -> DataFrame:
        pdx = self.__dataframe_namespace__()
        return self.with_columns(*[pdx.col(col).is_nan() for col in self.column_names])

    def fill_nan(self, value: float | Scalar | NullType) -> DataFrame:
        _value = validate_column_comparand(self, value)
        new_cols = {}
        df = self.dataframe
        for col in df.columns:
            ser = df[col].copy()
            if is_extension_array_dtype(ser.dtype):
                if self.__dataframe_namespace__().is_null(_value):
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = pd.NA
                else:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = _value
            else:
                if self.__dataframe_namespace__().is_null(_value):
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = np.nan
                else:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = _value
            new_cols[col] = ser
        df = pd.DataFrame(new_cols)
        return self._from_dataframe(df)

    def fill_null(
        self,
        value: AnyScalar,
        *,
        column_names: list[str] | None = None,
    ) -> DataFrame:
        if column_names is None:
            column_names = self.dataframe.columns.tolist()
        assert isinstance(column_names, list)  # help type checkers
        pdx = self.__dataframe_namespace__()
        return self.with_columns(
            *[pdx.col(col).fill_null(value) for col in column_names],
        )

    def drop_nulls(
        self,
        *,
        column_names: list[str] | None = None,
    ) -> DataFrame:
        pdx = self.__dataframe_namespace__()
        mask = ~pdx.any_horizontal(
            *[
                pdx.col(col_name).is_null()
                for col_name in column_names or self.column_names
            ],
        )
        return self.filter(mask)

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

        if overlap := (set(self.column_names) - set(left_on)).intersection(
            set(other.column_names) - set(right_on),
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

    def collect(self) -> DataFrame:
        if self._is_persisted:
            warnings.warn(
                "Calling `.persist` on DataFrame that was already persisted",
                UserWarning,
                stacklevel=2,
            )
        return DataFrame(
            self.dataframe,
            api_version=self._api_version,
            is_persisted=True,
        )

    # Conversion

    def to_array(self, dtype: DType | None = None) -> Any:
        self._validate_is_persisted()
        return self.dataframe.to_numpy()

    def cast(self, dtypes: Mapping[str, DType]) -> DataFrame:
        from polars_api_compat.pandas import (
            map_standard_dtype_to_pandas_dtype,
        )

        df = self._dataframe
        return self._from_dataframe(
            df.astype(
                {
                    col: map_standard_dtype_to_pandas_dtype(dtype)
                    for col, dtype in dtypes.items()
                },
            ),
        )
