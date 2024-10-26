from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._polars.utils import extract_args_kwargs

if TYPE_CHECKING:
    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame


class PolarsGroupBy:
    def __init__(self, df: Any, keys: list[str], *, drop_null_keys: bool) -> None:
        self._compliant_frame = df
        self.keys = keys
        self._grouped = df._native_frame.group_by(keys)
        self._drop_null_keys = drop_null_keys

    def agg(self, *aggs: Any, **named_aggs: Any) -> PolarsDataFrame:
        aggs, named_aggs = extract_args_kwargs(aggs, named_aggs)  # type: ignore[assignment]
        result = self._compliant_frame._from_native_frame(
            self._grouped.agg(*aggs, **named_aggs),
        )
        if self._drop_null_keys:
            return result.drop_nulls(self.keys)  # type: ignore[no-any-return]
        return result  # type: ignore[no-any-return]

    def __iter__(self) -> Any:
        if self._drop_null_keys:
            for key, df in self._grouped:
                key_tuple = tuple(key)
                if all(x is not None for x in key_tuple):
                    yield key_tuple, self._compliant_frame._from_native_frame(df)
        else:
            for key, df in self._grouped:
                key_tuple = tuple(key)
                yield key_tuple, self._compliant_frame._from_native_frame(df)


class PolarsLazyGroupBy:
    def __init__(self, df: Any, keys: list[str], *, drop_null_keys: bool) -> None:
        self._compliant_frame = df
        self.keys = keys
        self._grouped = df._native_frame.group_by(keys)
        self._drop_null_keys = drop_null_keys

    def agg(self, *aggs: Any, **named_aggs: Any) -> PolarsLazyFrame:
        aggs, named_aggs = extract_args_kwargs(aggs, named_aggs)  # type: ignore[assignment]
        result = self._compliant_frame._from_native_frame(
            self._grouped.agg(*aggs, **named_aggs),
        )
        if self._drop_null_keys:
            return result.drop_nulls(self.keys)  # type: ignore[no-any-return]
        return result  # type: ignore[no-any-return]
