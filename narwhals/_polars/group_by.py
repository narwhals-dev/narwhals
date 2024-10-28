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
        if drop_null_keys:
            self._grouped = df.drop_nulls(keys)._native_frame.group_by(keys)
        else:
            self._grouped = df._native_frame.group_by(keys)

    def agg(self, *aggs: Any, **named_aggs: Any) -> PolarsDataFrame:
        aggs, named_aggs = extract_args_kwargs(aggs, named_aggs)  # type: ignore[assignment]
        return self._compliant_frame._from_native_frame(  # type: ignore[no-any-return]
            self._grouped.agg(*aggs, **named_aggs),
        )

    def __iter__(self) -> Any:
        for key, df in self._grouped:
            yield tuple(key), self._compliant_frame._from_native_frame(df)


class PolarsLazyGroupBy:
    def __init__(self, df: Any, keys: list[str], *, drop_null_keys: bool) -> None:
        self._compliant_frame = df
        self.keys = keys
        if drop_null_keys:
            self._grouped = df.drop_nulls(keys)._native_frame.group_by(keys)
        else:
            self._grouped = df._native_frame.group_by(keys)

    def agg(self, *aggs: Any, **named_aggs: Any) -> PolarsLazyFrame:
        aggs, named_aggs = extract_args_kwargs(aggs, named_aggs)  # type: ignore[assignment]
        return self._compliant_frame._from_native_frame(  # type: ignore[no-any-return]
            self._grouped.agg(*aggs, **named_aggs),
        )
