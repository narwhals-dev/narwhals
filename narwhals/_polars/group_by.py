from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterator
from typing import Sequence
from typing import cast

from narwhals._polars.utils import extract_native

if TYPE_CHECKING:
    from polars.dataframe.group_by import GroupBy as NativeGroupBy
    from polars.lazyframe.group_by import LazyGroupBy as NativeLazyGroupBy

    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame
    from narwhals._polars.expr import PolarsExpr


class PolarsGroupBy:
    _compliant_frame: PolarsDataFrame
    _keys: Sequence[str]

    @property
    def compliant(self) -> PolarsDataFrame:
        return self._compliant_frame

    def __init__(
        self, df: PolarsDataFrame, keys: Sequence[str], /, *, drop_null_keys: bool
    ) -> None:
        self._compliant_frame = df
        self._keys = list(keys)
        df = df.drop_nulls(keys) if drop_null_keys else df
        self._grouped: NativeGroupBy = df._native_frame.group_by(keys)

    def agg(self, *aggs: PolarsExpr) -> PolarsDataFrame:
        from_native = self.compliant._with_native
        return from_native(self._grouped.agg(extract_native(arg) for arg in aggs))

    def __iter__(self) -> Iterator[tuple[tuple[str, ...], PolarsDataFrame]]:
        for key, df in self._grouped:
            yield tuple(cast("str", key)), self.compliant._with_native(df)


class PolarsLazyGroupBy:
    _compliant_frame: PolarsLazyFrame
    _keys: Sequence[str]

    @property
    def compliant(self) -> PolarsLazyFrame:
        return self._compliant_frame

    def __init__(
        self, df: PolarsLazyFrame, keys: Sequence[str], /, *, drop_null_keys: bool
    ) -> None:
        self._compliant_frame = df
        self._keys = list(keys)
        df = df.drop_nulls(keys) if drop_null_keys else df
        self._grouped: NativeLazyGroupBy = df._native_frame.group_by(keys)

    def agg(self, *aggs: PolarsExpr) -> PolarsLazyFrame:
        from_native = self.compliant._with_native
        return from_native(self._grouped.agg(extract_native(arg) for arg in aggs))
