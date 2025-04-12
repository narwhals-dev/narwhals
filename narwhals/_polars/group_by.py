from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterator
from typing import Sequence
from typing import cast

if TYPE_CHECKING:
    import polars as pl
    from polars.dataframe.group_by import GroupBy as NativeGroupBy
    from polars.lazyframe.group_by import LazyGroupBy as NativeLazyGroupBy

    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame
    from narwhals._polars.expr import PolarsExpr


class PolarsGroupBy:
    _compliant_frame: PolarsDataFrame
    _keys: Sequence[str]
    _grouped: NativeGroupBy

    @property
    def compliant(self) -> PolarsDataFrame:
        return self._compliant_frame

    def __init__(
        self, df: PolarsDataFrame, keys: Sequence[PolarsExpr], /, *, drop_null_keys: bool
    ) -> None:
        by: Sequence[pl.Expr | str]
        if not drop_null_keys:
            by = [arg.native for arg in keys]
            self._compliant_frame = df
        else:
            by = [arg.native.meta.output_name() for arg in keys]
            self._compliant_frame = df.with_columns(*keys).drop_nulls(by)
        self._grouped = self.compliant.native.group_by(by)

    def agg(self, *aggs: PolarsExpr) -> PolarsDataFrame:
        return self.compliant._with_native(self._grouped.agg(arg.native for arg in aggs))

    def __iter__(self) -> Iterator[tuple[tuple[str, ...], PolarsDataFrame]]:
        for key, df in self._grouped:
            yield tuple(cast("str", key)), self.compliant._with_native(df)


class PolarsLazyGroupBy:
    _compliant_frame: PolarsLazyFrame
    _keys: Sequence[str]
    _grouped: NativeLazyGroupBy

    @property
    def compliant(self) -> PolarsLazyFrame:
        return self._compliant_frame

    def __init__(
        self, df: PolarsLazyFrame, keys: Sequence[PolarsExpr], /, *, drop_null_keys: bool
    ) -> None:
        by: Sequence[pl.Expr | str]
        if not drop_null_keys:
            by = [arg.native for arg in keys]
            self._compliant_frame = df
        else:
            by = [arg.native.meta.output_name() for arg in keys]
            self._compliant_frame = df.with_columns(*keys).drop_nulls(by)
        self._grouped = self.compliant.native.group_by(by)

    def agg(self, *aggs: PolarsExpr) -> PolarsLazyFrame:
        return self.compliant._with_native(self._grouped.agg(arg.native for arg in aggs))
