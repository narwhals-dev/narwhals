from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterator
from typing import Sequence
from typing import cast

from narwhals._polars.utils import extract_native

if TYPE_CHECKING:
    from polars._typing import IntoExpr
    from polars.dataframe.group_by import GroupBy as NativeGroupBy
    from polars.lazyframe.group_by import LazyGroupBy as NativeLazyGroupBy
    from typing_extensions import Self

    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame
    from narwhals._polars.expr import PolarsExpr


class PolarsGroupBy:
    _compliant_frame: PolarsDataFrame
    _keys: Sequence[IntoExpr]

    @property
    def compliant(self) -> PolarsDataFrame:
        return self._compliant_frame

    def __init__(
        self,
        compliant_frame: PolarsDataFrame,
        keys: Sequence[PolarsExpr],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        if not drop_null_keys:
            self._compliant_frame = compliant_frame
            self._keys = [extract_native(arg) for arg in keys]
        else:
            compliant_frame = compliant_frame.with_columns(*keys)
            self._keys = [extract_native(arg).meta.output_name() for arg in keys]
            self._compliant_frame = compliant_frame.drop_nulls(self._keys)

        self._grouped: NativeGroupBy = self._compliant_frame._native_frame.group_by(
            self._keys
        )

    def agg(self: Self, *aggs: PolarsExpr) -> PolarsDataFrame:
        return self.compliant._with_native(
            self._grouped.agg(extract_native(arg) for arg in aggs)
        )

    def __iter__(self: Self) -> Iterator[tuple[tuple[str, ...], PolarsDataFrame]]:
        for key, df in self._grouped:
            yield tuple(cast("str", key)), self.compliant._with_native(df)


class PolarsLazyGroupBy:
    _compliant_frame: PolarsLazyFrame
    _keys: Sequence[IntoExpr]

    @property
    def compliant(self) -> PolarsLazyFrame:
        return self._compliant_frame

    def __init__(
        self,
        compliant_frame: PolarsLazyFrame,
        keys: Sequence[PolarsExpr],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        if not drop_null_keys:
            self._compliant_frame = compliant_frame
            self._keys = [extract_native(arg) for arg in keys]
        else:
            compliant_frame = compliant_frame.with_columns(*keys)
            self._keys = [extract_native(arg).meta.output_name() for arg in keys]
            self._compliant_frame = compliant_frame.drop_nulls(self._keys)

        self._grouped: NativeLazyGroupBy = self._compliant_frame._native_frame.group_by(
            self._keys
        )

    def agg(self: Self, *aggs: PolarsExpr) -> PolarsLazyFrame:
        return self.compliant._with_native(
            self._grouped.agg(extract_native(arg) for arg in aggs)
        )
