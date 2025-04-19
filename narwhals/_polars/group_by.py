from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterator
from typing import Sequence
from typing import cast

from narwhals.utils import flatten
from narwhals.utils import is_sequence_of

if TYPE_CHECKING:
    from polars.dataframe.group_by import GroupBy as NativeGroupBy
    from polars.lazyframe.group_by import LazyGroupBy as NativeLazyGroupBy

    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame
    from narwhals._polars.expr import PolarsExpr


class PolarsGroupBy:
    _compliant_frame: PolarsDataFrame
    _grouped: NativeGroupBy
    _drop_null_keys: bool

    @property
    def compliant(self) -> PolarsDataFrame:
        return self._compliant_frame

    def __init__(
        self,
        df: PolarsDataFrame,
        keys: Sequence[PolarsExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._compliant_frame = df
        self._drop_null_keys = drop_null_keys

        if is_sequence_of(keys, str):
            self._output_names = list(keys)
            self._grouped = self.compliant.native.group_by(list(keys))

        else:
            self._output_names = flatten(
                [
                    arg.native.meta.root_names()
                    if arg.native.meta.has_multiple_outputs()
                    else arg.native.meta.output_name()
                    for arg in keys
                ]
            )

            self._grouped = self.compliant.native.group_by(*[arg.native for arg in keys])

    def agg(self, *aggs: PolarsExpr) -> PolarsDataFrame:
        agg_result = self._grouped.agg(arg.native for arg in aggs)

        if self._drop_null_keys:
            agg_result = agg_result.drop_nulls(subset=self._output_names)

        return self.compliant._with_native(agg_result)

    def __iter__(self) -> Iterator[tuple[tuple[str, ...], PolarsDataFrame]]:
        for key, df in self._grouped:
            if self._drop_null_keys and any(k is None for k in key):
                continue
            yield tuple(cast("str", key)), self.compliant._with_native(df)


class PolarsLazyGroupBy:
    _compliant_frame: PolarsLazyFrame
    _grouped: NativeLazyGroupBy
    _drop_null_keys: bool
    _output_names: list[str]

    @property
    def compliant(self) -> PolarsLazyFrame:
        return self._compliant_frame

    def __init__(
        self,
        df: PolarsLazyFrame,
        keys: Sequence[PolarsExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._compliant_frame = df
        self._drop_null_keys = drop_null_keys
        if is_sequence_of(keys, str):
            self._output_names = list(keys)
            self._grouped = self.compliant.native.group_by(keys)

        else:
            self._output_names = flatten(
                [
                    arg.native.meta.root_names()
                    if arg.native.meta.has_multiple_outputs()
                    else arg.native.meta.output_name()
                    for arg in keys
                ]
            )

            self._grouped = self.compliant.native.group_by(*[arg.native for arg in keys])

    def agg(self, *aggs: PolarsExpr) -> PolarsLazyFrame:
        agg_result = self._grouped.agg(arg.native for arg in aggs)

        if self._drop_null_keys:
            agg_result = agg_result.drop_nulls(subset=self._output_names)

        return self.compliant._with_native(agg_result)
