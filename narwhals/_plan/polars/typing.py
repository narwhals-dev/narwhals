from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl
    from typing_extensions import Self, TypeAlias

    from narwhals._native import NativeDataFrame, NativeLazyFrame, NativeSeries
    from narwhals._plan.compliant.dataframe import (
        CompliantDataFrame as _CompliantDataFrame,
    )

    # NOTE: Some obscure picks to make each protocol disjoint from `Native*`
    # and creating a link between each `NativePolars*`

    class NativePolarsSeries(NativeSeries, Protocol):
        def is_sorted(
            self, *, descending: bool = ..., nulls_last: bool = ...
        ) -> bool: ...
        def has_nulls(self) -> bool: ...
        def to_frame(self, name: str | None = ...) -> NativePolarsDataFrame: ...
        def rechunk(self, *, in_place: bool = ...) -> Self: ...

    class NativePolarsDataFrame(NativeDataFrame, Protocol):
        @classmethod
        def deserialize(
            cls, source: Any, *, format: Any = ...
        ) -> NativePolarsDataFrame: ...
        def iter_columns(self) -> Iterator[NativePolarsSeries]: ...
        # note how this differs to `NativePolarsSeries.rechunk`
        def rechunk(self) -> NativePolarsDataFrame: ...
        def lazy(self) -> NativePolarsLazyFrame: ...

    class NativePolarsLazyFrame(NativeLazyFrame, Protocol):
        def collect(self, *args: Any, **kwds: Any) -> NativePolarsDataFrame | Any: ...
        def profile(
            self, *args: Any, **kwds: Any
        ) -> tuple[NativePolarsDataFrame, NativePolarsDataFrame]: ...
        def sink_ndjson(
            self, *args: Any, **kwds: Any
        ) -> NativePolarsLazyFrame | None: ...


CompliantDataFrame: TypeAlias = "_CompliantDataFrame[pl.DataFrame, pl.Series]"
"""Alias for `PolarsDataFrame` when used in the parameter position of a protocol method."""
