from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from warnings import warn

from narwhals.series import Series as NwSeries
from narwhals.stable.v1.dataframe import DataFrame
from narwhals.utils import find_stacklevel
from narwhals.utils import inherit_doc

if TYPE_CHECKING:
    from typing_extensions import ParamSpec
    from typing_extensions import TypeVar

    from narwhals.stable.v1.dataframe import LazyFrame
    from narwhals.typing import IntoSeries

    FrameT = TypeVar("FrameT", "DataFrame[Any]", "LazyFrame[Any]")
    DataFrameT = TypeVar("DataFrameT", bound="DataFrame[Any]")
    LazyFrameT = TypeVar("LazyFrameT", bound="LazyFrame[Any]")
    SeriesT = TypeVar("SeriesT", bound="Series[Any]")
    IntoSeriesT = TypeVar("IntoSeriesT", bound="IntoSeries", default=Any)
    T = TypeVar("T", default=Any)
    P = ParamSpec("P")
    R = TypeVar("R")
else:
    from typing import TypeVar

    IntoSeriesT = TypeVar("IntoSeriesT", bound="IntoSeries")
    T = TypeVar("T")


class Series(NwSeries[IntoSeriesT]):
    @inherit_doc(NwSeries)
    def __init__(
        self, series: Any, *, level: Literal["full", "lazy", "interchange"]
    ) -> None:
        super().__init__(series, level=level)

    # We need to override any method which don't return Self so that type
    # annotations are correct.

    @property
    def _dataframe(self) -> type[DataFrame[Any]]:
        return DataFrame

    def to_frame(self) -> DataFrame[Any]:
        return super().to_frame()  # type: ignore[return-value]

    def value_counts(
        self,
        *,
        sort: bool = False,
        parallel: bool = False,
        name: str | None = None,
        normalize: bool = False,
    ) -> DataFrame[Any]:
        return super().value_counts(  # type: ignore[return-value]
            sort=sort, parallel=parallel, name=name, normalize=normalize
        )

    def hist(
        self,
        bins: list[float | int] | None = None,
        *,
        bin_count: int | None = None,
        include_breakpoint: bool = True,
    ) -> DataFrame[Any]:
        from narwhals.exceptions import NarwhalsUnstableWarning

        msg = (
            "`Series.hist` is being called from the stable API although considered "
            "an unstable feature."
        )
        warn(message=msg, category=NarwhalsUnstableWarning, stacklevel=find_stacklevel())
        return super().hist(  # type: ignore[return-value]
            bins=bins,
            bin_count=bin_count,
            include_breakpoint=include_breakpoint,
        )
