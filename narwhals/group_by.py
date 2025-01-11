from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import TypeVar
from typing import cast

from narwhals.dataframe import DataFrame
from narwhals.dataframe import LazyFrame
from narwhals.utils import tupleify

if TYPE_CHECKING:
    from narwhals.typing import IntoExpr

DataFrameT = TypeVar("DataFrameT")
LazyFrameT = TypeVar("LazyFrameT")


class GroupBy(Generic[DataFrameT]):
    def __init__(self, df: DataFrameT, *keys: str, drop_null_keys: bool) -> None:
        self._df = cast(DataFrame[Any], df)
        self._keys = keys
        self._grouped = self._df._compliant_frame.group_by(
            *self._keys, drop_null_keys=drop_null_keys
        )

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> DataFrameT:
        """Compute aggregations for each group of a group by operation.

        Arguments:
            aggs: Aggregations to compute for each group of the group by operation,
                specified as positional arguments.
            named_aggs: Additional aggregations, specified as keyword arguments.

        Returns:
            A new Dataframe.
        """
        aggs, named_aggs = self._df._flatten_and_extract(*aggs, **named_aggs)
        return self._df._from_compliant_dataframe(  # type: ignore[return-value]
            self._grouped.agg(*aggs, **named_aggs),
        )

    def __iter__(self) -> Iterator[tuple[Any, DataFrameT]]:
        yield from (  # type: ignore[misc]
            (tupleify(key), self._df._from_compliant_dataframe(df))
            for (key, df) in self._grouped.__iter__()
        )


class LazyGroupBy(Generic[LazyFrameT]):
    def __init__(self, df: LazyFrameT, *keys: str, drop_null_keys: bool) -> None:
        self._df = cast(LazyFrame[Any], df)
        self._keys = keys
        self._grouped = self._df._compliant_frame.group_by(
            *self._keys, drop_null_keys=drop_null_keys
        )

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> LazyFrameT:
        """Compute aggregations for each group of a group by operation.

        If a library does not support lazy execution, then this is a no-op.

        Arguments:
            aggs: Aggregations to compute for each group of the group by operation,
                specified as positional arguments.
            named_aggs: Additional aggregations, specified as keyword arguments.

        Returns:
            A new LazyFrame.
        """
        aggs, named_aggs = self._df._flatten_and_extract(*aggs, **named_aggs)
        return self._df._from_compliant_dataframe(  # type: ignore[return-value]
            self._grouped.agg(*aggs, **named_aggs),
        )
