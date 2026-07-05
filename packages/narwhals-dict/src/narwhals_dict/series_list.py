from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerSeriesNamespace
from narwhals._compliant.any_namespace import ListNamespace

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from narwhals.typing import NonNestedLiteral
    from narwhals_dict.series import DictSeries


def _non_null(values: Sequence[Any]) -> list[Any]:
    return [value for value in values if value is not None]


class DictSeriesListNamespace(
    EagerSeriesNamespace["DictSeries", Any], ListNamespace["DictSeries"]
):
    def _unary(self, fn: Callable[[Sequence[Any]], Any]) -> DictSeries:
        """Apply `fn` to each inner list, propagating outer nulls."""
        return self.with_native(
            [None if values is None else fn(values) for values in self.native]
        )

    def len(self) -> DictSeries:
        return self._unary(len)

    def get(self, index: int) -> DictSeries:
        return self._unary(lambda values: values[index] if index < len(values) else None)

    def contains(self, item: NonNestedLiteral) -> DictSeries:
        return self._unary(lambda values: item in values)

    def unique(self) -> DictSeries:
        return self._unary(lambda values: list(dict.fromkeys(values)))

    def min(self) -> DictSeries:
        return self._unary(
            lambda values: min(vals) if (vals := _non_null(values)) else None
        )

    def max(self) -> DictSeries:
        return self._unary(
            lambda values: max(vals) if (vals := _non_null(values)) else None
        )

    def mean(self) -> DictSeries:
        return self._unary(
            lambda values: sum(vals) / len(vals) if (vals := _non_null(values)) else None
        )

    def median(self) -> DictSeries:
        return self._unary(
            lambda values: (
                statistics.median(vals) if (vals := _non_null(values)) else None
            )
        )

    def sum(self) -> DictSeries:
        return self._unary(lambda values: sum(_non_null(values)))

    def sort(self, *, descending: bool, nulls_last: bool) -> DictSeries:
        def fn(values: Sequence[Any]) -> list[Any]:
            nulls = [value for value in values if value is None]
            vals = sorted(_non_null(values), reverse=descending)
            return vals + nulls if nulls_last else nulls + vals

        return self._unary(fn)
