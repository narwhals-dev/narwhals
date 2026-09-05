from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerSeriesNamespace
from narwhals._compliant.any_namespace import ListNamespace
from narwhals_dict.utils import non_null, sort_with_nulls

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from narwhals.typing import NonNestedLiteral
    from narwhals_dict.series import DictSeries


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

    def unique(self, *, maintain_order: bool) -> DictSeries:
        # `dict.fromkeys` keeps first-appearance order, which satisfies both settings.
        return self._unary(lambda values: list(dict.fromkeys(values)))

    def min(self) -> DictSeries:
        return self._unary(
            lambda values: min(vals) if (vals := non_null(values)) else None
        )

    def max(self) -> DictSeries:
        return self._unary(
            lambda values: max(vals) if (vals := non_null(values)) else None
        )

    def mean(self) -> DictSeries:
        return self._unary(
            lambda values: sum(vals) / len(vals) if (vals := non_null(values)) else None
        )

    def median(self) -> DictSeries:
        return self._unary(
            lambda values: statistics.median(vals) if (vals := non_null(values)) else None
        )

    def sum(self) -> DictSeries:
        return self._unary(lambda values: sum(non_null(values)))

    def sort(self, *, descending: bool, nulls_last: bool) -> DictSeries:
        return self._unary(
            lambda values: sort_with_nulls(
                values, descending=descending, nulls_last=nulls_last
            )
        )
