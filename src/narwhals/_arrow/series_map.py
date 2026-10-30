from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._arrow.utils import ArrowSeriesNamespace, map_get
from narwhals._compliant.any_namespace import MapNamespace

if TYPE_CHECKING:
    from narwhals._arrow.series import ArrowSeries
    from narwhals.typing import NonNestedLiteral


class ArrowSeriesMapNamespace(ArrowSeriesNamespace, MapNamespace["ArrowSeries"]):
    def get(self, key: NonNestedLiteral) -> ArrowSeries:
        return self.with_native(map_get(self.native, key))
