from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._arrow.utils import ArrowSeriesNamespace, map_get
from narwhals._compliant.any_namespace import MapNamespace
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from narwhals._arrow.series import ArrowSeries
    from narwhals.typing import NonNestedLiteral


class ArrowSeriesMapNamespace(ArrowSeriesNamespace, MapNamespace["ArrowSeries"]):
    len = not_implemented()

    def get(self, key: NonNestedLiteral) -> ArrowSeries:
        return self.with_native(map_get(self.native, key))
