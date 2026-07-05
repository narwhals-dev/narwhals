from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerSeriesNamespace
from narwhals._compliant.any_namespace import StructNamespace

if TYPE_CHECKING:
    from narwhals_dict.series import DictSeries


class DictSeriesStructNamespace(
    EagerSeriesNamespace["DictSeries", Any], StructNamespace["DictSeries"]
):
    def field(self, name: str) -> DictSeries:
        """Extract `name` from each struct value, propagating outer nulls."""
        native = [None if row is None else row[name] for row in self.native]
        return self.with_native(native).alias(name)
