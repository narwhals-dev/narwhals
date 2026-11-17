from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals.compliant import StructNamespace
from pizzlies._plugin._vendored import SeriesNamespace

if TYPE_CHECKING:
    from pizzlies._plugin.series import DictSeries


class DictSeriesStructNamespace(
    SeriesNamespace["DictSeries"], StructNamespace["DictSeries"]
):
    def field(self, name: str) -> DictSeries:
        """Extract `name` from each struct value, propagating outer nulls."""
        native = [None if row is None else row[name] for row in self.native]
        return self.with_native(native).alias(name)
