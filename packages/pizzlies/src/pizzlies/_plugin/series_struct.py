from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals.compliant import StructNamespace
from narwhals.exceptions import InvalidOperationError
from pizzlies._plugin._vendored import SeriesNamespace

if TYPE_CHECKING:
    from pizzlies._plugin.series import DictSeries


class DictSeriesStructNamespace(
    SeriesNamespace["DictSeries"], StructNamespace["DictSeries"]
):
    def field(self, name: str) -> DictSeries:
        """Extract `name` from each struct value, propagating outer nulls."""
        dtype = self.compliant.dtype
        if isinstance(dtype, self.version.dtypes.Struct) and name not in {
            field.name for field in dtype.fields
        }:
            msg = f"Struct field not found: {name!r}"
            raise InvalidOperationError(msg)
        native = [None if row is None else row.get(name) for row in self.native]
        return self.with_native(native).alias(name)
