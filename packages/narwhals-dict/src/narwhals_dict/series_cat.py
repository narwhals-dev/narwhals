from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals.compliant import CatNamespace
from narwhals_dict._vendored import SeriesNamespace

if TYPE_CHECKING:
    from narwhals_dict.series import DictSeries


class DictSeriesCatNamespace(SeriesNamespace["DictSeries"], CatNamespace["DictSeries"]):
    def get_categories(self) -> DictSeries:
        # `Categorical` is stored as plain strings (there is no separate categories
        # table to read back), so the categories are just the unique non-null values.
        return self.with_native(
            list(dict.fromkeys(value for value in self.native if value is not None))
        )
