from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerSeriesNamespace
from narwhals._compliant.any_namespace import CatNamespace

if TYPE_CHECKING:
    from narwhals_dict.series import DictSeries


class DictSeriesCatNamespace(
    EagerSeriesNamespace["DictSeries", Any], CatNamespace["DictSeries"]
):
    def get_categories(self) -> DictSeries:
        # `Categorical` is stored as plain strings (there is no separate categories
        # table to read back), so the categories are just the unique non-null values.
        return self.with_native(
            list(dict.fromkeys(value for value in self.native if value is not None))
        )
