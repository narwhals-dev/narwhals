from __future__ import annotations

from typing import Generic

from narwhals._constants import GET_CATEGORIES_DEPRECATION_TEMPLATE
from narwhals._exceptions import issue_deprecation_warning
from narwhals._utils import Version
from narwhals.dtypes import String
from narwhals.typing import SeriesT


class SeriesCatNamespace(Generic[SeriesT]):
    def __init__(self, series: SeriesT) -> None:
        self._narwhals_series = series

    def get_categories(self) -> SeriesT:
        """Get unique categories from column.

        Warning:
            This is deprecated and it will be removed in a future version, as Polars
            removed its own `cat.get_categories`.
            To get the distinct values present in a Categorical column, use
            [`Series.unique`][narwhals.series.Series.unique].
            For the fixed category list of an Enum, use its `dtype.categories`.

            Until it is removed, it is implemented as
            `unique(maintain_order=True).drop_nulls().cast(String)`, so it returns only
            the values which are actually present, in order of appearance.

        Note:
            In `narwhals.stable.v1` and `narwhals.stable.v2` this stays available, and
            keeps dispatching to each backend's native implementation.

        Examples:
            >>> import pandas as pd
            >>> import narwhals.stable.v2 as nw
            >>> s_native = pd.Series(["apple", "mango", "mango"], dtype="category")
            >>> s = nw.from_native(s_native, series_only=True)
            >>> s.cat.get_categories().to_native()
            0    apple
            1    mango
            dtype: str
        """
        if self._narwhals_series._version in {Version.V1, Version.V2}:
            return self._narwhals_series._with_compliant(
                self._narwhals_series._compliant_series.cat.get_categories()
            )

        issue_deprecation_warning(
            GET_CATEGORIES_DEPRECATION_TEMPLATE.format(cls="Series"), _version="2.26.0"
        )
        return self._narwhals_series.unique(maintain_order=True).drop_nulls().cast(String)
