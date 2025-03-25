from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._compliant.series import EagerSeriesCatNamespace

if TYPE_CHECKING:
    from narwhals._pandas_like.series import PandasLikeSeries


class PandasLikeSeriesCatNamespace(EagerSeriesCatNamespace["PandasLikeSeries", Any]):
    def get_categories(self) -> PandasLikeSeries:
        s = self.native
        return self.from_native(type(s)(s.cat.categories, name=s.name))
