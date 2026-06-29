from __future__ import annotations

from typing import TYPE_CHECKING

import ibis

from narwhals import _interchange
from narwhals._utils import Implementation

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


class IbisDataFrame(_interchange.LazyFrame[ibis.Table]):
    _implementation = Implementation.IBIS

    def to_pandas(self) -> pd.DataFrame:
        return self._compliant.native.to_pandas()

    def to_arrow(self) -> pa.Table:
        return self._compliant.native.to_pyarrow()
