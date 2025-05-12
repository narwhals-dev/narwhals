from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Mapping

from narwhals.sql import table as nw_table
from narwhals.stable.v1.utils import _stableify

if TYPE_CHECKING:
    from sqlframe.standalone.dataframe import StandaloneDataFrame

    from narwhals.stable.v1.dataframe import LazyFrame
    from narwhals.stable.v1.dtypes import DType
    from narwhals.stable.v1.schema import Schema


def table(
    name: str, schema: Mapping[str, DType] | Schema
) -> LazyFrame[StandaloneDataFrame]:
    return _stableify(nw_table(name, schema))
