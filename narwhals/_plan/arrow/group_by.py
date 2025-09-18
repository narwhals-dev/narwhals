from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa  # ignore-banned-import

from narwhals._plan.protocols import DataFrameGroupBy

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from narwhals._plan.arrow.dataframe import ArrowDataFrame
    from narwhals._plan.expressions import NamedIR
    from narwhals._plan.typing import Seq


class ArrowGroupBy(DataFrameGroupBy["ArrowDataFrame"]):
    """What narwhals is doing.

    - Keys are handled only at compliant
       - `ParseKeysGroupBy` does weird stuff
       - But has a fast path for all `str` keys
    - Aggs are handled in both levels
      - Some compliant have more restrictions
    """

    _df: ArrowDataFrame
    _grouped: pa.TableGroupBy
    _keys: Seq[NamedIR]
    _keys_names: Seq[str]

    @classmethod
    def by_names(cls, df: ArrowDataFrame, names: Seq[str], /) -> Self:
        obj = cls.__new__(cls)
        obj._df = df
        obj._keys = ()
        obj._keys_names = names
        obj._grouped = pa.TableGroupBy(df.native, list(names))
        return obj

    @classmethod
    def by_named_irs(cls, df: ArrowDataFrame, irs: Seq[NamedIR], /) -> Self:
        raise NotImplementedError

    @property
    def compliant(self) -> ArrowDataFrame:
        return self._df

    def __iter__(self) -> Iterator[tuple[Any, ArrowDataFrame]]:
        raise NotImplementedError

    def agg(self, irs: Seq[NamedIR]) -> ArrowDataFrame:
        raise NotImplementedError
