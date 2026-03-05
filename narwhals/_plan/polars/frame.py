from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import polars as pl

from narwhals._utils import Implementation, Version, _into_arrow_table

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.dataframe import DataFrame as NwDataFrame
    from narwhals._translate import IntoArrowTable


Incomplete: TypeAlias = Any
MAIN = Version.MAIN
POLARS = Implementation.POLARS


class PolarsFrame:
    __slots__ = ()
    implementation: ClassVar = Implementation.POLARS

    @classmethod
    def from_arrow(cls, data: IntoArrowTable, /, version: Version = MAIN) -> Self:
        if POLARS._backend_version() >= (1, 3):
            native = pl.DataFrame(data)
        else:  # pragma: no cover
            # NOTE: Hack to reuse `main`
            context = version.namespace.from_backend("polars").compliant
            native = cast("pl.DataFrame", pl.from_arrow(_into_arrow_table(data, context)))
        return cls.from_polars(native, version)

    @classmethod
    def from_polars(cls, frame: pl.DataFrame, /, version: Version = MAIN) -> Self:
        # the impl of this will differ for eager and lazy
        raise NotImplementedError

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame, /, version: Version = MAIN) -> Self:
        return cls.from_polars(pl.from_pandas(frame), version)

    @classmethod
    def from_narwhals(
        cls, frame: NwDataFrame[Any, Any] | CompliantDataFrame[Any, Any], /
    ) -> Self:
        return cls.from_polars(frame.to_polars(), frame.version)

    @classmethod
    def from_compliant(cls, frame: CompliantDataFrame[Any, Any], /) -> Self:
        return cls.from_narwhals(frame)
