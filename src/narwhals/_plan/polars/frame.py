from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import polars as pl

from narwhals._plan.polars import compat
from narwhals._utils import Implementation, Version, _into_arrow_table

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.dataframe import DataFrame as NwDataFrame
    from narwhals._translate import IntoArrowTable

MAIN = Version.MAIN


class PolarsFrame:
    __slots__ = ()
    implementation: ClassVar = Implementation.POLARS
    version: ClassVar[Version] = Version.MAIN

    @classmethod
    def from_arrow(cls, data: IntoArrowTable, /) -> Self:
        if compat.CONSTRUCTOR_ACCEPTS_PYCAPSULE:
            native = pl.DataFrame(data)
        else:  # pragma: no cover
            # NOTE: Hack to reuse `main`
            context = cls.version.namespace.from_backend("polars").compliant
            native = cast("pl.DataFrame", pl.from_arrow(_into_arrow_table(data, context)))
        return cls.from_polars(native)

    @classmethod
    def from_polars(cls, frame: pl.DataFrame, /) -> Self:
        # the impl of this will differ for eager and lazy
        raise NotImplementedError

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame, /) -> Self:
        return cls.from_polars(pl.from_pandas(frame))

    @classmethod
    def from_narwhals(
        cls, frame: NwDataFrame[Any, Any] | CompliantDataFrame[Any, Any], /
    ) -> Self:
        return cls.from_polars(frame.to_polars())

    @classmethod
    def from_compliant(cls, frame: CompliantDataFrame[Any, Any], /) -> Self:
        return cls.from_narwhals(frame)
