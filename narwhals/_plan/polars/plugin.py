from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._plan.compliant.plugins import BuiltinPlugin
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl
    from typing_extensions import TypeIs

    from narwhals._native import NativePolars
    from narwhals._plan.polars.classes import PolarsClasses


# TODO @dangotbanned: Move to `_plan.polars.__init__.py`
class PolarsPlugin(
    BuiltinPlugin["PolarsClasses", "pl.DataFrame", "pl.LazyFrame", "pl.Series"]
):
    __slots__ = ()
    implementation = Implementation.POLARS
    sys_modules_targets = ("polars",)

    def is_native(self, obj: Any) -> TypeIs[NativePolars]:
        import polars as pl

        return isinstance(obj, (pl.DataFrame, pl.LazyFrame, pl.Series))

    def native_dataframe_classes(self) -> Iterator[type[pl.DataFrame]]:
        import polars as pl

        yield pl.DataFrame

    def native_lazyframe_classes(self) -> Iterator[type[pl.LazyFrame]]:
        import polars as pl

        yield pl.LazyFrame

    def native_series_classes(self) -> Iterator[type[pl.Series]]:
        import polars as pl

        yield pl.Series

    @property
    def __narwhals_classes__(self) -> PolarsClasses:
        from narwhals._plan.polars.classes import PolarsClasses

        return PolarsClasses()


PolarsPlugin()
