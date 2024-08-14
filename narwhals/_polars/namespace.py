from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence

from narwhals import dtypes
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import narwhals_to_native_dtype
from narwhals.dependencies import get_polars
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame
    from narwhals._polars.expr import PolarsExpr


class PolarsNamespace:
    Int64 = dtypes.Int64
    Int32 = dtypes.Int32
    Int16 = dtypes.Int16
    Int8 = dtypes.Int8
    UInt64 = dtypes.UInt64
    UInt32 = dtypes.UInt32
    UInt16 = dtypes.UInt16
    UInt8 = dtypes.UInt8
    Float64 = dtypes.Float64
    Float32 = dtypes.Float32
    Boolean = dtypes.Boolean
    Object = dtypes.Object
    Unknown = dtypes.Unknown
    Categorical = dtypes.Categorical
    Enum = dtypes.Enum
    String = dtypes.String
    Datetime = dtypes.Datetime
    Duration = dtypes.Duration
    Date = dtypes.Date

    def __init__(self, *, backend_version: tuple[int, ...]) -> None:
        self._backend_version = backend_version
        self._implementation = Implementation.POLARS

    def __getattr__(self, attr: str) -> Any:
        from narwhals._polars.expr import PolarsExpr

        pl = get_polars()

        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return PolarsExpr(getattr(pl, attr)(*args, **kwargs))

        return func

    def len(self) -> PolarsExpr:
        from narwhals._polars.expr import PolarsExpr

        pl = get_polars()
        if self._backend_version < (0, 20, 5):  # pragma: no cover
            return PolarsExpr(pl.count().alias("len"))
        return PolarsExpr(pl.len())

    def concat(
        self,
        items: Sequence[PolarsDataFrame | PolarsLazyFrame],
        *,
        how: str = "vertical",
    ) -> PolarsDataFrame | PolarsLazyFrame:
        from narwhals._polars.dataframe import PolarsDataFrame
        from narwhals._polars.dataframe import PolarsLazyFrame

        pl = get_polars()
        dfs: list[Any] = [item._native_frame for item in items]
        result = pl.concat(dfs, how=how)
        if isinstance(result, pl.DataFrame):
            return PolarsDataFrame(result, backend_version=items[0]._backend_version)
        return PolarsLazyFrame(result, backend_version=items[0]._backend_version)

    def lit(self, value: Any, dtype: dtypes.DType | None = None) -> PolarsExpr:
        from narwhals._polars.expr import PolarsExpr

        pl = get_polars()
        if dtype is not None:
            return PolarsExpr(pl.lit(value, dtype=narwhals_to_native_dtype(dtype)))
        return PolarsExpr(pl.lit(value))

    def mean(self, *column_names: str) -> Any:
        pl = get_polars()
        if self._backend_version < (0, 20, 4):  # pragma: no cover
            return pl.mean([*column_names])
        return pl.mean(*column_names)

    @property
    def selectors(self) -> PolarsSelectors:
        return PolarsSelectors()


class PolarsSelectors:
    def by_dtype(self, dtypes: Iterable[dtypes.DType]) -> PolarsExpr:
        from narwhals._polars.expr import PolarsExpr

        pl = get_polars()
        return PolarsExpr(
            pl.selectors.by_dtype([narwhals_to_native_dtype(dtype) for dtype in dtypes])
        )

    def numeric(self) -> PolarsExpr:
        from narwhals._polars.expr import PolarsExpr

        pl = get_polars()
        return PolarsExpr(pl.selectors.numeric())

    def boolean(self) -> PolarsExpr:
        from narwhals._polars.expr import PolarsExpr

        pl = get_polars()
        return PolarsExpr(pl.selectors.boolean())

    def string(self) -> PolarsExpr:
        from narwhals._polars.expr import PolarsExpr

        pl = get_polars()
        return PolarsExpr(pl.selectors.string())

    def categorical(self) -> PolarsExpr:
        from narwhals._polars.expr import PolarsExpr

        pl = get_polars()
        return PolarsExpr(pl.selectors.categorical())

    def all(self) -> PolarsExpr:
        from narwhals._polars.expr import PolarsExpr

        pl = get_polars()
        return PolarsExpr(pl.selectors.all())
