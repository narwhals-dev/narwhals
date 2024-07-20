from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable

from narwhals import dtypes
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import reverse_translate_dtype
from narwhals.dependencies import get_polars

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

    def __getattr__(self, attr: str) -> Any:
        from narwhals._polars.expr import PolarsExpr

        pl = get_polars()

        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return PolarsExpr(getattr(pl, attr)(*args, **kwargs))

        return func

    def concat(
        self,
        items: Iterable[PolarsDataFrame | PolarsLazyFrame],
        *,
        how: str = "vertical",
    ) -> PolarsDataFrame | PolarsLazyFrame:
        from narwhals._polars.dataframe import PolarsDataFrame
        from narwhals._polars.dataframe import PolarsLazyFrame

        pl = get_polars()
        dfs: list[Any] = [item._native_dataframe for item in items]
        result = pl.concat(dfs, how=how)
        if isinstance(result, pl.DataFrame):
            return PolarsDataFrame(result)
        return PolarsLazyFrame(result)

    def lit(self, value: Any, dtype: dtypes.DType) -> PolarsExpr:
        from narwhals._polars.expr import PolarsExpr

        pl = get_polars()
        return PolarsExpr(pl.lit(value, dtype=reverse_translate_dtype(dtype)))
