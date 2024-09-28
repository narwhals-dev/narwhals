from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence

from narwhals import dtypes
from narwhals._expression_parsing import parse_into_exprs
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import narwhals_to_native_dtype
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame
    from narwhals._polars.expr import PolarsExpr
    from narwhals._polars.typing import IntoPolarsExpr


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
    List = dtypes.List
    Struct = dtypes.Struct
    Array = dtypes.Array

    def __init__(self, *, backend_version: tuple[int, ...]) -> None:
        self._backend_version = backend_version
        self._implementation = Implementation.POLARS

    def __getattr__(self, attr: str) -> Any:
        import polars as pl  # ignore-banned-import

        from narwhals._polars.expr import PolarsExpr

        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return PolarsExpr(getattr(pl, attr)(*args, **kwargs))

        return func

    def nth(self, *indices: int) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        if self._backend_version < (1, 0, 0):  # pragma: no cover
            msg = "`nth` is only supported for Polars>=1.0.0. Please use `col` for columns selection instead."
            raise AttributeError(msg)
        return PolarsExpr(pl.nth(*indices))

    def len(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        if self._backend_version < (0, 20, 5):  # pragma: no cover
            return PolarsExpr(pl.count().alias("len"))
        return PolarsExpr(pl.len())

    def concat(
        self,
        items: Sequence[PolarsDataFrame | PolarsLazyFrame],
        *,
        how: Literal["vertical", "horizontal"],
    ) -> PolarsDataFrame | PolarsLazyFrame:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.dataframe import PolarsDataFrame
        from narwhals._polars.dataframe import PolarsLazyFrame

        dfs: list[Any] = [item._native_frame for item in items]
        result = pl.concat(dfs, how=how)
        if isinstance(result, pl.DataFrame):
            return PolarsDataFrame(result, backend_version=items[0]._backend_version)
        return PolarsLazyFrame(result, backend_version=items[0]._backend_version)

    def lit(self, value: Any, dtype: dtypes.DType | None = None) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        if dtype is not None:
            return PolarsExpr(pl.lit(value, dtype=narwhals_to_native_dtype(dtype)))
        return PolarsExpr(pl.lit(value))

    def mean(self, *column_names: str) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        if self._backend_version < (0, 20, 4):  # pragma: no cover
            return PolarsExpr(pl.mean([*column_names]))  # type: ignore[arg-type]
        return PolarsExpr(pl.mean(*column_names))

    def mean_horizontal(self, *exprs: IntoPolarsExpr) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        polars_exprs = parse_into_exprs(*exprs, namespace=self)

        if self._backend_version < (0, 20, 8):  # pragma: no cover
            total = reduce(lambda x, y: x + y, (e.fill_null(0.0) for e in polars_exprs))
            n_non_zero = reduce(
                lambda x, y: x + y, ((1 - e.is_null()) for e in polars_exprs)
            )
            return PolarsExpr(total._native_expr / n_non_zero._native_expr)

        return PolarsExpr(pl.mean_horizontal([e._native_expr for e in polars_exprs]))

    @property
    def selectors(self) -> PolarsSelectors:
        return PolarsSelectors()


class PolarsSelectors:
    def by_dtype(self, dtypes: Iterable[dtypes.DType]) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(
            pl.selectors.by_dtype([narwhals_to_native_dtype(dtype) for dtype in dtypes])
        )

    def numeric(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(pl.selectors.numeric())

    def boolean(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(pl.selectors.boolean())

    def string(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(pl.selectors.string())

    def categorical(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(pl.selectors.categorical())

    def all(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(pl.selectors.all())
