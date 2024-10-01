from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence

from narwhals._expression_parsing import parse_into_exprs
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import narwhals_to_native_dtype
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame
    from narwhals._polars.expr import PolarsExpr
    from narwhals._polars.typing import IntoPolarsExpr
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes


class PolarsNamespace:
    def __init__(self, *, backend_version: tuple[int, ...], dtypes: DTypes) -> None:
        self._backend_version = backend_version
        self._implementation = Implementation.POLARS
        self._dtypes = dtypes

    def __getattr__(self, attr: str) -> Any:
        import polars as pl  # ignore-banned-import

        from narwhals._polars.expr import PolarsExpr

        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return PolarsExpr(getattr(pl, attr)(*args, **kwargs), dtypes=self._dtypes)

        return func

    def nth(self, *indices: int) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        if self._backend_version < (1, 0, 0):  # pragma: no cover
            msg = "`nth` is only supported for Polars>=1.0.0. Please use `col` for columns selection instead."
            raise AttributeError(msg)
        return PolarsExpr(pl.nth(*indices), dtypes=self._dtypes)

    def len(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        if self._backend_version < (0, 20, 5):  # pragma: no cover
            return PolarsExpr(pl.count().alias("len"), dtypes=self._dtypes)
        return PolarsExpr(pl.len(), dtypes=self._dtypes)

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
            return PolarsDataFrame(
                result, backend_version=items[0]._backend_version, dtypes=items[0]._dtypes
            )
        return PolarsLazyFrame(
            result, backend_version=items[0]._backend_version, dtypes=items[0]._dtypes
        )

    def lit(self, value: Any, dtype: DType | None = None) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        if dtype is not None:
            return PolarsExpr(
                pl.lit(value, dtype=narwhals_to_native_dtype(dtype, self._dtypes)),
                dtypes=self._dtypes,
            )
        return PolarsExpr(pl.lit(value), dtypes=self._dtypes)

    def mean(self, *column_names: str) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        if self._backend_version < (0, 20, 4):  # pragma: no cover
            return PolarsExpr(pl.mean([*column_names]), dtypes=self._dtypes)  # type: ignore[arg-type]
        return PolarsExpr(pl.mean(*column_names), dtypes=self._dtypes)

    def mean_horizontal(self, *exprs: IntoPolarsExpr) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        polars_exprs = parse_into_exprs(*exprs, namespace=self)

        if self._backend_version < (0, 20, 8):  # pragma: no cover
            total = reduce(lambda x, y: x + y, (e.fill_null(0.0) for e in polars_exprs))
            n_non_zero = reduce(
                lambda x, y: x + y, ((1 - e.is_null()) for e in polars_exprs)
            )
            return PolarsExpr(
                total._native_expr / n_non_zero._native_expr, dtypes=self._dtypes
            )

        return PolarsExpr(
            pl.mean_horizontal([e._native_expr for e in polars_exprs]),
            dtypes=self._dtypes,
        )

    @property
    def selectors(self) -> PolarsSelectors:
        return PolarsSelectors(self._dtypes)


class PolarsSelectors:
    def __init__(self, dtypes: DTypes) -> None:
        self._dtypes = dtypes

    def by_dtype(self, dtypes: Iterable[DType]) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(
            pl.selectors.by_dtype(
                [narwhals_to_native_dtype(dtype, self._dtypes) for dtype in dtypes]
            ),
            dtypes=self._dtypes,
        )

    def numeric(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(pl.selectors.numeric(), dtypes=self._dtypes)

    def boolean(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(pl.selectors.boolean(), dtypes=self._dtypes)

    def string(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(pl.selectors.string(), dtypes=self._dtypes)

    def categorical(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(pl.selectors.categorical(), dtypes=self._dtypes)

    def all(self) -> PolarsExpr:
        import polars as pl  # ignore-banned-import()

        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(pl.selectors.all(), dtypes=self._dtypes)
