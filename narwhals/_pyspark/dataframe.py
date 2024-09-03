from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._pandas_like.utils import translate_dtype
from narwhals._pyspark.utils import parse_exprs_and_named_exprs
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_pyspark_sql
from narwhals.utils import Implementation
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from typing_extensions import Self

    from narwhals._pyspark.expr import PySparkExpr
    from narwhals._pyspark.namespace import PySparkNamespace
    from narwhals._pyspark.typing import IntoPySparkExpr
    from narwhals.dtypes import DType


class PySparkLazyFrame:
    def __init__(self, native_dataframe: DataFrame) -> None:
        self._native_frame = native_dataframe
        self._implementation = Implementation.PYSPARK

    def __native_namespace__(self) -> Any:  # pragma: no cover
        return get_pyspark_sql()

    def __narwhals_namespace__(self) -> PySparkNamespace:
        from narwhals._pyspark.namespace import PySparkNamespace

        return PySparkNamespace()

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def _from_native_frame(self, df: DataFrame) -> Self:
        return self.__class__(df)

    def lazy(self) -> Self:
        return self

    @property
    def columns(self) -> list[str]:
        return self._native_frame.columns  # type: ignore[no-any-return]

    def collect(self) -> Any:
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        return PandasLikeDataFrame(
            native_dataframe=self._native_frame.toPandas(),
            implementation=Implementation.PANDAS,
            backend_version=parse_version(get_pandas().__version__),
        )

    def select(
        self: Self,
        *exprs: IntoPySparkExpr,
        **named_exprs: IntoPySparkExpr,
    ) -> Self:
        if exprs and all(isinstance(x, str) for x in exprs) and not named_exprs:
            # This is a simple select
            return self._from_native_frame(self._native_frame.select(*exprs))

        new_columns = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)

        if not new_columns:
            # return empty dataframe, like Polars does
            import pyspark.pandas as ps

            return self._from_native_frame(ps.DataFrame().to_spark())

        return self._from_native_frame(self._native_frame.select(*new_columns))

    def filter(self, *predicates: PySparkExpr) -> Self:
        from narwhals._pyspark.namespace import PySparkNamespace

        if (
            len(predicates) == 1
            and isinstance(predicates[0], list)
            and all(isinstance(x, bool) for x in predicates[0])
        ):
            msg = "Filtering by a list of booleans is not supported."
            raise ValueError(msg)
        plx = PySparkNamespace()
        expr = plx.all_horizontal(*predicates)
        # Safety: all_horizontal's expression only returns a single column.
        condition = expr._call(self)[0]
        spark_df = self._native_frame.where(condition)
        return self._from_native_frame(spark_df)

    @property
    def schema(self) -> dict[str, DType]:
        return {
            col: translate_dtype(self._native_frame.loc[:, col])
            for col in self._native_frame.columns
        }

    def collect_schema(self) -> dict[str, DType]:
        return self.schema
