from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence

from narwhals._pyspark.utils import parse_exprs_and_named_exprs
from narwhals._pyspark.utils import translate_sql_api_dtype
from narwhals.utils import Implementation
from narwhals.utils import flatten
from narwhals.utils import parse_columns_to_drop
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from typing_extensions import Self

    from narwhals._pyspark.expr import PySparkExpr
    from narwhals._pyspark.group_by import PySparkLazyGroupBy
    from narwhals._pyspark.namespace import PySparkNamespace
    from narwhals._pyspark.typing import IntoPySparkExpr
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes


class PySparkLazyFrame:
    def __init__(
        self,
        native_dataframe: DataFrame,
        *,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> None:
        self._native_frame = native_dataframe
        self._backend_version = backend_version
        self._implementation = Implementation.PYSPARK
        self._dtypes = dtypes

    def __native_namespace__(self) -> Any:  # pragma: no cover
        if self._implementation is Implementation.PYSPARK:
            return self._implementation.to_native_namespace()

        msg = f"Expected pyspark, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def __narwhals_namespace__(self) -> PySparkNamespace:
        from narwhals._pyspark.namespace import PySparkNamespace

        return PySparkNamespace(
            backend_version=self._backend_version, dtypes=self._dtypes
        )

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def _from_native_frame(self, df: DataFrame) -> Self:
        return self.__class__(
            df, backend_version=self._backend_version, dtypes=self._dtypes
        )

    @property
    def columns(self) -> list[str]:
        return self._native_frame.columns  # type: ignore[no-any-return]

    def collect(self) -> Any:
        import pandas as pd  # ignore-banned-import()

        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        return PandasLikeDataFrame(
            native_dataframe=self._native_frame.toPandas(),
            implementation=Implementation.PANDAS,
            backend_version=parse_version(pd.__version__),
            dtypes=self._dtypes,
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
            from pyspark.sql.types import StructType

            spark_session = self._native_frame.sparkSession
            spark_df = spark_session.createDataFrame([], StructType([]))

            return self._from_native_frame(spark_df)

        new_columns_list = [col.alias(col_name) for col_name, col in new_columns.items()]
        return self._from_native_frame(self._native_frame.select(*new_columns_list))

    def filter(self, *predicates: PySparkExpr) -> Self:
        from narwhals._pyspark.namespace import PySparkNamespace

        if (
            len(predicates) == 1
            and isinstance(predicates[0], list)
            and all(isinstance(x, bool) for x in predicates[0])
        ):
            msg = "`LazyFrame.filter` is not supported for PySpark backend with boolean masks."
            raise NotImplementedError(msg)
        plx = PySparkNamespace(backend_version=self._backend_version, dtypes=self._dtypes)
        expr = plx.all_horizontal(*predicates)
        # Safety: all_horizontal's expression only returns a single column.
        condition = expr._call(self)[0]
        spark_df = self._native_frame.where(condition)
        return self._from_native_frame(spark_df)

    @property
    def schema(self) -> dict[str, DType]:
        return {
            field.name: translate_sql_api_dtype(field.dataType)
            for field in self._native_frame.schema
        }

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    def with_columns(
        self: Self,
        *exprs: IntoPySparkExpr,
        **named_exprs: IntoPySparkExpr,
    ) -> Self:
        new_columns_map = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)
        return self._from_native_frame(self._native_frame.withColumns(new_columns_map))

    def drop(self: Self, columns: list[str], strict: bool) -> Self:  # noqa: FBT001
        columns_to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )
        return self._from_native_frame(self._native_frame.drop(*columns_to_drop))

    def head(self: Self, n: int) -> Self:
        spark_session = self._native_frame.sparkSession

        return self._from_native_frame(
            spark_session.createDataFrame(self._native_frame.take(num=n))
        )

    def group_by(self: Self, *by: str) -> PySparkLazyGroupBy:
        from narwhals._pyspark.group_by import PySparkLazyGroupBy

        return PySparkLazyGroupBy(df=self, keys=list(by))

    def sort(
        self: Self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = False,
    ) -> Self:
        import pyspark.sql.functions as F  # noqa: N812

        flat_by = flatten([*flatten([by]), *more_by])
        if isinstance(descending, bool):
            descending = [descending]

        if nulls_last:
            sort_funcs = [
                F.desc_nulls_last if d else F.asc_nulls_last for d in descending
            ]
        else:
            sort_funcs = [
                F.desc_nulls_first if d else F.asc_nulls_first for d in descending
            ]

        sort_cols = [sort_f(col) for col, sort_f in zip(flat_by, sort_funcs)]
        return self._from_native_frame(self._native_frame.sort(*sort_cols))
