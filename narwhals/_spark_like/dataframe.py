from __future__ import annotations

import warnings
from functools import reduce
from operator import and_
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Mapping
from typing import Sequence

from narwhals._spark_like.utils import evaluate_exprs
from narwhals._spark_like.utils import import_functions
from narwhals._spark_like.utils import import_native_dtypes
from narwhals._spark_like.utils import import_window
from narwhals._spark_like.utils import native_to_narwhals_dtype
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import CompliantDataFrame
from narwhals.typing import CompliantLazyFrame
from narwhals.utils import Implementation
from narwhals.utils import check_column_exists
from narwhals.utils import find_stacklevel
from narwhals.utils import generate_temporary_column_name
from narwhals.utils import import_dtypes_module
from narwhals.utils import is_spark_like_dataframe
from narwhals.utils import not_implemented
from narwhals.utils import parse_columns_to_drop
from narwhals.utils import parse_version
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from types import ModuleType

    import pyarrow as pa
    from sqlframe.base.column import Column
    from sqlframe.base.dataframe import BaseDataFrame
    from sqlframe.base.window import Window
    from typing_extensions import Self
    from typing_extensions import TypeAlias
    from typing_extensions import TypeIs

    from narwhals._spark_like.expr import SparkLikeExpr
    from narwhals._spark_like.group_by import SparkLikeLazyGroupBy
    from narwhals._spark_like.namespace import SparkLikeNamespace
    from narwhals.dtypes import DType
    from narwhals.typing import JoinStrategy
    from narwhals.typing import LazyUniqueKeepStrategy
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

    SQLFrameDataFrame = BaseDataFrame[Any, Any, Any, Any, Any]

Incomplete: TypeAlias = Any  # pragma: no cover
"""Marker for working code that fails type checking."""


class SparkLikeLazyFrame(CompliantLazyFrame["SparkLikeExpr", "SQLFrameDataFrame"]):
    def __init__(
        self,
        native_dataframe: SQLFrameDataFrame,
        *,
        backend_version: tuple[int, ...],
        version: Version,
        implementation: Implementation,
    ) -> None:
        self._native_frame: SQLFrameDataFrame = native_dataframe
        self._backend_version = backend_version
        self._implementation = implementation
        self._version = version
        self._cached_schema: dict[str, DType] | None = None
        validate_backend_version(self._implementation, self._backend_version)

    @property
    def _F(self):  # type: ignore[no-untyped-def] # noqa: ANN202, N802
        if TYPE_CHECKING:
            from sqlframe.base import functions

            return functions
        else:
            return import_functions(self._implementation)

    @property
    def _native_dtypes(self):  # type: ignore[no-untyped-def] # noqa: ANN202
        if TYPE_CHECKING:
            from sqlframe.base import types

            return types
        else:
            return import_native_dtypes(self._implementation)

    @property
    def _Window(self) -> type[Window]:  # noqa: N802
        if TYPE_CHECKING:
            from sqlframe.base.window import Window

            return Window
        else:
            return import_window(self._implementation)

    @staticmethod
    def _is_native(obj: SQLFrameDataFrame | Any) -> TypeIs[SQLFrameDataFrame]:
        return is_spark_like_dataframe(obj)

    @classmethod
    def from_native(cls, data: SQLFrameDataFrame, /, *, context: _FullContext) -> Self:
        return cls(
            data,
            backend_version=context._backend_version,
            version=context._version,
            implementation=context._implementation,
        )

    def __native_namespace__(self) -> ModuleType:  # pragma: no cover
        return self._implementation.to_native_namespace()

    def __narwhals_namespace__(self) -> SparkLikeNamespace:
        from narwhals._spark_like.namespace import SparkLikeNamespace

        return SparkLikeNamespace(
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def _with_version(self, version: Version) -> Self:
        return self.__class__(
            self.native,
            backend_version=self._backend_version,
            version=version,
            implementation=self._implementation,
        )

    def _with_native(self, df: SQLFrameDataFrame) -> Self:
        return self.__class__(
            df,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def _collect_to_arrow(self) -> pa.Table:
        if self._implementation is Implementation.PYSPARK and self._backend_version < (
            4,
        ):
            import pyarrow as pa  # ignore-banned-import

            try:
                return pa.Table.from_batches(self.native._collect_as_arrow())
            except ValueError as exc:
                if "at least one RecordBatch" in str(exc):
                    # Empty dataframe
                    from narwhals._arrow.utils import narwhals_to_native_dtype

                    data: dict[str, list[Any]] = {}
                    schema: list[tuple[str, pa.DataType]] = []
                    current_schema = self.collect_schema()
                    for key, value in current_schema.items():
                        data[key] = []
                        try:
                            native_dtype = narwhals_to_native_dtype(value, self._version)
                        except Exception as exc:  # noqa: BLE001
                            native_spark_dtype = self.native.schema[key].dataType  # type: ignore[index]
                            # If we can't convert the type, just set it to `pa.null`, and warn.
                            # Avoid the warning if we're starting from PySpark's void type.
                            # We can avoid the check when we introduce `nw.Null` dtype.
                            null_type = self._native_dtypes.NullType  # pyright: ignore[reportAttributeAccessIssue]
                            if not isinstance(native_spark_dtype, null_type):
                                warnings.warn(
                                    f"Could not convert dtype {native_spark_dtype} to PyArrow dtype, {exc!r}",
                                    stacklevel=find_stacklevel(),
                                )
                            schema.append((key, pa.null()))
                        else:
                            schema.append((key, native_dtype))
                    return pa.Table.from_pydict(data, schema=pa.schema(schema))
                else:  # pragma: no cover
                    raise
        else:
            return self.native.toArrow()

    def _iter_columns(self) -> Iterator[Column]:
        for col in self.columns:
            yield self._F.col(col)

    @property
    def columns(self) -> list[str]:
        return list(self.schema) if self._cached_schema else list(self.native.columns)

    def collect(
        self,
        backend: ModuleType | Implementation | str | None,
        **kwargs: Any,
    ) -> CompliantDataFrame[Any, Any, Any]:
        if backend is Implementation.PANDAS:
            import pandas as pd  # ignore-banned-import

            from narwhals._pandas_like.dataframe import PandasLikeDataFrame

            return PandasLikeDataFrame(
                self.native.toPandas(),
                implementation=Implementation.PANDAS,
                backend_version=parse_version(pd),
                version=self._version,
                validate_column_names=True,
            )

        elif backend is None or backend is Implementation.PYARROW:
            import pyarrow as pa  # ignore-banned-import

            from narwhals._arrow.dataframe import ArrowDataFrame

            return ArrowDataFrame(
                self._collect_to_arrow(),
                backend_version=parse_version(pa),
                version=self._version,
                validate_column_names=True,
            )

        elif backend is Implementation.POLARS:
            import polars as pl  # ignore-banned-import
            import pyarrow as pa  # ignore-banned-import

            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(
                pl.from_arrow(self._collect_to_arrow()),  # type: ignore[arg-type]
                backend_version=parse_version(pl),
                version=self._version,
            )

        msg = f"Unsupported `backend` value: {backend}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def simple_select(self, *column_names: str) -> Self:
        return self._with_native(self.native.select(*column_names))

    def aggregate(
        self,
        *exprs: SparkLikeExpr,
    ) -> Self:
        new_columns = evaluate_exprs(self, *exprs)

        new_columns_list = [col.alias(col_name) for col_name, col in new_columns]
        return self._with_native(self.native.agg(*new_columns_list))

    def select(
        self,
        *exprs: SparkLikeExpr,
    ) -> Self:
        new_columns = evaluate_exprs(self, *exprs)
        new_columns_list = [col.alias(col_name) for (col_name, col) in new_columns]
        return self._with_native(self.native.select(*new_columns_list))

    def with_columns(self, *exprs: SparkLikeExpr) -> Self:
        new_columns = evaluate_exprs(self, *exprs)
        return self._with_native(self.native.withColumns(dict(new_columns)))

    def filter(self, predicate: SparkLikeExpr) -> Self:
        # `[0]` is safe as the predicate's expression only returns a single column
        condition = predicate._call(self)[0]
        spark_df = self.native.where(condition)
        return self._with_native(spark_df)

    @property
    def schema(self) -> dict[str, DType]:
        if self._cached_schema is None:
            self._cached_schema = {
                field.name: native_to_narwhals_dtype(
                    dtype=field.dataType,
                    version=self._version,
                    spark_types=self._native_dtypes,
                )
                for field in self.native.schema
            }
        return self._cached_schema

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    def drop(self, columns: Sequence[str], *, strict: bool) -> Self:
        columns_to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )
        return self._with_native(self.native.drop(*columns_to_drop))

    def head(self, n: int) -> Self:
        return self._with_native(self.native.limit(num=n))

    def group_by(
        self, keys: Sequence[str] | Sequence[SparkLikeExpr], *, drop_null_keys: bool
    ) -> SparkLikeLazyGroupBy:
        from narwhals._spark_like.group_by import SparkLikeLazyGroupBy

        return SparkLikeLazyGroupBy(self, keys, drop_null_keys=drop_null_keys)

    def sort(
        self,
        *by: str,
        descending: bool | Sequence[bool],
        nulls_last: bool,
    ) -> Self:
        if isinstance(descending, bool):
            descending = [descending] * len(by)

        if nulls_last:
            sort_funcs = (
                self._F.desc_nulls_last if d else self._F.asc_nulls_last
                for d in descending
            )
        else:
            sort_funcs = (
                self._F.desc_nulls_first if d else self._F.asc_nulls_first
                for d in descending
            )

        sort_cols = [sort_f(col) for col, sort_f in zip(by, sort_funcs)]
        return self._with_native(self.native.sort(*sort_cols))

    def drop_nulls(self, subset: Sequence[str] | None) -> Self:
        subset = list(subset) if subset else None
        return self._with_native(self.native.dropna(subset=subset))

    def rename(self, mapping: Mapping[str, str]) -> Self:
        rename_mapping = {
            colname: mapping.get(colname, colname) for colname in self.columns
        }
        return self._with_native(
            self.native.select(
                [self._F.col(old).alias(new) for old, new in rename_mapping.items()]
            )
        )

    def unique(
        self, subset: Sequence[str] | None, *, keep: LazyUniqueKeepStrategy
    ) -> Self:
        check_column_exists(self.columns, subset)
        subset = list(subset) if subset else None
        if keep == "none":
            tmp = generate_temporary_column_name(8, self.columns)
            window = self._Window().partitionBy(subset or self.columns)
            df = (
                self.native.withColumn(tmp, self._F.count("*").over(window))
                .filter(self._F.col(tmp) == 1)
                .drop(tmp)
            )
            return self._with_native(df)
        return self._with_native(self.native.dropDuplicates(subset=subset))

    def join(
        self,
        other: Self,
        how: JoinStrategy,
        left_on: Sequence[str] | None,
        right_on: Sequence[str] | None,
        suffix: str,
    ) -> Self:
        left_columns = self.columns
        right_columns = other.columns

        right_on_: list[str] = list(right_on) if right_on is not None else []
        left_on_: list[str] = list(left_on) if left_on is not None else []

        # create a mapping for columns on other
        # `right_on` columns will be renamed as `left_on`
        # the remaining columns will be either added the suffix or left unchanged.
        right_cols_to_rename = (
            [c for c in right_columns if c not in right_on_]
            if how != "full"
            else right_columns
        )

        rename_mapping = {
            **dict(zip(right_on_, left_on_)),
            **{
                colname: f"{colname}{suffix}" if colname in left_columns else colname
                for colname in right_cols_to_rename
            },
        }
        other_native = other.native.select(
            [self._F.col(old).alias(new) for old, new in rename_mapping.items()]
        )

        # If how in {"semi", "anti"}, then resulting columns are same as left columns
        # Otherwise, we add the right columns with the new mapping, while keeping the
        # original order of right_columns.
        col_order = left_columns

        if how in {"inner", "left", "cross"}:
            col_order.extend(
                rename_mapping[colname]
                for colname in right_columns
                if colname not in right_on_
            )
        elif how == "full":
            col_order.extend(rename_mapping.values())

        right_on_remapped = [rename_mapping[c] for c in right_on_]
        on_ = (
            reduce(
                and_,
                (
                    getattr(self.native, left_key) == getattr(other_native, right_key)
                    for left_key, right_key in zip(left_on_, right_on_remapped)
                ),
            )
            if how == "full"
            else None
            if how == "cross"
            else left_on_
        )
        how_native = "full_outer" if how == "full" else how

        return self._with_native(
            self.native.join(other_native, on=on_, how=how_native).select(col_order)
        )

    def explode(self, columns: Sequence[str]) -> Self:
        dtypes = import_dtypes_module(self._version)

        schema = self.collect_schema()
        for col_to_explode in columns:
            dtype = schema[col_to_explode]

            if dtype != dtypes.List:
                msg = (
                    f"`explode` operation not supported for dtype `{dtype}`, "
                    "expected List type"
                )
                raise InvalidOperationError(msg)

        column_names = self.columns

        if len(columns) != 1:
            msg = (
                "Exploding on multiple columns is not supported with SparkLike backend since "
                "we cannot guarantee that the exploded columns have matching element counts."
            )
            raise NotImplementedError(msg)

        if self._implementation.is_pyspark():
            return self._with_native(
                self.native.select(
                    *[
                        self._F.col(col_name).alias(col_name)
                        if col_name != columns[0]
                        else self._F.explode_outer(col_name).alias(col_name)
                        for col_name in column_names
                    ]
                )
            )
        elif self._implementation.is_sqlframe():
            # Not every sqlframe dialect supports `explode_outer` function
            # (see https://github.com/eakmanrq/sqlframe/blob/3cb899c515b101ff4c197d84b34fae490d0ed257/sqlframe/base/functions.py#L2288-L2289)
            # therefore we simply explode the array column which will ignore nulls and
            # zero sized arrays, and append these specific condition with nulls (to
            # match polars behavior).

            def null_condition(col_name: str) -> Column:
                return self._F.isnull(col_name) | (self._F.array_size(col_name) == 0)

            return self._with_native(
                self.native.select(
                    *[
                        self._F.col(col_name).alias(col_name)
                        if col_name != columns[0]
                        else self._F.explode(col_name).alias(col_name)
                        for col_name in column_names
                    ]
                ).union(
                    self.native.filter(null_condition(columns[0])).select(
                        *[
                            self._F.col(col_name).alias(col_name)
                            if col_name != columns[0]
                            else self._F.lit(None).alias(col_name)
                            for col_name in column_names
                        ]
                    )
                ),
            )
        else:  # pragma: no cover
            msg = "Unreachable code, please report an issue at https://github.com/narwhals-dev/narwhals/issues"
            raise AssertionError(msg)

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        if self._implementation.is_sqlframe():
            if variable_name == "":
                msg = "`variable_name` cannot be empty string for sqlframe backend."
                raise NotImplementedError(msg)

            if value_name == "":
                msg = "`value_name` cannot be empty string for sqlframe backend."
                raise NotImplementedError(msg)
        else:  # pragma: no cover
            pass

        ids = tuple(index) if index else ()
        values = (
            tuple(set(self.columns).difference(set(ids))) if on is None else tuple(on)
        )
        unpivoted_native_frame = self.native.unpivot(
            ids=ids,
            values=values,
            variableColumnName=variable_name,
            valueColumnName=value_name,
        )
        if index is None:
            unpivoted_native_frame = unpivoted_native_frame.drop(*ids)
        return self._with_native(unpivoted_native_frame)

    gather_every = not_implemented.deprecated(
        "`LazyFrame.gather_every` is deprecated and will be removed in a future version."
    )
    join_asof = not_implemented()
    tail = not_implemented.deprecated(
        "`LazyFrame.tail` is deprecated and will be removed in a future version."
    )
    with_row_index = not_implemented()
