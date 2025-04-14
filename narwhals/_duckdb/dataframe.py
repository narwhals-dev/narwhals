from __future__ import annotations

import contextlib
from functools import reduce
from operator import and_
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Mapping
from typing import Sequence

import duckdb
from duckdb import FunctionExpression
from duckdb import StarExpression

from narwhals._duckdb.utils import col
from narwhals._duckdb.utils import evaluate_exprs
from narwhals._duckdb.utils import generate_partition_by_sql
from narwhals._duckdb.utils import lit
from narwhals._duckdb.utils import native_to_narwhals_dtype
from narwhals.dependencies import get_duckdb
from narwhals.exceptions import ColumnNotFoundError
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import CompliantDataFrame
from narwhals.typing import CompliantLazyFrame
from narwhals.utils import Implementation
from narwhals.utils import Version
from narwhals.utils import generate_temporary_column_name
from narwhals.utils import import_dtypes_module
from narwhals.utils import not_implemented
from narwhals.utils import parse_columns_to_drop
from narwhals.utils import parse_version
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from types import ModuleType

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self
    from typing_extensions import TypeIs

    from narwhals._duckdb.expr import DuckDBExpr
    from narwhals._duckdb.group_by import DuckDBGroupBy
    from narwhals._duckdb.namespace import DuckDBNamespace
    from narwhals._duckdb.series import DuckDBInterchangeSeries
    from narwhals.dtypes import DType
    from narwhals.typing import AsofJoinStrategy
    from narwhals.typing import JoinStrategy
    from narwhals.typing import LazyUniqueKeepStrategy
    from narwhals.utils import _FullContext

with contextlib.suppress(ImportError):  # requires duckdb>=1.3.0
    from duckdb import SQLExpression  # type: ignore[attr-defined, unused-ignore]


class DuckDBLazyFrame(CompliantLazyFrame["DuckDBExpr", "duckdb.DuckDBPyRelation"]):
    _implementation = Implementation.DUCKDB

    def __init__(
        self: Self,
        df: duckdb.DuckDBPyRelation,
        *,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._native_frame: duckdb.DuckDBPyRelation = df
        self._version = version
        self._backend_version = backend_version
        self._cached_schema: dict[str, DType] | None = None
        validate_backend_version(self._implementation, self._backend_version)

    @staticmethod
    def _is_native(obj: duckdb.DuckDBPyRelation | Any) -> TypeIs[duckdb.DuckDBPyRelation]:
        return isinstance(obj, duckdb.DuckDBPyRelation)

    @classmethod
    def from_native(
        cls, data: duckdb.DuckDBPyRelation, /, *, context: _FullContext
    ) -> Self:
        return cls(
            data, backend_version=context._backend_version, version=context._version
        )

    def __narwhals_dataframe__(self: Self) -> Self:  # pragma: no cover
        # Keep around for backcompat.
        if self._version is not Version.V1:
            msg = "__narwhals_dataframe__ is not implemented for DuckDBLazyFrame"
            raise AttributeError(msg)
        return self

    def __narwhals_lazyframe__(self: Self) -> Self:
        return self

    def __native_namespace__(self: Self) -> ModuleType:
        return get_duckdb()  # type: ignore[no-any-return]

    def __narwhals_namespace__(self: Self) -> DuckDBNamespace:
        from narwhals._duckdb.namespace import DuckDBNamespace

        return DuckDBNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def __getitem__(self: Self, item: str) -> DuckDBInterchangeSeries:
        from narwhals._duckdb.series import DuckDBInterchangeSeries

        return DuckDBInterchangeSeries(self.native.select(item), version=self._version)

    def _iter_columns(self) -> Iterator[duckdb.Expression]:
        for name in self.columns:
            yield col(name)

    def collect(
        self: Self,
        backend: ModuleType | Implementation | str | None,
        **kwargs: Any,
    ) -> CompliantDataFrame[Any, Any, Any]:
        if backend is None or backend is Implementation.PYARROW:
            import pyarrow as pa  # ignore-banned-import

            from narwhals._arrow.dataframe import ArrowDataFrame

            return ArrowDataFrame(
                self.native.arrow(),
                backend_version=parse_version(pa),
                version=self._version,
                validate_column_names=True,
            )

        if backend is Implementation.PANDAS:
            import pandas as pd  # ignore-banned-import

            from narwhals._pandas_like.dataframe import PandasLikeDataFrame

            return PandasLikeDataFrame(
                self.native.df(),
                implementation=Implementation.PANDAS,
                backend_version=parse_version(pd),
                version=self._version,
                validate_column_names=True,
            )

        if backend is Implementation.POLARS:
            import polars as pl  # ignore-banned-import

            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(
                self.native.pl(), backend_version=parse_version(pl), version=self._version
            )

        msg = f"Unsupported `backend` value: {backend}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def head(self: Self, n: int) -> Self:
        return self._with_native(self.native.limit(n))

    def simple_select(self, *column_names: str) -> Self:
        return self._with_native(self.native.select(*column_names))

    def aggregate(self: Self, *exprs: DuckDBExpr) -> Self:
        selection = [val.alias(name) for name, val in evaluate_exprs(self, *exprs)]
        return self._with_native(self.native.aggregate(selection))  # type: ignore[arg-type]

    def select(
        self: Self,
        *exprs: DuckDBExpr,
    ) -> Self:
        selection = (val.alias(name) for name, val in evaluate_exprs(self, *exprs))
        return self._with_native(self.native.select(*selection))

    def drop(self: Self, columns: Sequence[str], *, strict: bool) -> Self:
        columns_to_drop = parse_columns_to_drop(self, columns=columns, strict=strict)
        selection = (name for name in self.columns if name not in columns_to_drop)
        return self._with_native(self.native.select(*selection))

    def lazy(self: Self, *, backend: Implementation | None = None) -> Self:
        # The `backend`` argument has no effect but we keep it here for
        # backwards compatibility because in `narwhals.stable.v1`
        # function `.from_native()` will return a DataFrame for DuckDB.

        if backend is not None:  # pragma: no cover
            msg = "`backend` argument is not supported for DuckDB"
            raise ValueError(msg)
        return self

    def with_columns(self: Self, *exprs: DuckDBExpr) -> Self:
        new_columns_map = dict(evaluate_exprs(self, *exprs))
        result = [
            new_columns_map.pop(name).alias(name)
            if name in new_columns_map
            else col(name)
            for name in self.native.columns
        ]
        result.extend(value.alias(name) for name, value in new_columns_map.items())
        return self._with_native(self.native.select(*result))

    def filter(self: Self, predicate: DuckDBExpr) -> Self:
        # `[0]` is safe as the predicate's expression only returns a single column
        mask = predicate(self)[0]
        return self._with_native(self.native.filter(mask))

    @property
    def schema(self: Self) -> dict[str, DType]:
        if self._cached_schema is None:
            # Note: prefer `self._cached_schema` over `functools.cached_property`
            # due to Python3.13 failures.
            self._cached_schema = {
                column_name: native_to_narwhals_dtype(str(duckdb_dtype), self._version)
                for column_name, duckdb_dtype in zip(
                    self.native.columns, self.native.types
                )
            }
        return self._cached_schema

    @property
    def columns(self: Self) -> list[str]:
        return list(self.schema)

    def to_pandas(self: Self) -> pd.DataFrame:
        # only if version is v1, keep around for backcompat
        import pandas as pd  # ignore-banned-import()

        if parse_version(pd) >= (1, 0, 0):
            return self.native.df()
        else:  # pragma: no cover
            msg = f"Conversion to pandas requires pandas>=1.0.0, found {pd.__version__}"
            raise NotImplementedError(msg)

    def to_arrow(self: Self) -> pa.Table:
        # only if version is v1, keep around for backcompat
        return self.native.arrow()

    def _with_version(self: Self, version: Version) -> Self:
        return self.__class__(
            self.native, version=version, backend_version=self._backend_version
        )

    def _with_native(self: Self, df: duckdb.DuckDBPyRelation) -> Self:
        return self.__class__(
            df, backend_version=self._backend_version, version=self._version
        )

    def group_by(self: Self, *keys: str, drop_null_keys: bool) -> DuckDBGroupBy:
        from narwhals._duckdb.group_by import DuckDBGroupBy

        return DuckDBGroupBy(self, keys, drop_null_keys=drop_null_keys)

    def rename(self: Self, mapping: Mapping[str, str]) -> Self:
        df = self.native
        selection = (
            col(name).alias(mapping[name]) if name in mapping else col(name)
            for name in df.columns
        )
        return self._with_native(self.native.select(*selection))

    def join(
        self: Self,
        other: Self,
        *,
        how: JoinStrategy,
        left_on: Sequence[str] | None,
        right_on: Sequence[str] | None,
        suffix: str,
    ) -> Self:
        native_how = "outer" if how == "full" else how

        if native_how == "cross":
            if self._backend_version < (1, 1, 4):
                msg = f"DuckDB>=1.1.4 is required for cross-join, found version: {self._backend_version}"
                raise NotImplementedError(msg)
            rel = self.native.set_alias("lhs").cross(other.native.set_alias("rhs"))
        else:
            # help mypy
            assert left_on is not None  # noqa: S101
            assert right_on is not None  # noqa: S101
            it = (
                col(f'lhs."{left}"') == col(f'rhs."{right}"')
                for left, right in zip(left_on, right_on)
            )
            condition: duckdb.Expression = reduce(and_, it)
            rel = self.native.set_alias("lhs").join(
                other.native.set_alias("rhs"),
                # NOTE: Fixed in `--pre` https://github.com/duckdb/duckdb/pull/16933
                condition=condition,  # type: ignore[arg-type, unused-ignore]
                how=native_how,
            )

        if native_how in {"inner", "left", "cross", "outer"}:
            select = [col(f'lhs."{x}"') for x in self.native.columns]
            for name in other.native.columns:
                col_in_lhs: bool = name in self.native.columns
                if native_how == "outer" and not col_in_lhs:
                    select.append(col(f'rhs."{name}"'))
                elif (native_how == "outer") or (
                    col_in_lhs and (right_on is None or name not in right_on)
                ):
                    select.append(col(f'rhs."{name}"').alias(f"{name}{suffix}"))
                elif right_on is None or name not in right_on:
                    select.append(col(name))
            res = rel.select(*select).set_alias(self.native.alias)
        else:  # semi, anti
            res = rel.select("lhs.*").set_alias(self.native.alias)

        return self._with_native(res)

    def join_asof(
        self: Self,
        other: Self,
        *,
        left_on: str | None,
        right_on: str | None,
        by_left: Sequence[str] | None,
        by_right: Sequence[str] | None,
        strategy: AsofJoinStrategy,
        suffix: str,
    ) -> Self:
        lhs = self.native
        rhs = other.native
        conditions: list[duckdb.Expression] = []
        if by_left is not None and by_right is not None:
            conditions.extend(
                col(f'lhs."{left}"') == col(f'rhs."{right}"')
                for left, right in zip(by_left, by_right)
            )
        else:
            by_left = by_right = []
        if strategy == "backward":
            conditions.append(col(f'lhs."{left_on}"') >= col(f'rhs."{right_on}"'))
        elif strategy == "forward":
            conditions.append(col(f'lhs."{left_on}"') <= col(f'rhs."{right_on}"'))
        else:
            msg = "Only 'backward' and 'forward' strategies are currently supported for DuckDB"
            raise NotImplementedError(msg)
        condition: duckdb.Expression = reduce(and_, conditions)
        select = ["lhs.*"]
        for name in rhs.columns:
            if name in lhs.columns and (
                right_on is None or name not in {right_on, *by_right}
            ):
                select.append(f'rhs."{name}" as "{name}{suffix}"')
            elif right_on is None or name not in {right_on, *by_right}:
                select.append(str(col(name)))
        # Replace with Python API call once
        # https://github.com/duckdb/duckdb/discussions/16947 is addressed.
        query = f"""
            SELECT {",".join(select)}
            FROM lhs
            ASOF LEFT JOIN rhs
            ON {condition}
            """  # noqa: S608
        return self._with_native(duckdb.sql(query))

    def collect_schema(self: Self) -> dict[str, DType]:
        return {
            column_name: native_to_narwhals_dtype(str(duckdb_dtype), self._version)
            for column_name, duckdb_dtype in zip(self.native.columns, self.native.types)
        }

    def unique(
        self, subset: Sequence[str] | None, *, keep: LazyUniqueKeepStrategy
    ) -> Self:
        if subset_ := subset if keep == "any" else (subset or self.columns):
            if self._backend_version < (1, 3):
                msg = (
                    "At least version 1.3 of DuckDB is required for `unique` operation\n"
                    "with `subset` specified."
                )
                raise NotImplementedError(msg)
            # Sanitise input
            if any(x not in self.columns for x in subset_):
                msg = f"Columns {set(subset_).difference(self.columns)} not found in {self.columns}."
                raise ColumnNotFoundError(msg)
            idx_name = generate_temporary_column_name(8, self.columns)
            count_name = generate_temporary_column_name(8, [*self.columns, idx_name])
            partition_by_sql = generate_partition_by_sql(*(subset_))
            name = count_name if keep == "none" else idx_name
            idx_expr = SQLExpression(
                f"{FunctionExpression('row_number')} over ({partition_by_sql})"
            ).alias(idx_name)
            count_expr = SQLExpression(
                f"{FunctionExpression('count', StarExpression())} over ({partition_by_sql})"
            ).alias(count_name)
            return self._with_native(
                self.native.select(StarExpression(), idx_expr, count_expr)
                .filter(col(name) == lit(1))
                .select(StarExpression(exclude=[count_name, idx_name]))
            )
        return self._with_native(self.native.unique(", ".join(self.columns)))

    def sort(
        self: Self, *by: str, descending: bool | Sequence[bool], nulls_last: bool
    ) -> Self:
        if isinstance(descending, bool):
            descending = [descending] * len(by)
        if nulls_last:
            it = (
                col(name).nulls_last() if not desc else col(name).desc().nulls_last()
                for name, desc in zip(by, descending)
            )
        else:
            it = (
                col(name).nulls_first() if not desc else col(name).desc().nulls_first()
                for name, desc in zip(by, descending)
            )
        return self._with_native(self.native.sort(*it))

    def drop_nulls(self: Self, subset: Sequence[str] | None) -> Self:
        subset_ = subset if subset is not None else self.columns
        keep_condition = reduce(and_, (col(name).isnotnull() for name in subset_))
        return self._with_native(self.native.filter(keep_condition))

    def explode(self: Self, columns: Sequence[str]) -> Self:
        dtypes = import_dtypes_module(self._version)
        schema = self.collect_schema()
        for name in columns:
            dtype = schema[name]
            if dtype != dtypes.List:
                msg = (
                    f"`explode` operation not supported for dtype `{dtype}`, "
                    "expected List type"
                )
                raise InvalidOperationError(msg)

        if len(columns) != 1:
            msg = (
                "Exploding on multiple columns is not supported with DuckDB backend since "
                "we cannot guarantee that the exploded columns have matching element counts."
            )
            raise NotImplementedError(msg)

        col_to_explode = col(columns[0])
        rel = self.native
        original_columns = self.columns

        not_null_condition = col_to_explode.isnotnull() & FunctionExpression(
            "len", col_to_explode
        ) > lit(0)
        non_null_rel = rel.filter(not_null_condition).select(
            *(
                FunctionExpression("unnest", col_to_explode).alias(name)
                if name in columns
                else name
                for name in original_columns
            )
        )

        null_rel = rel.filter(~not_null_condition).select(
            *(
                lit(None).alias(name) if name in columns else name
                for name in original_columns
            )
        )

        return self._with_native(non_null_rel.union(null_rel))

    def unpivot(
        self: Self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        index_ = [] if index is None else index
        on_ = [c for c in self.columns if c not in index_] if on is None else on

        if variable_name == "":
            msg = "`variable_name` cannot be empty string for duckdb backend."
            raise NotImplementedError(msg)

        if value_name == "":
            msg = "`value_name` cannot be empty string for duckdb backend."
            raise NotImplementedError(msg)

        unpivot_on = ", ".join(str(col(name)) for name in on_)
        rel = self.native  # noqa: F841
        # Replace with Python API once
        # https://github.com/duckdb/duckdb/discussions/16980 is addressed.
        query = f"""
            unpivot rel
            on {unpivot_on}
            into
                name "{variable_name}"
                value "{value_name}"
            """
        return self._with_native(
            duckdb.sql(query).select(*[*index_, variable_name, value_name])
        )

    gather_every = not_implemented.deprecated(
        "`LazyFrame.gather_every` is deprecated and will be removed in a future version."
    )
    tail = not_implemented.deprecated(
        "`LazyFrame.tail` is deprecated and will be removed in a future version."
    )
    with_row_index = not_implemented()
