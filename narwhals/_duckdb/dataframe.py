from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Sequence

import duckdb
from duckdb import ColumnExpression

from narwhals._duckdb.utils import ExprKind
from narwhals._duckdb.utils import native_to_narwhals_dtype
from narwhals._duckdb.utils import parse_exprs_and_named_exprs
from narwhals.dependencies import get_duckdb
from narwhals.exceptions import ColumnNotFoundError
from narwhals.utils import Implementation
from narwhals.utils import Version
from narwhals.utils import generate_temporary_column_name
from narwhals.utils import parse_columns_to_drop
from narwhals.utils import parse_version
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from types import ModuleType

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._duckdb.expr import DuckDBExpr
    from narwhals._duckdb.group_by import DuckDBGroupBy
    from narwhals._duckdb.namespace import DuckDBNamespace
    from narwhals._duckdb.series import DuckDBInterchangeSeries
    from narwhals.dtypes import DType

from narwhals.typing import CompliantLazyFrame


class DuckDBLazyFrame(CompliantLazyFrame):
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
        validate_backend_version(self._implementation, self._backend_version)

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

        return DuckDBInterchangeSeries(
            self._native_frame.select(item), version=self._version
        )

    def collect(self: Self) -> pa.Table:
        try:
            import pyarrow as pa  # ignore-banned-import
        except ModuleNotFoundError as exc:  # pragma: no cover
            msg = "PyArrow>=11.0.0 is required to collect `LazyFrame` backed by DuckDcollect `LazyFrame` backed by DuckDB"
            raise ModuleNotFoundError(msg) from exc

        from narwhals._arrow.dataframe import ArrowDataFrame

        return ArrowDataFrame(
            native_dataframe=self._native_frame.arrow(),
            backend_version=parse_version(pa.__version__),
            version=self._version,
        )

    def head(self: Self, n: int) -> Self:
        return self._from_native_frame(self._native_frame.limit(n))

    def simple_select(self, *column_names: str) -> Self:
        return self._from_native_frame(self._native_frame.select(*column_names))

    def select(
        self: Self,
        *exprs: DuckDBExpr,
        **named_exprs: DuckDBExpr,
    ) -> Self:
        new_columns_map = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)
        if not new_columns_map:
            # TODO(marco): return empty relation with 0 columns?
            return self._from_native_frame(self._native_frame.limit(0))

        if not any(expr._expr_kind is ExprKind.TRANSFORM for expr in exprs) and not any(
            expr._expr_kind is ExprKind.TRANSFORM for expr in named_exprs.values()
        ):
            return self._from_native_frame(
                self._native_frame.aggregate(
                    [val.alias(col) for col, val in new_columns_map.items()]
                )
            )
        if any(expr._expr_kind is ExprKind.AGGREGATION for expr in exprs) or any(
            expr._expr_kind is ExprKind.AGGREGATION for expr in named_exprs.values()
        ):
            msg = (
                "Mixing expressions which aggregate and expressions which don't\n"
                "is not yet supported by the DuckDB backend. Once they introduce\n"
                "duckdb.WindowExpression to their Python API, we'll be able to\n"
                "support this."
            )
            raise NotImplementedError(msg)

        return self._from_native_frame(
            self._native_frame.select(
                *(val.alias(col) for col, val in new_columns_map.items())
            )
        )

    def drop(self: Self, columns: list[str], strict: bool) -> Self:  # noqa: FBT001
        columns_to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )
        selection = (col for col in self.columns if col not in columns_to_drop)
        return self._from_native_frame(self._native_frame.select(*selection))

    def lazy(self: Self) -> Self:
        return self

    def with_columns(
        self: Self,
        *exprs: DuckDBExpr,
        **named_exprs: DuckDBExpr,
    ) -> Self:
        new_columns_map = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)

        if any(expr._expr_kind is ExprKind.AGGREGATION for expr in exprs) or any(
            expr._expr_kind is ExprKind.AGGREGATION for expr in named_exprs.values()
        ):
            msg = (
                "Mixing expressions which aggregate and expressions which don't\n"
                "is not yet supported by the DuckDB backend. Once they introduce\n"
                "duckdb.WindowExpression to their Python API, we'll be able to\n"
                "support this."
            )
            raise NotImplementedError(msg)

        result = []
        for col in self._native_frame.columns:
            if col in new_columns_map:
                result.append(new_columns_map.pop(col).alias(col))
            else:
                result.append(ColumnExpression(col))
        for col, value in new_columns_map.items():
            result.append(value.alias(col))
        return self._from_native_frame(self._native_frame.select(*result))

    def filter(self: Self, *predicates: DuckDBExpr, **constraints: Any) -> Self:
        plx = self.__narwhals_namespace__()
        expr = plx.all_horizontal(
            *chain(predicates, (plx.col(name) == v for name, v in constraints.items()))
        )
        # `[0]` is safe as all_horizontal's expression only returns a single column
        mask = expr._call(self)[0]
        return self._from_native_frame(self._native_frame.filter(mask))

    @property
    def schema(self: Self) -> dict[str, DType]:
        return {
            column_name: native_to_narwhals_dtype(str(duckdb_dtype), self._version)
            for column_name, duckdb_dtype in zip(
                self._native_frame.columns, self._native_frame.types
            )
        }

    @property
    def columns(self: Self) -> list[str]:
        return self._native_frame.columns  # type: ignore[no-any-return]

    def to_pandas(self: Self) -> pd.DataFrame:
        # only if version is v1, keep around for backcompat
        import pandas as pd  # ignore-banned-import()

        if parse_version(pd.__version__) >= parse_version("1.0.0"):
            return self._native_frame.df()
        else:  # pragma: no cover
            msg = f"Conversion to pandas requires pandas>=1.0.0, found {pd.__version__}"
            raise NotImplementedError(msg)

    def to_arrow(self: Self) -> pa.Table:
        # only if version is v1, keep around for backcompat
        return self._native_frame.arrow()

    def _change_version(self: Self, version: Version) -> Self:
        return self.__class__(
            self._native_frame, version=version, backend_version=self._backend_version
        )

    def _from_native_frame(self: Self, df: duckdb.DuckDBPyRelation) -> Self:
        return self.__class__(
            df, backend_version=self._backend_version, version=self._version
        )

    def group_by(self: Self, *keys: str, drop_null_keys: bool) -> DuckDBGroupBy:
        from narwhals._duckdb.group_by import DuckDBGroupBy

        return DuckDBGroupBy(
            compliant_frame=self, keys=list(keys), drop_null_keys=drop_null_keys
        )

    def rename(self: Self, mapping: dict[str, str]) -> Self:
        df = self._native_frame
        selection = [
            f"{col} as {mapping[col]}" if col in mapping else col for col in df.columns
        ]
        return self._from_native_frame(df.select(", ".join(selection)))

    def join(
        self: Self,
        other: Self,
        *,
        how: Literal["left", "inner", "cross", "anti", "semi"],
        left_on: str | list[str] | None,
        right_on: str | list[str] | None,
        suffix: str,
    ) -> Self:
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]
        original_alias = self._native_frame.alias

        if how == "cross":
            if self._backend_version < (1, 1, 4):
                msg = f"DuckDB>=1.1.4 is required for cross-join, found version: {self._backend_version}"
                raise NotImplementedError(msg)
            rel = self._native_frame.set_alias("lhs").cross(  # pragma: no cover
                other._native_frame.set_alias("rhs")
            )
        else:
            # help mypy
            assert left_on is not None  # noqa: S101
            assert right_on is not None  # noqa: S101

            conditions = [
                f'lhs."{left}" = rhs."{right}"' for left, right in zip(left_on, right_on)
            ]
            condition = " and ".join(conditions)
            rel = self._native_frame.set_alias("lhs").join(
                other._native_frame.set_alias("rhs"), condition=condition, how=how
            )

        if how in ("inner", "left", "cross"):
            select = [f'lhs."{x}"' for x in self._native_frame.columns]
            for col in other._native_frame.columns:
                if col in self._native_frame.columns and (
                    right_on is None or col not in right_on
                ):
                    select.append(f'rhs."{col}" as "{col}{suffix}"')
                elif right_on is None or col not in right_on:
                    select.append(col)
        else:  # semi
            select = ["lhs.*"]

        res = rel.select(", ".join(select)).set_alias(original_alias)
        return self._from_native_frame(res)

    def join_asof(
        self: Self,
        other: Self,
        *,
        left_on: str | None,
        right_on: str | None,
        by_left: list[str] | None,
        by_right: list[str] | None,
        strategy: Literal["backward", "forward", "nearest"],
        suffix: str,
    ) -> Self:
        lhs = self._native_frame
        rhs = other._native_frame
        conditions = []
        if by_left is not None and by_right is not None:
            conditions += [
                f'lhs."{left}" = rhs."{right}"' for left, right in zip(by_left, by_right)
            ]
        else:
            by_left = by_right = []
        if strategy == "backward":
            conditions += [f'lhs."{left_on}" >= rhs."{right_on}"']
        elif strategy == "forward":
            conditions += [f'lhs."{left_on}" <= rhs."{right_on}"']
        else:
            msg = "Only 'backward' and 'forward' strategies are currently supported for DuckDB"
            raise NotImplementedError(msg)
        condition = " and ".join(conditions)
        select = ["lhs.*"]
        for col in rhs.columns:
            if col in lhs.columns and (
                right_on is None or col not in [right_on, *by_right]
            ):
                select.append(f'rhs."{col}" as "{col}{suffix}"')
            elif right_on is None or col not in [right_on, *by_right]:
                select.append(col)
        query = f"""
            SELECT {",".join(select)}
            FROM lhs
            ASOF LEFT JOIN rhs
            ON {condition}
            """  # noqa: S608
        res = duckdb.sql(query)
        return self._from_native_frame(res)

    def collect_schema(self: Self) -> dict[str, DType]:
        return {
            column_name: native_to_narwhals_dtype(str(duckdb_dtype), self._version)
            for column_name, duckdb_dtype in zip(
                self._native_frame.columns, self._native_frame.types
            )
        }

    def unique(self: Self, subset: Sequence[str] | None, keep: str) -> Self:
        if subset is not None:
            import duckdb

            rel = self._native_frame
            # Sanitise input
            if any(x not in rel.columns for x in subset):
                msg = f"Columns {set(subset).difference(rel.columns)} not found in {rel.columns}."
                raise ColumnNotFoundError(msg)
            idx_name = f'"{generate_temporary_column_name(8, rel.columns)}"'
            count_name = (
                f'"{generate_temporary_column_name(8, [*rel.columns, idx_name])}"'
            )
            if keep == "none":
                keep_condition = f"where {count_name}=1"
            else:
                keep_condition = f"where {idx_name}=1"
            query = f"""
                with cte as (
                    select *,
                           row_number() over (partition by {",".join(subset)}) as {idx_name},
                           count(*) over (partition by {",".join(subset)}) as {count_name}
                    from rel
                )
                select * exclude ({idx_name}, {count_name}) from cte {keep_condition}
                """  # noqa: S608
            return self._from_native_frame(duckdb.sql(query))
        return self._from_native_frame(self._native_frame.unique(", ".join(self.columns)))

    def sort(
        self: Self,
        *by: str,
        descending: bool | Sequence[bool],
        nulls_last: bool,
    ) -> Self:
        if isinstance(descending, bool):
            descending = [descending] * len(by)
        descending_str = ["desc" if x else "" for x in descending]

        result = self._native_frame.order(
            ",".join(
                (
                    f'"{col}" {desc} nulls last'
                    if nulls_last
                    else f'"{col}" {desc} nulls first'
                    for col, desc in zip(by, descending_str)
                )
            )
        )
        return self._from_native_frame(result)

    def drop_nulls(self: Self, subset: list[str] | None) -> Self:
        import duckdb

        rel = self._native_frame
        subset_ = subset if subset is not None else rel.columns
        keep_condition = " and ".join(f'"{col}" is not null' for col in subset_)
        query = f"select * from rel where {keep_condition}"  # noqa: S608
        return self._from_native_frame(duckdb.sql(query))
