from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Sequence

from narwhals._ibis.utils import native_to_narwhals_dtype
from narwhals._ibis.utils import parse_exprs
from narwhals.dependencies import get_ibis
from narwhals.exceptions import ColumnNotFoundError
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import CompliantDataFrame
from narwhals.utils import Implementation
from narwhals.utils import Version
from narwhals.utils import check_column_names_are_unique
from narwhals.utils import import_dtypes_module
from narwhals.utils import parse_columns_to_drop
from narwhals.utils import parse_version
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from types import ModuleType

    import ibis.expr.types as ir
    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._ibis.expr import IbisExpr
    from narwhals._ibis.group_by import IbisGroupBy
    from narwhals._ibis.namespace import IbisNamespace
    from narwhals._ibis.series import IbisInterchangeSeries
    from narwhals.dtypes import DType

from narwhals.typing import CompliantLazyFrame


class IbisLazyFrame(CompliantLazyFrame):
    _implementation = Implementation.IBIS

    def __init__(
        self: Self,
        df: ir.Table,
        *,
        backend_version: tuple[int, ...],
        version: Version,
        validate_column_names: bool,
    ) -> None:
        if validate_column_names:
            check_column_names_are_unique(list(df.columns))
        self._native_frame: ir.Table = df
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
        return get_ibis()  # type: ignore[no-any-return]

    def __narwhals_namespace__(self: Self) -> IbisNamespace:
        from narwhals._ibis.namespace import IbisNamespace

        return IbisNamespace(backend_version=self._backend_version, version=self._version)

    def __getitem__(self: Self, item: str) -> IbisInterchangeSeries:
        from narwhals._ibis.series import IbisInterchangeSeries

        return IbisInterchangeSeries(
            self._native_frame.select(item), version=self._version
        )

    def collect(
        self: Self,
        backend: ModuleType | Implementation | str | None,
        **kwargs: Any,
    ) -> CompliantDataFrame:
        if backend is None or backend is Implementation.PYARROW:
            import pyarrow as pa  # ignore-banned-import

            from narwhals._arrow.dataframe import ArrowDataFrame

            return ArrowDataFrame(
                native_dataframe=self._native_frame.to_pyarrow(),
                backend_version=parse_version(pa),
                version=self._version,
                validate_column_names=False,
            )

        if backend is Implementation.PANDAS:
            import pandas as pd  # ignore-banned-import

            from narwhals._pandas_like.dataframe import PandasLikeDataFrame

            return PandasLikeDataFrame(
                native_dataframe=self._native_frame.to_pandas(),
                implementation=Implementation.PANDAS,
                backend_version=parse_version(pd),
                version=self._version,
                validate_column_names=False,
            )

        if backend is Implementation.POLARS:
            import polars as pl  # ignore-banned-import

            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(
                df=self._native_frame.to_polars(),
                backend_version=parse_version(pl),
                version=self._version,
            )

        msg = f"Unsupported `backend` value: {backend}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def head(self: Self, n: int) -> Self:
        return self._from_native_frame(
            self._native_frame.head(n), validate_column_names=False
        )

    def simple_select(self, *column_names: str) -> Self:
        return self._from_native_frame(
            self._native_frame.select(*column_names), validate_column_names=False
        )

    def aggregate(self: Self, *exprs: IbisExpr) -> Self:
        new_columns_map = parse_exprs(self, *exprs)
        return self._from_native_frame(
            self._native_frame.aggregate(
                [val.name(col) for col, val in new_columns_map.items()]  # type: ignore[arg-type]
            ),
            validate_column_names=False,
        )

    def select(
        self: Self,
        *exprs: IbisExpr,
    ) -> Self:
        from ibis.expr.operations.window import WindowFunction

        new_columns_map = parse_exprs(self, *exprs)
        if not new_columns_map:
            msg = "No columns to select. Must provide at least one column."
            raise ValueError(msg)

        t = self._native_frame.select(**new_columns_map)

        # Ibis broadcasts aggregate functions in selects as window functions, keeping the original number of rows.
        # Need to reduce it to a single row if they are all window functions, by calling .distinct()
        if all(isinstance(c, WindowFunction) for c in t.op().values.values()):  # noqa: PD011
            t = t.distinct()

        return self._from_native_frame(t, validate_column_names=False)

    def drop(self: Self, columns: list[str], strict: bool) -> Self:  # noqa: FBT001
        columns_to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )
        selection = (col for col in self.columns if col not in columns_to_drop)
        return self._from_native_frame(
            self._native_frame.select(*selection), validate_column_names=False
        )

    def lazy(self: Self, *, backend: Implementation | None = None) -> Self:
        # The `backend`` argument has no effect but we keep it here for
        # backwards compatibility because in `narwhals.stable.v1`
        # function `.from_native()` will return a DataFrame for Ibis.

        if backend is not None:  # pragma: no cover
            msg = "`backend` argument is not supported for Ibis"
            raise ValueError(msg)
        return self

    def with_columns(self: Self, *exprs: IbisExpr) -> Self:
        new_columns_map = parse_exprs(self, *exprs)

        return self._from_native_frame(
            self._native_frame.mutate(**new_columns_map), validate_column_names=False
        )

    def filter(self: Self, predicate: IbisExpr) -> Self:
        # `[0]` is safe as the predicate's expression only returns a single column
        mask = predicate._call(self)[0]
        return self._from_native_frame(
            self._native_frame.filter(mask), validate_column_names=False
        )

    @property
    def schema(self: Self) -> dict[str, DType]:
        return {
            name: native_to_narwhals_dtype(dtype=dtype, version=self._version)
            for name, dtype in self._native_frame.schema().fields.items()
        }

    @property
    def columns(self: Self) -> list[str]:
        return list(self._native_frame.columns)  # type: ignore[no-any-return]

    def to_pandas(self: Self) -> pd.DataFrame:
        # only if version is v1, keep around for backcompat
        import pandas as pd  # ignore-banned-import()

        if parse_version(pd) >= (1, 0, 0):
            return self._native_frame.to_pandas()
        else:  # pragma: no cover
            msg = f"Conversion to pandas requires pandas>=1.0.0, found {pd.__version__}"
            raise NotImplementedError(msg)

    def to_arrow(self: Self) -> pa.Table:
        # only if version is v1, keep around for backcompat
        return self._native_frame.to_pyarrow()

    def _change_version(self: Self, version: Version) -> Self:
        return self.__class__(
            self._native_frame,
            version=version,
            backend_version=self._backend_version,
            validate_column_names=False,
        )

    def _from_native_frame(
        self: Self, df: ir.Table, *, validate_column_names: bool = True
    ) -> Self:
        return self.__class__(
            df,
            backend_version=self._backend_version,
            version=self._version,
            validate_column_names=validate_column_names,
        )

    def group_by(self: Self, *keys: str, drop_null_keys: bool) -> IbisGroupBy:
        from narwhals._ibis.group_by import IbisGroupBy

        return IbisGroupBy(
            compliant_frame=self, keys=list(keys), drop_null_keys=drop_null_keys
        )

    def rename(self: Self, mapping: dict[str, str]) -> Self:
        def _rename(col: str) -> str:
            return mapping.get(col, col)

        return self._from_native_frame(self._native_frame.rename(_rename))

    def join(
        self: Self,
        other: Self,
        *,
        how: Literal["left", "inner", "cross", "anti", "semi"],
        left_on: list[str] | None,
        right_on: list[str] | None,
        suffix: str,
    ) -> Self:
        if how != "cross":
            if left_on is None or right_on is None:
                msg = (
                    f"For '{how}' joins, both 'left_on' and 'right_on' must be provided."
                )
                raise ValueError(msg)  # pragma: no cover (caught upstream)
            predicates = self._convert_predicates(other, left_on, right_on)
        else:
            # For cross joins, no predicates are needed
            predicates = []

        joined = self._native_frame.join(
            other._native_frame, predicates=predicates, how=how, rname="{name}" + suffix
        )
        if how == "left":
            # Drop duplicate columns from the right table. Ibis keeps them.
            if right_on is not None:
                for right in right_on if isinstance(right_on, list) else [right_on]:
                    to_drop = right + suffix
                    if to_drop in joined.columns:
                        joined = joined.drop(right + suffix)

            for pred in predicates:
                left = pred.op().left.name
                right = pred.op().right.name

                # If right column is not in the left table, drop it as it will be present in the left column
                # Mirrors how polars works.
                if left != right and right not in self.columns:
                    joined = joined.drop(right)

        return self._from_native_frame(joined)

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
        if strategy == "backward":
            on_condition = self._native_frame[left_on] >= other._native_frame[right_on]
        elif strategy == "forward":
            on_condition = self._native_frame[left_on] <= other._native_frame[right_on]
        else:
            msg = "Only 'backward' and 'forward' strategies are currently supported for Ibis"
            raise NotImplementedError(msg)

        if by_left is not None and by_right is not None:
            predicates = self._convert_predicates(other, by_left, by_right)
        else:
            predicates = []

        joined = self._native_frame.asof_join(
            other._native_frame, on=on_condition, predicates=predicates
        )  # type: ignore[operator]

        # Drop duplicate columns from the right table. Ibis keeps them.
        if right_on is not None:
            for right in right_on if isinstance(right_on, list) else [right_on]:
                to_drop = right + suffix
                if to_drop in joined.columns:
                    joined = joined.drop(right + suffix)

        if by_right is not None:
            for right in by_right if isinstance(by_right, list) else [by_right]:
                to_drop = right + suffix
                if to_drop in joined.columns:
                    joined = joined.drop(right + suffix)

        return self._from_native_frame(joined)

    def _convert_predicates(
        self, other: Self, left_on: str | list[str], right_on: str | list[str]
    ) -> list[ir.BooleanValue]:
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        if len(left_on) != len(right_on):
            msg = "'left_on' and 'right_on' must have the same number of columns."
            raise ValueError(msg)

        return [
            self._native_frame[left] == other._native_frame[right]
            for left, right in zip(left_on, right_on)
        ]

    def collect_schema(self: Self) -> dict[str, DType]:
        return self.schema

    def unique(self: Self, subset: Sequence[str] | None, keep: str) -> Self:
        if subset is not None:
            rel = self._native_frame
            # Sanitise input
            if any(x not in rel.columns for x in subset):
                msg = f"Columns {set(subset).difference(rel.columns)} not found in {rel.columns}."
                raise ColumnNotFoundError(msg)

            mapped_keep: dict[str, Literal["first"] | None] = {
                "any": "first",
                "none": None,
            }
            to_keep = mapped_keep[keep]
            return self._from_native_frame(
                self._native_frame.distinct(on=subset, keep=to_keep),
                validate_column_names=False,
            )
        return self._from_native_frame(
            self._native_frame.distinct(on=self.columns),
            validate_column_names=False,
        )

    def sort(
        self: Self,
        *by: str,
        descending: bool | Sequence[bool],
        nulls_last: bool,
    ) -> Self:
        import ibis

        if isinstance(descending, bool):
            descending = [descending for _ in range(len(by))]

        if nulls_last:
            sort_cols = [
                ibis.desc(by[i], nulls_first=False)
                if descending[i]
                else ibis.asc(by[i], nulls_first=False)
                for i in range(len(by))
            ]
        else:
            sort_cols = [
                ibis.desc(by[i], nulls_first=True)
                if descending[i]
                else ibis.asc(by[i], nulls_first=True)
                for i in range(len(by))
            ]

        return self._from_native_frame(
            self._native_frame.order_by(*sort_cols), validate_column_names=False
        )

    def drop_nulls(self: Self, subset: list[str] | None) -> Self:
        rel = self._native_frame
        subset_ = subset if subset is not None else rel.columns
        return self._from_native_frame(
            self._native_frame.drop(*subset_), validate_column_names=False
        )

    def explode(self: Self, columns: list[str]) -> Self:  # TODO(rwhitten577): IMPLEMENT
        dtypes = import_dtypes_module(self._version)
        schema = self.collect_schema()
        for col in columns:
            dtype = schema[col]

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

        col_to_explode = ColumnExpression(columns[0])
        rel = self._native_frame
        original_columns = self.columns

        not_null_condition = col_to_explode.isnotnull() & FunctionExpression(
            "len", col_to_explode
        ) > lit(0)
        non_null_rel = rel.filter(not_null_condition).select(
            *(
                FunctionExpression("unnest", col_to_explode).alias(col)
                if col in columns
                else col
                for col in original_columns
            )
        )

        null_rel = rel.filter(~not_null_condition).select(
            *(lit(None).alias(col) if col in columns else col for col in original_columns)
        )

        return self._from_native_frame(
            non_null_rel.union(null_rel), validate_column_names=False
        )

    def unpivot(
        self: Self,
        on: list[str] | None,
        index: list[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        import ibis.selectors as s

        index_: list[str] = [] if index is None else index
        on_: list[str] = (
            [c for c in self.columns if c not in index_] if on is None else on
        )

        if variable_name == "":
            variable_name = "variable"
        if value_name == "":
            value_name = "value"

        # Discard columns not in the index
        final_columns = list(dict.fromkeys([*index, variable_name, value_name]))

        unpivoted = self._native_frame.pivot_longer(
            s.cols(*on_),
            names_to=variable_name,
            values_to=value_name,
        )
        return self._from_native_frame(unpivoted.select(*final_columns))
