from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Literal
from typing import Mapping
from typing import Sequence

import ibis.expr.types as ir

from narwhals._ibis.utils import evaluate_exprs
from narwhals._ibis.utils import native_to_narwhals_dtype
from narwhals.dependencies import get_ibis
from narwhals.exceptions import ColumnNotFoundError
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import CompliantDataFrame
from narwhals.typing import CompliantLazyFrame
from narwhals.utils import Implementation
from narwhals.utils import Version
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

    from narwhals._ibis.expr import IbisExpr
    from narwhals._ibis.group_by import IbisGroupBy
    from narwhals._ibis.namespace import IbisNamespace
    from narwhals._ibis.series import IbisInterchangeSeries
    from narwhals.dtypes import DType
    from narwhals.typing import AsofJoinStrategy
    from narwhals.typing import JoinStrategy
    from narwhals.typing import LazyUniqueKeepStrategy
    from narwhals.utils import _FullContext


class IbisLazyFrame(CompliantLazyFrame["IbisExpr", "ir.Table"]):
    _implementation = Implementation.IBIS

    def __init__(
        self: Self,
        df: ir.Table,
        *,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._native_frame: ir.Table = df
        self._version = version
        self._backend_version = backend_version
        self._cached_schema: dict[str, DType] | None = None
        validate_backend_version(self._implementation, self._backend_version)

    @staticmethod
    def _is_native(obj: ir.Table | Any) -> TypeIs[ir.Table]:
        return isinstance(obj, ir.Table)

    @classmethod
    def from_native(cls, data: ir.Table, /, *, context: _FullContext) -> Self:
        return cls(
            data, backend_version=context._backend_version, version=context._version
        )

    def __narwhals_dataframe__(self: Self) -> Self:  # pragma: no cover
        # Keep around for backcompat.
        if self._version is not Version.V1:
            msg = "__narwhals_dataframe__ is not implemented for IbisLazyFrame"
            raise AttributeError(msg)
        return self

    def __narwhals_lazyframe__(self: Self) -> Self:
        return self

    def __native_namespace__(self: Self) -> ModuleType:
        return get_ibis()

    def __narwhals_namespace__(self: Self) -> IbisNamespace:
        from narwhals._ibis.namespace import IbisNamespace

        return IbisNamespace(backend_version=self._backend_version, version=self._version)

    def __getitem__(self: Self, item: str) -> IbisInterchangeSeries:
        from narwhals._ibis.series import IbisInterchangeSeries

        return IbisInterchangeSeries(self.native.select(item), version=self._version)

    def _iter_columns(self) -> Iterator[ir.Expr]:
        for name in self.columns:
            yield self.native[name]

    def collect(
        self: Self,
        backend: ModuleType | Implementation | str | None,
        **kwargs: Any,
    ) -> CompliantDataFrame[Any, Any, Any]:
        if backend is None or backend is Implementation.PYARROW:
            import pyarrow as pa  # ignore-banned-import

            from narwhals._arrow.dataframe import ArrowDataFrame

            return ArrowDataFrame(
                native_dataframe=self.native.to_pyarrow(),
                backend_version=parse_version(pa),
                version=self._version,
                validate_column_names=True,
            )

        if backend is Implementation.PANDAS:
            import pandas as pd  # ignore-banned-import

            from narwhals._pandas_like.dataframe import PandasLikeDataFrame

            return PandasLikeDataFrame(
                native_dataframe=self.native.to_pandas(),
                implementation=Implementation.PANDAS,
                backend_version=parse_version(pd),
                version=self._version,
                validate_column_names=True,
            )

        if backend is Implementation.POLARS:
            import polars as pl  # ignore-banned-import

            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(
                df=self.native.to_polars(),
                backend_version=parse_version(pl),
                version=self._version,
            )

        msg = f"Unsupported `backend` value: {backend}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def head(self: Self, n: int) -> Self:
        return self._with_native(self.native.head(n))

    def simple_select(self, *column_names: str) -> Self:
        return self._with_native(self.native.select(*column_names))

    def aggregate(self: Self, *exprs: IbisExpr) -> Self:
        selection = [val.name(name) for name, val in evaluate_exprs(self, *exprs)]
        return self._with_native(self.native.aggregate(selection))

    def select(
        self: Self,
        *exprs: IbisExpr,
    ) -> Self:
        from ibis.expr.operations.window import WindowFunction

        selection = [val.name(name) for name, val in evaluate_exprs(self, *exprs)]
        if not selection:
            msg = "At least one expression must be provided to `select` with the Ibis backend."
            raise ValueError(msg)

        t = self.native.select(*selection)

        # Ibis broadcasts aggregate functions in selects as window functions, keeping the original number of rows.
        # Need to reduce it to a single row if they are all window functions, by calling .distinct()
        if all(isinstance(c, WindowFunction) for c in t.op().values.values()):  # noqa: PD011
            t = t.distinct()

        return self._with_native(t)

    def drop(self: Self, columns: Sequence[str], *, strict: bool) -> Self:
        columns_to_drop = parse_columns_to_drop(self, columns=columns, strict=strict)
        selection = (col for col in self.columns if col not in columns_to_drop)
        return self._with_native(self.native.select(*selection))

    def lazy(self: Self, *, backend: Implementation | None = None) -> Self:
        # The `backend`` argument has no effect but we keep it here for
        # backwards compatibility because in `narwhals.stable.v1`
        # function `.from_native()` will return a DataFrame for Ibis.

        if backend is not None:  # pragma: no cover
            msg = "`backend` argument is not supported for Ibis"
            raise ValueError(msg)
        return self

    def with_columns(self: Self, *exprs: IbisExpr) -> Self:
        new_columns_map = dict(evaluate_exprs(self, *exprs))
        return self._with_native(self.native.mutate(**new_columns_map))

    def filter(self: Self, predicate: IbisExpr) -> Self:
        # `[0]` is safe as the predicate's expression only returns a single column
        mask = predicate(self)[0]
        return self._with_native(self.native.filter(mask))

    @property
    def schema(self: Self) -> dict[str, DType]:
        if self._cached_schema is None:
            # Note: prefer `self._cached_schema` over `functools.cached_property`
            # due to Python3.13 failures.
            self._cached_schema = {
                name: native_to_narwhals_dtype(ibis_dtype=dtype, version=self._version)
                for name, dtype in self.native.schema().fields.items()
            }
        return self._cached_schema

    @property
    def columns(self: Self) -> list[str]:
        return list(self.native.columns)

    def to_pandas(self: Self) -> pd.DataFrame:
        # only if version is v1, keep around for backcompat
        import pandas as pd  # ignore-banned-import()

        if parse_version(pd) >= (1, 0, 0):
            return self.native.to_pandas()
        else:  # pragma: no cover
            msg = f"Conversion to pandas requires pandas>=1.0.0, found {pd.__version__}"
            raise NotImplementedError(msg)

    def to_arrow(self: Self) -> pa.Table:
        # only if version is v1, keep around for backcompat
        return self.native.to_pyarrow()

    def _with_version(self: Self, version: Version) -> Self:
        return self.__class__(
            self.native,
            version=version,
            backend_version=self._backend_version,
        )

    def _with_native(self: Self, df: ir.Table) -> Self:
        return self.__class__(
            df,
            backend_version=self._backend_version,
            version=self._version,
        )

    def group_by(self: Self, *keys: str, drop_null_keys: bool) -> IbisGroupBy:
        from narwhals._ibis.group_by import IbisGroupBy

        return IbisGroupBy(self, keys, drop_null_keys=drop_null_keys)

    def rename(self: Self, mapping: Mapping[str, str]) -> Self:
        def _rename(col: str) -> str:
            return mapping.get(col, col)

        return self._with_native(self.native.rename(_rename))

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

        if other == self:
            # Ibis does not support self-references unless created as a view
            other = self._with_version(other.native.view())

        if native_how != "cross":
            if left_on is None or right_on is None:
                msg = f"For '{native_how}' joins, both 'left_on' and 'right_on' must be provided."
                raise ValueError(msg)  # pragma: no cover (caught upstream)
            predicates = self._convert_predicates(other, left_on, right_on)
        else:
            # For cross joins, no predicates are needed
            predicates = []

        joined = self.native.join(
            other.native, predicates=predicates, how=native_how, rname="{name}" + suffix
        )
        if native_how == "left":
            # Drop duplicate columns from the right table. Ibis keeps them.
            if right_on is not None:
                for right in right_on if not isinstance(right_on, str) else [right_on]:
                    to_drop = right + suffix
                    if to_drop in joined.columns:
                        joined = joined.drop(right + suffix)

            for pred in predicates:
                if isinstance(pred, str):
                    continue

                left = pred.op().left.name
                right = pred.op().right.name

                # If right column is not in the left table, drop it as it will be present in the left column
                # Mirrors how polars works.
                if left != right and right not in self.columns:
                    joined = joined.drop(right)

        return self._with_native(joined)

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
        if strategy == "backward":
            on_condition = self.native[left_on] >= other.native[right_on]
        elif strategy == "forward":
            on_condition = self.native[left_on] <= other.native[right_on]
        else:
            msg = "Only 'backward' and 'forward' strategies are currently supported for Ibis"
            raise NotImplementedError(msg)

        if by_left is not None and by_right is not None:
            predicates = self._convert_predicates(other, by_left, by_right)
        else:
            predicates = []

        joined = self.native.asof_join(
            other.native,
            on=on_condition,
            predicates=predicates,
            rname="{name}" + suffix,
        )

        # Drop duplicate columns from the right table. Ibis keeps them.
        if right_on is not None:
            for right in right_on if isinstance(right_on, list) else [right_on]:
                to_drop = right + suffix
                if to_drop in joined.columns:
                    joined = joined.drop(right + suffix)

        if by_right is not None:
            for right in by_right if not isinstance(by_right, str) else [by_right]:
                to_drop = right + suffix
                if to_drop in joined.columns:
                    joined = joined.drop(right + suffix)

        return self._with_native(joined)

    def _convert_predicates(
        self, other: Self, left_on: str | Sequence[str], right_on: str | Sequence[str]
    ) -> list[ir.BooleanValue] | Sequence[str]:
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        if len(left_on) != len(right_on):
            msg = "'left_on' and 'right_on' must have the same number of columns."
            raise ValueError(msg)

        if left_on == right_on:
            return left_on

        return [
            self.native[left] == other.native[right]
            for left, right in zip(left_on, right_on)
        ]

    def collect_schema(self: Self) -> dict[str, DType]:
        return {
            name: native_to_narwhals_dtype(ibis_dtype=dtype, version=self._version)
            for name, dtype in self.native.schema().fields.items()
        }

    def unique(
        self: Self, subset: Sequence[str] | None, *, keep: LazyUniqueKeepStrategy
    ) -> Self:
        if subset_ := subset if keep == "any" else (subset or self.columns):
            # Sanitise input
            if any(x not in self.columns for x in subset_):
                msg = f"Columns {set(subset_).difference(self.columns)} not found in {self.columns}."
                raise ColumnNotFoundError(msg)

            mapped_keep: dict[str, Literal["first"] | None] = {
                "any": "first",
                "none": None,
            }
            to_keep = mapped_keep[keep]
            return self._with_native(self.native.distinct(on=subset_, keep=to_keep))
        return self._with_native(self.native.distinct(on=subset))

    def sort(
        self: Self,
        *by: str,
        descending: bool | Sequence[bool],
        nulls_last: bool,
    ) -> Self:
        ibis = get_ibis()

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

        return self._with_native(self.native.order_by(*sort_cols))

    def drop_nulls(self: Self, subset: Sequence[str] | None) -> Self:
        subset_ = subset if subset is not None else self.columns
        return self._with_native(self.native.drop_null(subset_))

    def explode(self: Self, columns: Sequence[str]) -> Self:
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
                "Exploding on multiple columns is not supported with Ibis backend since "
                "we cannot guarantee that the exploded columns have matching element counts."
            )
            raise NotImplementedError(msg)

        return self._with_native(self.native.unnest(columns[0], keep_empty=True))

    def unpivot(
        self: Self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        import ibis.selectors as s

        index_: Sequence[str] = [] if index is None else index
        on_: Sequence[str] = (
            [c for c in self.columns if c not in index_] if on is None else on
        )

        # Discard columns not in the index
        final_columns = list(dict.fromkeys([*index_, variable_name, value_name]))

        unpivoted = self.native.pivot_longer(
            s.cols(*on_),
            names_to=variable_name,
            values_to=value_name,
        )
        return self._with_native(unpivoted.select(*final_columns))

    gather_every = not_implemented.deprecated(
        "`LazyFrame.gather_every` is deprecated and will be removed in a future version."
    )
    tail = not_implemented.deprecated(
        "`LazyFrame.tail` is deprecated and will be removed in a future version."
    )
    with_row_index = not_implemented()
