from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl

import narwhals.exceptions
from narwhals._plan._version import into_version
from narwhals._plan.common import temp
from narwhals._plan.compliant.dataframe import CompliantDataFrame
from narwhals._plan.polars import compat
from narwhals._plan.polars.expr import PolarsExpr as Expr
from narwhals._plan.polars.frame import PolarsFrame
from narwhals._plan.polars.namespace import (
    PolarsNamespace as Namespace,
    dtype_to_native,
    explode_todo,
)
from narwhals._plan.polars.series import PolarsSeries as Series
from narwhals._utils import Implementation, Version, not_implemented, requires
from narwhals.exceptions import NarwhalsError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from io import BytesIO
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self, TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
    from narwhals._plan.polars.lazyframe import PolarsLazyFrame
    from narwhals._plan.typing import Seq
    from narwhals.schema import Schema
    from narwhals.typing import (
        AsofJoinStrategy,
        IntoSchema,
        JoinStrategy,
        PivotAgg,
        UniqueKeepStrategy,
    )


Incomplete: TypeAlias = Any
MAIN = Version.MAIN


class remap_exceptions:  # noqa: N801
    """Fancy version of `catch_polars_exception`.

    Just write *potentially-raising* code `with`-in the context manager:

        with remap_exceptions():
            risky_business()  # Any native exceptions will be re-raised
                              # as their narwhals-equivalent
        business_as_usual()

    Works in a similar way to the implementation of [`suppress.__exit__`].

    See Also:
        [The `with` statement]

    [`suppress.__exit__`]: https://github.com/python/cpython/blob/fa7212b0af1c3d4e0cf8ac2ead35df3541436fb4/Lib/contextlib.py#L450-L469
    [The `with` statement]: https://docs.python.org/3/reference/compound_stmts.html#the-with-statement
    """

    _REMAP: Mapping[type[BaseException], type[NarwhalsError]] = {
        tp: getattr(narwhals.exceptions, tp.__name__, NarwhalsError)
        for tp in pl.exceptions.PolarsError.__subclasses__()
    }

    def __enter__(self) -> None:
        return

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _: object,
        /,
    ) -> bool | None:
        if exc_type is None or exc_value is None:
            return None
        if to_exc := remap_exceptions._REMAP.get(exc_type):
            raise to_exc(str(exc_value)) from None
        return False


class PolarsDataFrame(PolarsFrame, CompliantDataFrame[pl.DataFrame, pl.Series]):
    _native: pl.DataFrame
    _version: Version

    # NOTE: Aliases to integrate with `@requires.backend_version`
    _backend_version = compat.BACKEND_VERSION
    _implementation = Implementation.POLARS

    def __len__(self) -> int:
        return self.native.__len__()

    @property
    def columns(self) -> list[str]:
        return self.native.columns

    @property
    def native(self) -> pl.DataFrame:
        return self._native

    @property
    def version(self) -> Version:
        return self._version

    @property
    def schema(self) -> Schema:
        return into_version(self.version).schema.from_polars(self.native.schema)

    @property
    def shape(self) -> tuple[int, int]:
        return self.native.shape

    @classmethod
    def from_native(cls, native: pl.DataFrame, /, version: Version = MAIN) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._version = version
        return obj

    from_polars = from_native

    def to_arrow(self) -> pa.Table:
        return self.native.to_arrow()

    def to_pandas(self) -> pd.DataFrame:
        return self.native.to_pandas()

    def to_polars(self) -> pl.DataFrame:
        return self.native

    def to_lazy(self) -> PolarsLazyFrame:
        from narwhals._plan.polars.lazyframe import PolarsLazyFrame

        return PolarsLazyFrame.from_native(self.native.lazy(), self.version)

    @overload
    def write_csv(self, target: None, /, **kwds: Any) -> str: ...
    @overload
    def write_csv(self, target: str | Path | BytesIO, /, **kwds: Any) -> None: ...
    def write_csv(
        self, target: str | Path | BytesIO | None, /, **kwds: Any
    ) -> str | None:
        return self.native.write_csv(target, **kwds)

    def write_parquet(self, target: str | BytesIO, /, **kwds: Any) -> None:
        self.native.write_parquet(target, **kwds)

    def __narwhals_namespace__(self) -> Namespace:
        return Namespace(self.version)

    def clone(self) -> Self:
        return self._with_native(self.native.clone())

    def drop(self, columns: Sequence[str]) -> Self:
        return self._with_native(self.native.drop(columns))

    def drop_nulls(self, subset: Sequence[str] | None) -> Self:
        return self._with_native(self.native.drop_nulls(subset))

    def explode(self, columns: Sequence[str], options: ExplodeOptions) -> Self:
        explode_todo(empty_as_null=options.empty_as_null, keep_nulls=options.keep_nulls)
        return self._with_native(self.native.explode(columns))

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        /,
        *,
        schema: IntoSchema | None = None,
        version: Version = MAIN,
    ) -> Self:
        if not schema:
            return cls.from_native(pl.from_dict(data), version)
        s: Any = {k: (dtype_to_native(d, version) if d else d) for k, d in schema.items()}
        return cls.from_native(pl.from_dict(data, s), version)

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self._with_native(self.native.gather_every(n, offset))

    def get_column(self, name: str) -> Series:
        return Series.from_native(self.native.get_column(name), version=self.version)

    def iter_columns(self) -> Iterator[Series]:
        for series in self.native.iter_columns():
            yield Series.from_native(series, version=self.version)

    def join(
        self,
        other: Self,
        *,
        how: JoinStrategy,
        left_on: Sequence[str],
        right_on: Sequence[str],
        suffix: str = "_right",
    ) -> Self:
        how_: Any = (
            "outer" if how == "full" and compat.JOIN_OUTER_RENAMED_TO_FULL else how
        )
        return self._with_native(
            self.native.join(
                other.native, how=how_, left_on=left_on, right_on=right_on, suffix=suffix
            )
        )

    def join_asof(
        self,
        other: Self,
        *,
        left_on: str,
        right_on: str,
        left_by: Sequence[str] = (),
        right_by: Sequence[str] = (),
        strategy: AsofJoinStrategy = "backward",
        suffix: str = "_right",
    ) -> Self:
        return self._with_native(
            self.native.join_asof(
                other.native,
                left_on=left_on,
                right_on=right_on,
                by_left=left_by,
                by_right=right_by,
                strategy=strategy,
                suffix=suffix,
            )
        )

    def join_cross(self, other: Self, *, suffix: str = "_right") -> Self:
        return self._with_native(
            self.native.join(other.native, how="cross", suffix=suffix)
        )

    def select_names(self, *column_names: str) -> Self:
        return self._with_native(self.native.select(column_names))

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self._with_native(self.native.slice(offset, length))

    def sort(self, by: Sequence[str], options: SortMultipleOptions) -> Self:
        return self._with_native(self.native.sort(by, **options.to_polars(by)))

    @overload
    def to_dict(self, *, as_series: Literal[True]) -> Mapping[str, Series]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(
        self, *, as_series: bool
    ) -> Mapping[str, Series] | dict[str, list[Any]]: ...
    def to_dict(self, *, as_series: bool) -> Mapping[str, Series] | dict[str, list[Any]]:
        if as_series:
            return {s.name: s for s in self.iter_columns()}
        return self.native.to_dict(as_series=False)

    def to_series(self, index: int = 0) -> Series:
        return Series.from_native(self.native.to_series(index), version=self.version)

    def rename(self, mapping: Mapping[str, str]) -> Self:
        return self._with_native(self.native.rename(mapping))

    def row(self, index: int) -> tuple[Any, ...]:
        return self.native.row(index)

    def sample_n(
        self, n: int = 1, *, with_replacement: bool = False, seed: int | None = None
    ) -> Self:
        return self._with_native(
            self.native.sample(n, with_replacement=with_replacement, seed=seed)
        )

    def to_struct(self, name: str = "") -> Any:
        return Series.from_native(self.native.to_struct(name), version=self.version)

    def partition_by(self, by: Sequence[str], *, include_key: bool = True) -> list[Self]:
        results = self.native.partition_by(by, include_key=include_key, as_dict=False)
        return [self._with_native(p) for p in results]

    # TODO @dangotbanned: backcompat for `on_columns: DataFrame`?
    # - `sort_columns` has already been consumed to build `on_columns`
    @requires.backend_version((1,))
    def pivot(
        self,
        on: Sequence[str],
        on_columns: Self,
        *,
        index: Sequence[str],
        values: Sequence[str],
        aggregate_function: PivotAgg | None = None,
        separator: str = "_",
        sort_columns: bool = False,
    ) -> Self:
        kwds: dict[str, Incomplete] = (
            {"on_columns": on_columns.native}
            if compat.PIVOT_SUPPORTS_ON_COLUMNS
            else {"sort_columns": sort_columns}
        )
        with remap_exceptions():
            return self._with_native(
                self.native.pivot(
                    on,
                    index=index,
                    values=values,
                    aggregate_function=aggregate_function,
                    separator=separator,
                    **kwds,
                )
            )

    def unique(
        self,
        subset: Sequence[str] | None = None,
        *,
        keep: UniqueKeepStrategy = "any",
        maintain_order: bool = False,
    ) -> Self:
        return self._with_native(
            self.native.unique(subset, keep=keep, maintain_order=maintain_order)
        )

    # NOTE: @dangotbanned: Try wrapping with `.lazy()<query>.collect()` once running older versions in ci
    def unique_by(
        self,
        subset: Sequence[str] | None = None,
        *,
        order_by: Sequence[str],
        keep: UniqueKeepStrategy = "any",
        maintain_order: bool = False,
    ) -> Self:
        if not maintain_order:
            return self._with_native(self.native.sort(order_by).unique(subset, keep=keep))
        # NOTE: Unresolved performance question
        # https://github.com/narwhals-dev/narwhals/blob/3642b8c545be26136ca42306622cf1ade4807d86/narwhals/_plan/polars/lazyframe.py#L205-L222
        names = self.columns
        idx = temp.column_name(names)
        return self._with_native(
            self.native.with_row_index(idx)
            .sort(order_by)
            .unique(subset or names, keep=keep)
            .sort(idx)
            .select(names)
        )

    def unnest(self, columns: Sequence[str]) -> Self:
        return self._with_native(self.native.unnest(columns))

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        *,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> Self:
        if compat.MELT_RENAMED_TO_UNPIVOT:
            result = self.native.unpivot(
                on, index=index, variable_name=variable_name, value_name=value_name
            )
        else:
            result = self.native.melt(
                id_vars=index,
                value_vars=on,
                variable_name=variable_name,
                value_name=value_name,
            )
        return self._with_native(result)

    def with_row_index(self, name: str) -> Self:
        return self._with_native(self.native.with_row_index(name))

    def with_row_index_by(
        self, name: str, order_by: Sequence[str], *, nulls_last: bool = False
    ) -> Self:
        int_range = (
            pl.int_range(pl.len())
            .over(order_by=order_by, nulls_last=nulls_last)
            .alias(name)
        )
        return self._with_native(self.native.select(int_range, pl.all()))

    _group_by = not_implemented()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    lazy = not_implemented()
    filter = not_implemented()

    def _evaluate_irs(self, nodes: Iterable[ir.NamedIR]) -> Iterator[Expr]:
        yield from (Expr.from_named_ir(e, self) for e in nodes)

    def select(self, irs: Seq[ir.NamedIR]) -> Self:
        return self._with_native(
            self.native.select(e.native for e in self._evaluate_irs(irs))
        )

    def with_columns(self, irs: Seq[ir.NamedIR]) -> Self:
        return self._with_native(
            self.native.with_columns(e.native for e in self._evaluate_irs(irs))
        )


PolarsDataFrame()
