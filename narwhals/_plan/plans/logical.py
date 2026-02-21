"""Query plan representation that contains unresolved expressions.

The schema is not known at this stage.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any, Generic, Literal, overload

from narwhals._plan._guards import is_seq_column
from narwhals._plan._immutable import Immutable
from narwhals._plan.compliant.typing import Native
from narwhals._plan.expressions import selectors as s_ir
from narwhals._plan.expressions.boolean import all_horizontal
from narwhals._plan.options import (
    ExplodeOptions,
    JoinAsofOptions,
    JoinOptions,
    SortMultipleOptions,
    UniqueOptions,
    UnpivotOptions,
    VConcatOptions,
)
from narwhals._plan.plans._base import _BasePlan
from narwhals._plan.schema import freeze_schema
from narwhals._plan.typing import Seq
from narwhals._typing_compat import TypeVar
from narwhals._utils import (
    Implementation,
    normalize_path,
    qualified_type_name,
    zip_strict,
)
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.expressions import ExprIR, SelectorIR
    from narwhals._plan.plans.resolved import ResolvedPlan
    from narwhals._plan.plans.visitors import LogicalToResolved
    from narwhals._plan.schema import FrozenSchema
    from narwhals._typing import _ArrowImpl, _PolarsImpl
    from narwhals.typing import ConcatMethod, FileSource, PivotAgg

__all__ = [
    "Collect",
    "Explode",
    "Filter",
    "GroupBy",
    "HConcat",
    "Join",
    "LogicalPlan",
    "MapFunction",
    "MultipleInputs",
    "Pivot",
    "Rename",
    "Rename",
    "RowIndex",
    "RowIndexBy",
    "Scan",
    "ScanCsv",
    "ScanDataFrame",
    "ScanFile",
    "ScanParquet",
    "Select",
    "SingleInput",
    "Sink",
    "SinkFile",
    "SinkParquet",
    "Slice",
    "Sort",
    "Unique",
    "UniqueBy",
    "Unnest",
    "Unpivot",
    "VConcat",
    "WithColumns",
    "concat",
    "from_df",
    "scan_csv",
    "scan_parquet",
]

Incomplete: TypeAlias = Any
_Fwd: TypeAlias = "LogicalPlan"
_InputsT = TypeVar("_InputsT", bound="Seq[LogicalPlan]")
LpFunctionT = TypeVar("LpFunctionT", bound="LpFunction", default="LpFunction")
LpFunctionT_co = TypeVar(
    "LpFunctionT_co", bound="LpFunction", default="LpFunction", covariant=True
)
SinkT = TypeVar("SinkT", bound="Sink", default="Sink")
VConcatMethod: TypeAlias = Literal[
    "vertical", "diagonal", "vertical_relaxed", "diagonal_relaxed"
]

PivotOnColumns: TypeAlias = "DataFrame[Any, Any]"
"""See https://github.com/narwhals-dev/narwhals/issues/1901#issuecomment-3697700426
"""

ImplT = TypeVar("ImplT", "_ArrowImpl", "_PolarsImpl")


class LogicalPlan(_BasePlan[_Fwd], _root=True):
    """Representation of `LazyFrame` operations, based on [`polars_plan::dsl::plan::DslPlan`].

    ## Notes

    ### Collection
    Calling [`LazyFrame.collect`] takes us through [`LazyFrame.collect_with_engine`]
    where the plan ends with [`DslPlan::Sink(self.logical_plan, SinkType::Memory)`].

    ### Conversion/lowering
    The first pass, starts at [`LazyFrame.to_alp_optimized`] and leads over to [`polars_plan::plans::conversion::dsl_to_ir::to_alp_impl`].

    This is a big boi, recursive function handling the conversion of [`polars_plan::dsl::plan::DslPlan`] -> [`polars_plan::plans::ir::IR`].

    *Some elements* of the work done here are already part of `narwhals._plan`, like [`_plan._expansion.py`] and [`GroupByResolver`].

    ### Schema
    Part of the conversion uses some high-*er* level APIs for propagating `Schema` transformations between each plan:
    - [`expressions_to_schema`]
    - [`Expr.to_field_amortized`]
    - [`to_expr_ir`]

    This is currently *not* part of `narwhals` or `_plan`.

    Here the approximation of expressions is *richer* than `main`, but has not dived into the `DType` can-o-worms:

        # Real polars
        def lower_rust(expr: Expr) -> ExprIR: ...
        #                    ^^^^     ^^^^^^
        #                    |        |
        #                    |        Expanded, resolved name
        #                    |        Stores an index referring to an `AExpr` (another concept not here)
        #                    |        Has a write-once `output_dtype: DType`
        #                    Builder

        # Over here
        def lower_py(expr: ExprIR) -> NamedIR[ExprIR]: ...
        #                  ^^^^^^     ^^^^^^^ ^^^^^^
        #                  |          |       |
        #                  |          |       Stores the expanded version, using the same type
        #                  |          Expanded, resolved name
        #                  |          No concept of `DType`
        #                  Builder

    [`polars_plan::dsl::plan::DslPlan`]: https://github.com/pola-rs/polars/blob/00d7f7e1c3b24a54a13f235e69584614959f8837/crates/polars-plan/src/dsl/plan.rs#L28-L179
    [`LazyFrame.collect`]: https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-lazy/src/frame/mod.rs#L805-L824
    [`LazyFrame.collect_with_engine`]: https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-lazy/src/frame/mod.rs#L624-L628
    [`DslPlan::Sink(self.logical_plan, SinkType::Memory)`]: https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-lazy/src/frame/mod.rs#L631-L637
    [`LazyFrame.to_alp_optimized`]: https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-lazy/src/frame/mod.rs#L666
    [`polars_plan::plans::conversion::dsl_to_ir::to_alp_impl`]: https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs#L102-L1375
    [`polars_plan::plans::ir::IR`]: https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-plan/src/plans/ir/mod.rs#L38-L168
    [`_plan._expansion.py`]: https://github.com/narwhals-dev/narwhals/blob/9b9122b4ab38a6aebe2f09c29ad0f6191952a7a7/narwhals/_plan/_expansion.py
    [`GroupByResolver`]: https://github.com/narwhals-dev/narwhals/blob/9b9122b4ab38a6aebe2f09c29ad0f6191952a7a7/narwhals/_plan/compliant/group_by.py#L131-L194
    [`expressions_to_schema`]: https://github.com/pola-rs/polars/blob/00d7f7e1c3b24a54a13f235e69584614959f8837/crates/polars-plan/src/utils.rs#L217-L244
    [`Expr.to_field_amortized`]: https://github.com/pola-rs/polars/blob/00d7f7e1c3b24a54a13f235e69584614959f8837/crates/polars-plan/src/dsl/expr/mod.rs#L436-L455
    [`to_expr_ir`]: https://github.com/pola-rs/polars/blob/00d7f7e1c3b24a54a13f235e69584614959f8837/crates/polars-plan/src/plans/conversion/dsl_to_ir/expr_to_ir.rs#L6-L9
    """

    def iter_left(self) -> Iterator[LogicalPlan]:
        for input in self.iter_inputs():
            yield from input.iter_left()
        yield self

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        """Converts `LogicalPlan` to `ResolvedPlan`.

        Most `resolve` implementations call a method that is the
        **snake_case**-equivalent of the class name:

            class ScanParquet(ScanFile):
                def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
                    return resolver.scan_parquet(self)

        Arguments:
            resolver: Any object defining each method name, where all return a `ResolvedPlan`.
        """
        msg = f"TODO: `{type(self).__name__}.resolve`"
        raise NotImplementedError(msg)

    def explain(self) -> str:
        """Create a string representation of the query plan."""
        from narwhals._plan.plans import _explain

        return _explain.explain(self)

    def __repr__(self) -> str:
        from narwhals._plan.plans import _explain

        return _explain._format(self, 0)

    # NOTE: Methods based on [`polars_plan::dsl::builder_dsl::DslBuilder`]
    # [`polars_plan::dsl::builder_dsl::DslBuilder`]: https://github.com/pola-rs/polars/blob/092b7ba3c9c486decb52c7b65b50a31be4892437/crates/polars-plan/src/dsl/builder_dsl.rs

    # Scan
    @classmethod
    def from_df(cls, df: DataFrame[Any, Any], /) -> ScanDataFrame:
        return ScanDataFrame.from_narwhals(df)

    @classmethod
    def scan_csv(cls, source: FileSource, /) -> ScanCsv:
        return ScanCsv.from_source(source)

    @classmethod
    def scan_parquet(cls, source: FileSource, /) -> ScanParquet:
        return ScanParquet.from_source(source)

    # Single Input
    def explode(
        self, columns: SelectorIR, options: ExplodeOptions | None = None
    ) -> MapFunction[Explode]:
        options = options or ExplodeOptions.default()
        return self._map(Explode(columns=columns, options=options))

    def filter(self, predicate: ExprIR) -> Filter:
        return Filter(input=self, predicate=predicate)

    def group_by(self, keys: Seq[ExprIR], aggs: Seq[ExprIR]) -> GroupBy:
        return GroupBy(input=self, keys=keys, aggs=aggs)

    def pivot(
        self,
        on: SelectorIR,
        on_columns: PivotOnColumns,
        *,
        index: SelectorIR,
        values: SelectorIR,
        agg: PivotAgg | None = None,
        separator: str = "_",
    ) -> Pivot:
        return Pivot(
            input=self,
            on=on,
            on_columns=on_columns,
            index=index,
            values=values,
            agg=agg,
            separator=separator,
        )

    def rename(self, mapping: Mapping[str, str]) -> MapFunction[Rename]:
        return self._map(Rename.from_dict(mapping))

    def _select(self, exprs: Seq[ExprIR]) -> Select:
        return Select(input=self, exprs=exprs)

    def select(self, exprs: Seq[ExprIR]) -> Select | SelectNames:
        return (
            self.select_names(tuple(e.name for e in exprs))
            if is_seq_column(exprs)
            else self._select(exprs)
        )

    def select_names(self, names: Seq[str]) -> SelectNames:
        return SelectNames(input=self, names=names)

    def slice(self, offset: int, length: int | None = None) -> Slice:
        return Slice(input=self, offset=offset, length=length)

    def sort(
        self, by: Seq[SelectorIR], options: SortMultipleOptions | None = None
    ) -> Sort:
        return Sort(input=self, by=by, options=options or SortMultipleOptions.default())

    def unique(
        self, subset: Seq[SelectorIR] | None = None, options: UniqueOptions | None = None
    ) -> Unique:
        options = options or UniqueOptions.lazy()
        return Unique(input=self, subset=subset, options=options)

    # NOTE: Wish there was a nice way to give `subset` a default, without complicating `unique` too
    # - Most methods that accept `options` don't need them as keywords (too verbose)
    # - `order_by` intentionally doesn't have a default
    # - the parameter order is already janky
    #  - *`options` always last* is a nice rule, but makes `unique(_by)` is the ugly duckling
    def unique_by(
        self,
        subset: Seq[SelectorIR] | None,
        order_by: Seq[SelectorIR],
        options: UniqueOptions | None = None,
    ) -> UniqueBy:
        options = options or UniqueOptions.lazy()
        return UniqueBy(input=self, subset=subset, options=options, order_by=order_by)

    def unnest(self, columns: SelectorIR) -> MapFunction[Unnest]:
        return self._map(Unnest(columns=columns))

    def unpivot(
        self,
        on: SelectorIR | None,
        *,
        index: SelectorIR | None = None,
        options: UnpivotOptions | None = None,
    ) -> MapFunction[Unpivot]:
        # NOTE: polars `on` goes through a very long chain as None:
        #   (python) -> `PyLazyFrame` -> `LazyFrame` -> `DslPlan` -> `UnpivotArgsDSL`
        # then finally filled in for `UnpivotArgsIR::new`
        # https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-core/src/frame/explode.rs#L34-L49
        options = options or UnpivotOptions.default()
        return self._map(Unpivot(on=on, index=index or s_ir.empty(), options=options))

    def with_columns(self, exprs: Seq[ExprIR]) -> WithColumns:
        return WithColumns(input=self, exprs=exprs)

    def with_row_index(self, name: str = "index") -> MapFunction[RowIndex]:
        return self._map(RowIndex(name=name))

    def with_row_index_by(
        self, name: str = "index", *, order_by: Seq[SelectorIR]
    ) -> MapFunction[RowIndexBy]:
        return self._map(RowIndexBy(name=name, order_by=order_by))

    # Multiple Inputs
    @overload
    @staticmethod
    def concat(items: Seq[LogicalPlan], *, how: Literal["horizontal"]) -> HConcat: ...
    @overload
    @staticmethod
    def concat(
        items: Seq[LogicalPlan], *, how: VConcatMethod = "vertical"
    ) -> VConcat: ...
    @staticmethod
    def concat(
        items: Seq[LogicalPlan], *, how: ConcatMethod | VConcatMethod = "vertical"
    ) -> HConcat | VConcat:
        if how == "horizontal":
            return HConcat(inputs=items)
        return VConcat(inputs=items, options=VConcatOptions.from_how(how))

    def join(
        self,
        other: LogicalPlan,
        left_on: Seq[str],
        right_on: Seq[str],
        options: JoinOptions | None = None,
    ) -> Join:
        options = options or JoinOptions.default()
        inputs = (self, other)
        return Join(inputs=inputs, left_on=left_on, right_on=right_on, options=options)

    def join_cross(self, other: LogicalPlan, *, suffix: str = "_right") -> Join:
        return self.join(other, (), (), JoinOptions(how="cross", suffix=suffix))

    def join_asof(
        self,
        other: LogicalPlan,
        left_on: str,
        right_on: str,
        options: JoinAsofOptions | None = None,
    ) -> JoinAsof:
        inputs = (self, other)
        options = options or JoinAsofOptions.parse()
        return JoinAsof(
            inputs=inputs, left_on=left_on, right_on=right_on, options=options
        )

    # Terminal
    def collect(self) -> Collect:
        return self._sink(Collect(input=self))

    # TODO @dangotbanned: Handle `BytesIO`
    def sink_parquet(self, target: FileSource | BytesIO) -> SinkParquet:
        if isinstance(target, BytesIO):
            msg = "TODO: LazyFrame.sink_parquet(BytesIO)"
            raise NotImplementedError(msg)
        return self._sink(SinkParquet(input=self, target=normalize_path(target)))

    # Sugar
    def drop(self, columns: SelectorIR) -> Select:
        return self._select(((~columns.to_narwhals())._ir,))

    def drop_nulls(self, subset: SelectorIR | None = None) -> Filter:
        predicate = all_horizontal((subset or s_ir.all()).to_narwhals().is_not_null()._ir)
        return self.filter(predicate)

    def head(self, n: int = 5) -> Slice:
        return self.slice(0, n)

    def tail(self, n: int = 5) -> Slice:
        return self.slice(-n, n)

    def with_column(self, expr: ExprIR) -> WithColumns:
        return self.with_columns((expr,))

    def _map(self, function: LpFunctionT) -> MapFunction[LpFunctionT]:
        return MapFunction(input=self, function=function)

    def _sink(self, sink: SinkT) -> SinkT:
        if isinstance(self, Sink):
            msg = "cannot create a sink on top of another sink"
            raise InvalidOperationError(msg)
        return sink


class Scan(LogicalPlan, has_inputs=False):
    """Root node of a `LogicalPlan`.

    All plans start with either:
    - Reading from a file (`ScanFile`)
    - Reading from an in-memory dataset (`ScanDataFrame`)

    So the next question is, how do we introduce native lazy objects into mix?
    """

    def iter_right(self) -> Iterator[LogicalPlan]:
        yield self

    def iter_inputs(self) -> Iterator[LogicalPlan]:
        yield from ()


class ScanFile(Scan):
    # https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/dsl/plan.rs#L43-L52
    # https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/dsl/file_scan/mod.rs#L47-L74
    __slots__ = ("source",)
    source: str

    @classmethod
    def from_source(cls, source: FileSource, /) -> Self:
        return cls(source=normalize_path(source))


class ScanCsv(ScanFile):
    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.scan_csv(self)


class ScanParquet(ScanFile):
    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.scan_parquet(self)


class ScanParquetImpl(ScanParquet, Generic[ImplT]):
    """(experimental) Track the initial `backend` used on entry."""

    __slots__ = ("implementation",)
    implementation: ImplT

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.scan_parquet_impl(self)

    # NOTE: (temp) Hacking around adding a default in the signature
    # Alternatives:
    # 1. *not* adding another class and using a default of `Implementation.Unknown in `ScanFile.from_source`
    # 2. default in `LogicalPlan.scan_parquet`, which then picks either class
    @classmethod
    def from_source(  # type: ignore[override]
        cls: type[ScanParquetImpl[Any]], source: FileSource, /
    ) -> ScanParquetImpl[_ArrowImpl]:
        impl = Implementation.PYARROW
        return cls(source=normalize_path(source), implementation=impl)


# TODO @dangotbanned: Careful think about how (non-`ScanFile`) source nodes should work
# - Schema only?
# - Different for eager vs lazy?
class ScanDataFrame(Scan):
    # https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/dsl/plan.rs#L53-L58
    __slots__ = ("frame", "schema")
    frame: DataFrame[Any, Any]
    schema: FrozenSchema

    @classmethod
    def from_narwhals(cls, frame: DataFrame[Any, Any], /) -> ScanDataFrame:
        obj = cls.__new__(cls)
        object.__setattr__(obj, "frame", frame.clone())
        object.__setattr__(obj, "schema", freeze_schema(frame.schema))
        return obj

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        # NOTE: Deferring how to handle the hash *for now*
        # Currently, every `ScanDataFrame` will have a unique pseudo-hash
        # Caching a native table seems like a non-starter, once `pandas` enters the party
        yield from (id(self.frame), self.schema)

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"frame=nw.{type(self.frame).__name__}[{qualified_type_name(self.frame.to_native())}](...), "
            f"schema={self.schema!s})"
        )

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.scan_dataframe(self)


class ScanLazyFrame(Scan, Generic[Native]):
    """Target for `LazyFrame.from_native`.

    The resulting `LazyFrame` stores this as:

        LazyFrame._plan: ScanLazyFrame[Native]
    """

    __slots__ = ("frame", "schema")
    frame: CompliantLazyFrame[Native]
    schema: FrozenSchema

    @classmethod
    def from_compliant(
        cls, frame: CompliantLazyFrame[Native], /
    ) -> ScanLazyFrame[Native]:
        obj = cls.__new__(cls)
        object.__setattr__(obj, "frame", frame)
        object.__setattr__(obj, "schema", freeze_schema(frame.collect_schema()))
        return obj

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"frame=nw.{type(self.frame).__name__}[{qualified_type_name(self.frame.native)}](...), "
            f"schema={self.schema!s})"
        )


class SingleInput(LogicalPlan, has_inputs=True):
    __slots__ = ("input",)
    input: LogicalPlan

    def iter_right(self) -> Iterator[LogicalPlan]:
        yield self
        yield from self.input.iter_right()

    def iter_inputs(self) -> Iterator[LogicalPlan]:
        yield self.input


class MultipleInputs(LogicalPlan, Generic[_InputsT], has_inputs=True):
    __slots__ = ("inputs",)
    inputs: _InputsT

    def iter_right(self) -> Iterator[LogicalPlan]:
        yield self
        for input in reversed(self.inputs):
            yield from input.iter_right()

    def iter_inputs(self) -> Iterator[LogicalPlan]:
        yield from self.inputs


class Sink(SingleInput):
    """Terminal node of a `LogicalPlan`."""


class Collect(Sink):
    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.collect(self)


class SinkFile(Sink):
    __slots__ = ("target",)
    target: str
    """`file: str | Path | BytesIO` on main."""


class SinkParquet(SinkFile):
    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.sink_parquet(self)


class Select(SingleInput):
    __slots__ = ("exprs",)
    exprs: Seq[ExprIR]

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.select(self)


class SelectNames(SingleInput):
    __slots__ = ("names",)
    names: Seq[str]

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.select_names(self)


# `DslPlan::HStack`
class WithColumns(SingleInput):
    __slots__ = ("exprs",)
    exprs: Seq[ExprIR]

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.with_columns(self)


class Filter(SingleInput):
    __slots__ = ("predicate",)
    predicate: ExprIR

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.filter(self)


class GroupBy(SingleInput):
    __slots__ = ("aggs", "keys")
    keys: Seq[ExprIR]
    aggs: Seq[ExprIR]

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.group_by(self)


class Pivot(SingleInput):
    # https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/dsl/plan.rs#L109-L115
    __slots__ = ("agg", "index", "on", "on_columns", "separator", "values")
    on: SelectorIR
    on_columns: Incomplete
    """`DataFrame` in both narwhals and polars (not `DataFrameScan`)."""
    index: SelectorIR
    values: SelectorIR
    agg: PivotAgg | None
    """polars has *just* `Expr`."""
    separator: str

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.pivot(self)


# `DslPlan::Distinct`
class Unique(SingleInput):
    __slots__ = ("options", "subset")
    subset: Seq[SelectorIR] | None
    options: UniqueOptions

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.unique(self)


class UniqueBy(Unique):
    __slots__ = ("order_by",)
    order_by: Seq[SelectorIR]

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.unique_by(self)


class Sort(SingleInput):
    __slots__ = ("by", "options")
    by: Seq[SelectorIR]
    options: SortMultipleOptions

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.sort(self)


class Slice(SingleInput):
    __slots__ = ("length", "offset")
    offset: int
    length: int | None

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.slice(self)


class MapFunction(SingleInput, Generic[LpFunctionT]):
    __slots__ = ("function",)
    function: LpFunctionT

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.map_function(self)


class Join(MultipleInputs[tuple[LogicalPlan, LogicalPlan]]):
    """Join two tables in an SQL-like fashion."""

    __slots__ = ("left_on", "options", "right_on")
    left_on: Seq[str]
    right_on: Seq[str]
    options: JoinOptions

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.join(self)


class JoinAsof(MultipleInputs[tuple[LogicalPlan, LogicalPlan]]):
    __slots__ = ("left_on", "options", "right_on")
    left_on: str
    right_on: str
    options: JoinAsofOptions

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.join_asof(self)


# `DslPlan::Union`
class VConcat(MultipleInputs[Seq[LogicalPlan]]):
    """`concat(how= "vertical" | "diagonal")`."""

    __slots__ = ("options",)
    options: VConcatOptions

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.concat_vertical(self)


class HConcat(MultipleInputs[Seq[LogicalPlan]]):
    """`concat(how="horizontal")`."""

    def resolve(self, resolver: LogicalToResolved, /) -> ResolvedPlan:
        return resolver.concat_horizontal(self)


# NOTE: `DslFunction`
# (reprs from) https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/functions/mod.rs#L302-L382
class LpFunction(Immutable):
    def resolve(
        self, resolver: LogicalToResolved, plan: MapFunction[Incomplete], /
    ) -> ResolvedPlan:
        """Lower the `LogicalPlan` into `ResolvedPlan`."""
        msg = f"TODO: `{type(self).__name__}.resolve`"
        raise NotImplementedError(msg)


class Explode(LpFunction):
    __slots__ = ("columns", "options")
    columns: SelectorIR
    options: ExplodeOptions

    def __repr__(self) -> str:
        opts = self.options
        s = f"EXPLODE {self.columns!r}"
        if not opts.empty_as_null:
            s += ", empty_as_null: False"
        if not opts.keep_nulls:
            s += ", keep_nulls: False"
        return s

    def resolve(
        self, resolver: LogicalToResolved, plan: MapFunction[Explode], /
    ) -> ResolvedPlan:
        return resolver.explode(plan)


class Unnest(LpFunction):
    __slots__ = ("columns",)
    columns: SelectorIR

    def __repr__(self) -> str:
        return f"UNNEST by: {self.columns!r}"

    def resolve(
        self, resolver: LogicalToResolved, plan: MapFunction[Unnest], /
    ) -> ResolvedPlan:
        return resolver.unnest(plan)


class Unpivot(LpFunction):
    # NOTE: polars version is different and probably more efficient, but here:
    # - Trying to avoid `None`
    # - Keeping selectors outside of `*Options` for now
    # https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/functions/dsl.rs#L40-L42
    # https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/dsl/options/mod.rs#L189-L194
    __slots__ = ("index", "on", "options")
    on: SelectorIR | None
    """Default `~index`."""
    index: SelectorIR
    """Default `all()`."""
    options: UnpivotOptions

    def __repr__(self) -> str:
        var, val = self.options.variable_name, self.options.value_name
        return f"UNPIVOT[on: {self.on!r}, index: {self.index!r}, variable_name: {var}, value_name: {val}]"

    def resolve(
        self, resolver: LogicalToResolved, plan: MapFunction[Unpivot], /
    ) -> ResolvedPlan:
        return resolver.unpivot(plan)


class RowIndex(LpFunction):
    __slots__ = ("name",)
    name: str

    def __repr__(self) -> str:
        return f"ROW INDEX name: {self.name}"

    # TODO @dangotbanned: Try to get this and `RowIndexBy` to be nicer to eachother
    def resolve(
        self, resolver: LogicalToResolved, plan: MapFunction[Incomplete], /
    ) -> ResolvedPlan:
        return resolver.with_row_index(plan)


class RowIndexBy(RowIndex):
    __slots__ = ("order_by",)
    order_by: Seq[SelectorIR]

    def __repr__(self) -> str:
        return f"ROW INDEX[name: {self.name}, order_by: {list(self.order_by)!r}]"

    def resolve(
        self, resolver: LogicalToResolved, plan: MapFunction[RowIndexBy], /
    ) -> ResolvedPlan:
        return resolver.with_row_index_by(plan)


class Rename(LpFunction):
    __slots__ = ("new", "old")
    old: Seq[str]
    new: Seq[str]

    @staticmethod
    def from_dict(mapping: Mapping[str, str], /) -> Rename:
        return Rename(old=tuple(mapping), new=tuple(mapping.values()))

    @property
    def mapping(self) -> dict[str, str]:
        return dict(zip_strict(self.old, self.new))

    def __repr__(self) -> str:
        return f"RENAME {self.mapping!r}"


concat = LogicalPlan.concat
from_df = LogicalPlan.from_df
scan_csv = LogicalPlan.scan_csv
scan_parquet = LogicalPlan.scan_parquet
