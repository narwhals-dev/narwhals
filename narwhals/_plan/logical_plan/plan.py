from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, overload

from narwhals._plan._immutable import Immutable
from narwhals._plan.common import todo
from narwhals._plan.expressions import selectors as s_ir
from narwhals._plan.expressions.boolean import all_horizontal
from narwhals._plan.options import VConcatOptions
from narwhals._plan.schema import freeze_schema
from narwhals._plan.typing import Seq
from narwhals._typing_compat import TypeVar
from narwhals._utils import normalize_path, qualified_type_name, zip_strict
from narwhals.exceptions import InvalidOperationError

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

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.expressions import ExprIR, SelectorIR
    from narwhals._plan.options import (
        ExplodeOptions,
        JoinOptions,
        SortMultipleOptions,
        UniqueOptions,
        UnpivotOptions,
    )
    from narwhals._plan.schema import FrozenSchema
    from narwhals.typing import ConcatMethod, FileSource, PivotAgg

Incomplete: TypeAlias = Any
_InputsT = TypeVar("_InputsT", bound="Seq[LogicalPlan]")
LpFunctionT = TypeVar("LpFunctionT", bound="LpFunction", default="LpFunction")
SinkT = TypeVar("SinkT", bound="Sink", default="Sink")
VConcatMethod: TypeAlias = Literal[
    "vertical", "diagonal", "vertical_relaxed", "diagonal_relaxed"
]


class LogicalPlan(Immutable):
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

    has_inputs: ClassVar[bool]
    """Cheap check for `Scan` vs `SingleInput | MultipleInputs`"""

    def __init_subclass__(
        cls: type[Self], *args: Any, has_inputs: bool | None = None, **kwds: Any
    ) -> None:
        super().__init_subclass__(*args, **kwds)
        if has_inputs is not None:
            cls.has_inputs = has_inputs
        elif getattr(cls, "has_inputs", None) is None:
            parent_name = LogicalPlan.__name__
            msg = (
                f"`has_inputs` is a required argument in direct subclasses of {parent_name!r}.\n"
                f"Hint: instead try `class {cls.__name__}({parent_name}, has_inputs=<True|False>): ...`"
            )
            raise TypeError(msg)

    def iter_left(self) -> Iterator[LogicalPlan]:
        """Yield nodes recursively from root->leaf."""
        for input in self.iter_inputs():
            yield from input.iter_left()
        yield self

    def iter_right(self) -> Iterator[LogicalPlan]:
        """Yield nodes recursively from leaf->root."""
        msg = f"TODO: `{type(self).__name__}.iter_right`"
        raise NotImplementedError(msg)

    def iter_inputs(self) -> Iterator[LogicalPlan]:
        """Yield direct input nodes to leaf.

        Equivalent to [`IR.inputs`] and [`ir::Inputs`].

        [`IR.inputs`]: https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/ir/inputs.rs#L204-L239
        [`ir::Inputs`]: https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/ir/inputs.rs#L301-L335
        """
        msg = f"TODO: `{type(self).__name__}.iter_inputs`"
        raise NotImplementedError(msg)

    def explain(self) -> str:
        """Create a string representation of the query plan."""
        from narwhals._plan.logical_plan import _explain

        return _explain.explain(self)

    def __repr__(self) -> str:
        from narwhals._plan.logical_plan import _explain

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
        self, columns: SelectorIR, options: ExplodeOptions
    ) -> MapFunction[Explode]:
        return self._map(Explode(columns=columns, options=options))

    def filter(self, predicate: ExprIR) -> Filter:
        return Filter(input=self, predicate=predicate)

    def group_by(self, keys: Seq[ExprIR], aggs: Seq[ExprIR]) -> GroupBy:
        return GroupBy(input=self, keys=keys, aggs=aggs)

    def pivot(
        self,
        on: SelectorIR,
        on_columns: Incomplete,
        index: SelectorIR,
        values: SelectorIR,
        agg: PivotAgg | None,
        separator: str,
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
        return self._map(Rename(old=tuple(mapping), new=tuple(mapping.values())))

    # TODO @dangotbanned: Decide on if `ProjectionOptions` should be added
    # Either replace `Incomplete` or remove `options` (and the placeholder in `fill_null`)
    def select(self, exprs: Seq[ExprIR], options: Incomplete = None) -> Select:
        return Select(input=self, exprs=exprs)

    def slice(self, offset: int, length: int | None = None) -> Slice:
        return Slice(input=self, offset=offset, length=length)

    def sort(self, by: Seq[SelectorIR], options: SortMultipleOptions) -> Sort:
        return Sort(input=self, by=by, options=options)

    def unique(self, subset: Seq[SelectorIR] | None, options: UniqueOptions) -> Unique:
        return Unique(input=self, subset=subset, options=options)

    def unique_by(
        self,
        subset: Seq[SelectorIR] | None,
        options: UniqueOptions,
        order_by: Seq[SelectorIR],
    ) -> UniqueBy:
        return UniqueBy(input=self, subset=subset, options=options, order_by=order_by)

    def unnest(self, columns: SelectorIR) -> MapFunction[Unnest]:
        return self._map(Unnest(columns=columns))

    def unpivot(
        self, on: SelectorIR | None, index: SelectorIR, options: UnpivotOptions
    ) -> MapFunction[Unpivot]:
        # NOTE: polars uses `cs.empty()` when `index` is None
        # but `on` goes through a very long chain as None:
        #   (python) -> `PyLazyFrame` -> `LazyFrame` -> `DslPlan` -> `UnpivotArgsDSL`
        # then finally filled in for `UnpivotArgsIR::new`
        # https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-core/src/frame/explode.rs#L34-L49
        return self._map(Unpivot(on=on, index=index, options=options))

    def with_columns(self, exprs: Seq[ExprIR], options: Incomplete = None) -> WithColumns:
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
        options: JoinOptions,
    ) -> Join:
        return Join(
            inputs=(self, other), left_on=left_on, right_on=right_on, options=options
        )

    join_asof = todo()

    # Terminal
    def collect(self) -> Collect:
        return self._sink(Collect(input=self))

    def sink_parquet(self, target: FileSource) -> SinkParquet:
        return self._sink(SinkParquet(input=self, target=normalize_path(target)))

    # Sugar
    def drop(self, columns: SelectorIR) -> Select:
        return self.select(((~columns.to_narwhals())._ir,))

    def drop_nulls(self, subset: SelectorIR | None) -> Filter:
        predicate = all_horizontal((subset or s_ir.all()).to_narwhals().is_not_null()._ir)
        return self.filter(predicate)

    def fill_null(self, fill_value: ExprIR) -> Select:
        return self.select(
            (s_ir.all().to_narwhals().fill_null(fill_value.to_narwhals())._ir,),
            options={"duplicate_check": False},  # ProjectionOptions
        )

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


class ScanCsv(ScanFile): ...


class ScanParquet(ScanFile): ...


# TODO @dangotbanned: Careful think about how (non-`ScanFile`) source nodes should work
# - Schema only?
# - Different for eager vs lazy?
class ScanDataFrame(Scan):
    # https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/dsl/plan.rs#L53-L58
    __slots__ = ("df", "schema")
    df: DataFrame[Any, Any]
    schema: FrozenSchema

    @classmethod
    def from_narwhals(cls, df: DataFrame[Any, Any]) -> ScanDataFrame:
        obj = cls.__new__(cls)
        object.__setattr__(obj, "df", df.clone())
        object.__setattr__(obj, "schema", freeze_schema(df.schema))
        return obj

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        # NOTE: Deferring how to handle the hash *for now*
        # Currently, every `ScanDataFrame` will have a unique pseudo-hash
        # Caching a native table seems like a non-starter, once `pandas` enters the party
        yield from (id(self.df), self.schema)

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"df=nw.{type(self.df).__name__}[{qualified_type_name(self.df.to_native())}](...), "
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


class Collect(Sink): ...


class SinkFile(Sink):
    __slots__ = ("target",)
    target: str
    """`file: str | Path | BytesIO` on main."""


class SinkParquet(SinkFile): ...


class Select(SingleInput):
    __slots__ = ("exprs",)
    # NOTE: Contains a `should_broadcast` flag, but AFAICT
    # is only replaced with `False` during optimization (not when building the plan)
    # `options: ProjectionOptions`
    exprs: Seq[ExprIR]


# `DslPlan::HStack`
class WithColumns(SingleInput):
    __slots__ = ("exprs",)
    # NOTE: Same `ProjectionOptions` comment as `Select`
    exprs: Seq[ExprIR]


class Filter(SingleInput):
    __slots__ = ("predicate",)
    predicate: ExprIR


class GroupBy(SingleInput):
    __slots__ = ("aggs", "keys")
    keys: Seq[ExprIR]
    aggs: Seq[ExprIR]


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


# `DslPlan::Distinct`
class Unique(SingleInput):
    __slots__ = ("options", "subset")
    subset: Seq[SelectorIR] | None
    options: UniqueOptions


class UniqueBy(Unique):
    __slots__ = ("order_by",)
    order_by: Seq[SelectorIR]


class Sort(SingleInput):
    __slots__ = ("by", "options")
    by: Seq[SelectorIR]
    options: SortMultipleOptions


class Slice(SingleInput):
    __slots__ = ("length", "offset")
    offset: int
    length: int | None


class MapFunction(SingleInput, Generic[LpFunctionT]):
    __slots__ = ("function",)
    function: LpFunctionT


class Join(MultipleInputs[tuple[LogicalPlan, LogicalPlan]]):
    """Join two tables in an SQL-like fashion."""

    __slots__ = ("left_on", "options", "right_on")
    left_on: Seq[str]
    right_on: Seq[str]
    options: JoinOptions


# `DslPlan::Union`
class VConcat(MultipleInputs[Seq[LogicalPlan]]):
    """`concat(how= "vertical" | "diagonal")`."""

    __slots__ = ("options",)
    options: VConcatOptions


class HConcat(MultipleInputs[Seq[LogicalPlan]]):
    """`concat(how="horizontal")`."""


# NOTE: `DslFunction`
# (reprs from) https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/functions/mod.rs#L302-L382
class LpFunction(Immutable): ...


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


class Unnest(LpFunction):
    __slots__ = ("columns",)
    columns: SelectorIR

    def __repr__(self) -> str:
        return f"UNNEST by: {self.columns!r}"


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


class RowIndex(LpFunction):
    __slots__ = ("name",)
    name: str

    def __repr__(self) -> str:
        return f"ROW INDEX name: {self.name}"


class RowIndexBy(RowIndex):
    __slots__ = ("order_by",)
    order_by: Seq[SelectorIR]

    def __repr__(self) -> str:
        return f"ROW INDEX[name: {self.name}, order_by: {list(self.order_by)!r}]"


class Rename(LpFunction):
    __slots__ = ("new", "old")
    old: Seq[str]
    new: Seq[str]

    @property
    def mapping(self) -> dict[str, str]:
        return dict(zip_strict(self.old, self.new))

    def __repr__(self) -> str:
        return f"RENAME {self.mapping!r}"


concat = LogicalPlan.concat
from_df = LogicalPlan.from_df
scan_csv = LogicalPlan.scan_csv
scan_parquet = LogicalPlan.scan_parquet
