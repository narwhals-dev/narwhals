from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, Generic

from narwhals._plan._immutable import Immutable
from narwhals._plan.common import todo
from narwhals._plan.schema import freeze_schema
from narwhals._plan.typing import Seq
from narwhals._typing_compat import TypeVar
from narwhals._utils import normalize_path, qualified_type_name, zip_strict

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.expressions import ExprIR, SelectorIR
    from narwhals._plan.options import (
        ExplodeOptions,
        JoinOptions,
        SortMultipleOptions,
        UniqueOptions,
        UnpivotOptions,
        VConcatOptions,
    )
    from narwhals._plan.schema import FrozenSchema
    from narwhals.typing import FileSource, PivotAgg

Incomplete: TypeAlias = Any
_InputsT = TypeVar("_InputsT", bound="Seq[LogicalPlan]")

INDENT_INCREMENT = 2
INDENT = " "


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

    def _format_rec(self, indent: int) -> str:
        # `IRDisplay._format`
        # (here) https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/ir/format.rs#L259-L265
        # (overrides) https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/ir/format.rs#L148-L229
        self_repr = self._format_non_rec(indent)
        if not self.has_inputs:
            return self_repr
        sub_indent = indent + INDENT_INCREMENT
        it = (node._format_rec(sub_indent) for node in self.iter_inputs())
        return "".join(chain([self_repr], it))

    def _format_non_rec(self, indent: int) -> str:
        # `ir::format::write_ir_non_recursive`
        # https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/ir/format.rs#L705-L1006
        msg = f"TODO: `{type(self).__name__}._format_non_rec`"
        raise NotImplementedError(msg)

    def __repr__(self) -> str:
        return self._format_non_rec(0)


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

    # NOTE: Probably want a `staticmethod`, change if nothing is needed from `cls`
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

    def _format_non_rec(self, indent: int) -> str:
        names = self.schema.names
        n_columns = len(names)
        if n_columns > 4:
            it = (f'"{name}"' for name in names[:4])
            s = ", ".join((*it, "..."))
        elif n_columns == 0:
            s = ""
        else:
            s = ", ".join(f'"{name}"' for name in names)
        return f"{INDENT * indent}DF [{s}]; {n_columns} COLUMNS"

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


class Collect(Sink):
    def _format_non_rec(self, indent: int) -> str:
        return f"{INDENT * indent}SINK (memory)"


class SinkFile(Sink):
    __slots__ = ("target",)
    target: str
    """`file: str | Path | BytesIO` on main.

    Not sure `BytesIO` makes sense here.
    """

    def _format_non_rec(self, indent: int) -> str:
        return f"{INDENT * indent}SINK (file)"


class SinkParquet(SinkFile): ...


class Select(SingleInput):
    __slots__ = ("exprs",)
    # NOTE: Contains a `should_broadcast` flag, but AFAICT
    # is only replaced with `False` during optimization (not when building the plan)
    # `options: ProjectionOptions`
    exprs: Seq[ExprIR]

    def _format_non_rec(self, indent: int) -> str:
        return f"{INDENT * indent}SELECT {list(self.exprs)!r}"


# `DslPlan::HStack`
class WithColumns(SingleInput):
    __slots__ = ("exprs",)
    # NOTE: Same `ProjectionOptions` comment as `Select`
    exprs: Seq[ExprIR]

    def _format_non_rec(self, indent: int) -> str:
        pad = INDENT * indent
        return f"{pad} WITH_COLUMNS:\n {pad}{list(self.exprs)!r}"


class Filter(SingleInput):
    __slots__ = ("predicate",)
    predicate: ExprIR

    def _format_non_rec(self, indent: int) -> str:
        pad = INDENT * indent
        return f"{pad}FILTER {self.predicate!r}\n{pad}FROM"


class GroupBy(SingleInput):
    __slots__ = ("aggs", "keys")
    keys: Seq[ExprIR]
    aggs: Seq[ExprIR]

    def _format_non_rec(self, indent: int) -> str:
        pad = INDENT * indent
        return f"{pad}AGGREGATE\n{pad + INDENT}{list(self.aggs)!r} BY {list(self.keys)!r}"

    _format_rec = todo()


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

    def _format_non_rec(self, indent: int) -> str:
        # NOTE: Only exists in `DslPlan`, not `IR` which defines the displays
        return f"{INDENT * indent}PIVOT[...]"


# `DslPlan::Distinct`
class Unique(SingleInput):
    __slots__ = ("options", "subset")
    subset: Seq[SelectorIR] | None
    options: UniqueOptions

    def _format_non_rec(self, indent: int) -> str:
        s = f"{INDENT * indent}UNIQUE[maintain_order: {self.options.maintain_order}, keep: {self.options.keep}]"
        if subset := self.subset:
            s += f"BY {list(subset)!r}"
        return s


class Sort(SingleInput):
    __slots__ = ("by", "options")
    by: Seq[SelectorIR]
    options: SortMultipleOptions

    def _format_non_rec(self, indent: int) -> str:
        opts = self.options
        exprs = ", ".join(f"{e!r}" for e in self.by)
        s = f"{INDENT * indent}SORT BY[{exprs}"
        if any(opts.descending):
            s += f", descending: {list(opts.descending)}"
        if any(opts.nulls_last):
            s += f", nulls_last: {list(opts.nulls_last)}"
        return f"{s}]"


class Slice(SingleInput):
    __slots__ = ("length", "offset")
    offset: int
    length: int | None

    def _format_non_rec(self, indent: int) -> str:
        return f"{INDENT * indent}SLICE[offset: {self.offset}, len: {self.length}]"


class MapFunction(SingleInput):
    # `polars` says this is for UDFs, but uses it for: `Rename`, `RowIndex`, `Unnest`, `Explode`
    __slots__ = ("function",)
    function: LpFunction

    def _format_non_rec(self, indent: int) -> str:
        return f"{INDENT * indent}{self.function!r}"

    _format_rec = todo()


class Join(MultipleInputs[tuple[LogicalPlan, LogicalPlan]]):
    """Join two tables in an SQL-like fashion."""

    __slots__ = ("how", "left_on", "right_on", "suffix")
    left_on: Seq[str]
    right_on: Seq[str]
    options: JoinOptions

    def _format_non_rec(self, indent: int) -> str:
        pad = INDENT * indent
        how = self.options.how.upper()
        operation = f"{pad}{how} JOIN"
        if how == "CROSS":
            return operation
        return f"{operation}:\n{pad}LEFT PLAN ON: {list(self.left_on)!r}\n{pad}RIGHT PLAN ON: {list(self.right_on)!r}"

    _format_rec = todo()


# `DslPlan::Union`
class VConcat(MultipleInputs[Seq[LogicalPlan]]):
    """`concat(how= "vertical" | "diagonal")`."""

    __slots__ = ("options",)
    options: VConcatOptions

    def _format_non_rec(self, indent: int) -> str:
        return f"{INDENT * indent}UNION"

    _format_rec = todo()


class HConcat(MultipleInputs[Seq[LogicalPlan]]):
    """`concat(how="horizontal")`."""

    def _format_non_rec(self, indent: int) -> str:
        return f"{INDENT * indent}HCONCAT"

    _format_rec = todo()


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


class Rename(LpFunction):
    __slots__ = ("new", "old")
    old: Seq[str]
    new: Seq[str]

    @property
    def mapping(self) -> dict[str, str]:
        # Trying to avoid adding mutable fields
        return dict(zip_strict(self.old, self.new))

    def __repr__(self) -> str:
        return f"RENAME {self.mapping!r}"
