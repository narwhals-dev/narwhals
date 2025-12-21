from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from narwhals._plan._immutable import Immutable
from narwhals._plan.schema import freeze_schema
from narwhals._plan.typing import Seq
from narwhals._typing_compat import TypeVar
from narwhals._utils import qualified_type_name, zip_strict

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.expressions import ExprIR, SelectorIR
    from narwhals._plan.options import (
        ExplodeOptions,
        JoinOptions,
        SortMultipleOptions,
        UniqueOptions,
        VConcatOptions,
    )
    from narwhals._plan.schema import FrozenSchema

_InputsT = TypeVar("_InputsT", bound="Seq[LogicalPlan]")


# TODO @dangotbanned: Add `LogicalPlan`s for ops in `nw.*Frame`, that aren't yet in `nwp.*Frame`
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

    def iter_left(self) -> Iterator[LogicalPlan]:
        """Yield nodes root->leaf."""
        msg = f"TODO: `{type(self).__name__}.iter_left`"
        raise NotImplementedError(msg)

    def iter_right(self) -> Iterator[LogicalPlan]:
        """Yield nodes leaf->root."""
        msg = f"TODO: `{type(self).__name__}.iter_right`"
        raise NotImplementedError(msg)


# TODO @dangotbanned: Careful think about how (non-scan) source nodes should work
# - Schema only?
# - Different for eager vs lazy?
class DataFrameScan(LogicalPlan):
    __slots__ = ("df", "schema")
    df: DataFrame[Any, Any]
    schema: FrozenSchema

    # NOTE: Probably want a `staticmethod`, change if nothing is needed from `cls`
    @classmethod
    def from_narwhals(cls, df: DataFrame[Any, Any]) -> DataFrameScan:
        obj = cls.__new__(cls)
        object.__setattr__(obj, "df", df.clone())
        object.__setattr__(obj, "schema", freeze_schema(df.schema))
        return obj

    def iter_left(self) -> Iterator[LogicalPlan]:
        yield self

    def iter_right(self) -> Iterator[LogicalPlan]:
        yield self

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        # NOTE: Deferring how to handle the hash *for now*
        # Currently, every `DataFrameSource` will have a unique pseudo-hash
        # Caching a native table seems like a non-starter, once `pandas` enters the party
        yield from (id(self.df), self.schema)

    def __repr__(self) -> str:
        names = self.schema.names
        n_columns = len(names)
        if n_columns > 4:
            it = (f'"{name}"' for name in names[:4])
            s = ", ".join((*it, "..."))
        elif n_columns == 0:
            s = ""
        else:
            s = ", ".join(f'"{name}"' for name in names)
        return f"DF [{s}]; {n_columns} COLUMNS"

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"df=nw.DataFrame[{qualified_type_name(self.df.to_native())}](...), "
            f"schema={self.schema!s})"
        )


class SingleInput(LogicalPlan):
    __slots__ = ("input",)
    input: LogicalPlan

    def iter_left(self) -> Iterator[LogicalPlan]:
        yield from self.input.iter_left()
        yield self

    def iter_right(self) -> Iterator[LogicalPlan]:
        yield self
        yield from self.input.iter_right()


class MultipleInputs(LogicalPlan, Generic[_InputsT]):
    __slots__ = ("inputs",)
    inputs: _InputsT

    def iter_left(self) -> Iterator[LogicalPlan]:
        for input in self.inputs:
            yield from input.iter_left()
        yield self

    def iter_right(self) -> Iterator[LogicalPlan]:
        yield self
        for input in reversed(self.inputs):
            yield from input.iter_right()


class Sink(SingleInput):
    """Terminal nodes."""


class Collect(Sink): ...


class SinkParquet(Sink):
    __slots__ = ("target",)
    target: str
    """`file: str | Path | BytesIO` on main.

    Not sure `BytesIO` makes sense here.
    """


class Select(SingleInput):
    __slots__ = ("exprs",)
    exprs: Seq[ExprIR]

    # NOTE: Contains a `should_broadcast` flag, but AFAICT
    # is only replaced with `False` during optimization (not when building the plan)
    # `options: ProjectionOptions`
    def __repr__(self) -> str:
        return f"SELECT {list(self.exprs)!r}"


# `DslPlan::HStack`
class WithColumns(SingleInput):
    __slots__ = ("exprs",)
    exprs: Seq[ExprIR]
    # NOTE: Same `ProjectionOptions` comment as `Select`

    def __repr__(self) -> str:
        return f"WITH_COLUMNS:\n{list(self.exprs)!r}"


class Filter(SingleInput):
    __slots__ = ("predicate",)
    predicate: ExprIR

    def __repr__(self) -> str:
        return f"FILTER {self.predicate!r}\nFROM"


# TODO @dangotbanned: Add repr
class GroupBy(SingleInput):
    __slots__ = ("aggs", "keys")
    keys: Seq[ExprIR]
    aggs: Seq[ExprIR]

    def __repr__(self) -> str:
        msg = f"TODO: `{type(self).__name__}.__repr__`"
        raise NotImplementedError(msg)


# `DslPlan::Distinct`
class Unique(SingleInput):
    __slots__ = ("options", "subset")
    subset: Seq[SelectorIR] | None
    options: UniqueOptions

    def __repr__(self) -> str:
        opts = self.options
        return f"UNIQUE[maintain_order: {opts.maintain_order}, keep: {opts.keep}] BY {self.subset!r}"


# TODO @dangotbanned: Add repr
class Sort(SingleInput):
    __slots__ = ("by", "options")
    by: Seq[SelectorIR]
    options: SortMultipleOptions

    def __repr__(self) -> str:
        msg = f"TODO: `{type(self).__name__}.__repr__`"
        raise NotImplementedError(msg)


class Slice(SingleInput):
    __slots__ = ("length", "offset")
    offset: int
    length: int | None

    def __repr__(self) -> str:
        return f"SLICE[offset: {self.offset}, len: {self.length}]"


class MapFunction(SingleInput):
    # `polars` says this is for UDFs, but uses it for: `Rename`, `RowIndex`, `Unnest`, `Explode`
    __slots__ = ("function",)
    function: LpFunction

    def __repr__(self) -> str:
        return f"{self.function!r}"


class Join(MultipleInputs[tuple[LogicalPlan, LogicalPlan]]):
    """Join two tables in an SQL-like fashion."""

    __slots__ = ("how", "left_on", "right_on", "suffix")
    left_on: Seq[str]
    right_on: Seq[str]
    options: JoinOptions

    @property
    def input_left(self) -> LogicalPlan:
        return self.inputs[0]

    @property
    def input_right(self) -> LogicalPlan:
        return self.inputs[1]

    def __repr__(self) -> str:
        how = self.options.how.upper()
        if how == "CROSS":
            return f"{how} JOIN"
        return f"{how} JOIN:\nLEFT PLAN ON: {list(self.left_on)!r}\nRIGHT PLAN ON: {list(self.right_on)!r}"


# `DslPlan::Union`
class VConcat(MultipleInputs[Seq[LogicalPlan]]):
    """`concat(how= "vertical" | "diagonal")`."""

    __slots__ = ("options",)
    options: VConcatOptions

    def __repr__(self) -> str:
        return "UNION"


class HConcat(MultipleInputs[Seq[LogicalPlan]]):
    """`concat(how="horizontal")`."""

    __slots__ = ("strict",)
    strict: bool
    """Require all `inputs` to be the same height, raising an error if not, default False."""

    def __repr__(self) -> str:
        return "HCONCAT"


# NOTE: `DslFunction`
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
