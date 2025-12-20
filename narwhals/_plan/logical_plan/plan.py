from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._plan._immutable import Immutable
from narwhals._plan.schema import freeze_schema
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
    from narwhals._plan.typing import Seq


# `DslPlan`
# TODO @dangotbanned: Add `LogicalPlan`s for ops in `nw.*Frame`, that aren't yet in `nwp.*Frame`
class LogicalPlan(Immutable):
    """Misc notes.

    - `LazyFrame.collect` -> `LazyFrame.collect_with_engine` -> `DslPlan::Sink(self.logical_plan, SinkType::Memory)`
      - Adding the collect to the plan so far
      - https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-lazy/src/frame/mod.rs#L631-L637
    - `LazyFrame.to_alp_optimized`
      - https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-lazy/src/frame/mod.rs#L666
    - Here's the big boi, single function, handling `plan::DslPlan` -> `ir::IR`
      - `polars_plan::plans::conversion::dsl_to_ir::to_alp_impl`
      - https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs#L102-L1375
      - Recurses, calling on each input in a plan
      - Expansion is happening at this stage
      - `resolve_group_by` as well
    """


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

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        # NOTE: Deferring how to handle the hash *for now*
        # Currently, every `DataFrameSource` will have a unique psuedo-hash
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


class Join(LogicalPlan):
    __slots__ = ("how", "input_left", "input_right", "left_on", "right_on", "suffix")
    # NOTE: Might be nicer to have `inputs: tuple[LogicalPlan, LogicalPlan]`
    input_left: LogicalPlan
    input_right: LogicalPlan
    left_on: Seq[str]
    right_on: Seq[str]
    options: JoinOptions

    def __repr__(self) -> str:
        how = self.options.how.upper()
        if how == "CROSS":
            return f"{how} JOIN"
        return f"{how} JOIN:\nLEFT PLAN ON: {list(self.left_on)!r}\nRIGHT PLAN ON: {list(self.right_on)!r}"


class MapFunction(SingleInput):
    # `polars` says this is for UDFs, but uses it for: `Rename`, `RowIndex`, `Unnest`, `Explode`
    __slots__ = ("function",)
    function: LpFunction

    def __repr__(self) -> str:
        return f"{self.function!r}"


# `DslPlan::Union`
class VConcat(LogicalPlan):
    """`concat(how= "vertical" | "diagonal")`."""

    __slots__ = ("inputs", "options")
    inputs: Seq[LogicalPlan]
    options: VConcatOptions

    def __repr__(self) -> str:
        return "UNION"


class HConcat(LogicalPlan):
    """`concat(how="horizontal")`."""

    __slots__ = ("inputs", "strict")
    inputs: Seq[LogicalPlan]
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
