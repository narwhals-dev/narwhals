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
class LogicalPlan(Immutable): ...


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


class Select(SingleInput):
    __slots__ = ("exprs",)
    exprs: Seq[ExprIR]
    # NOTE: Contains a `should_broadcast` flag, but AFAICT
    # is only replaced with `False` during optimization (not when building the plan)
    # `options: ProjectionOptions`


# `DslPlan::HStack`
class WithColumns(SingleInput):
    __slots__ = ("exprs",)
    exprs: Seq[ExprIR]
    # NOTE: Same `ProjectionOptions` comment as `Select`


class Filter(SingleInput):
    __slots__ = ("predicate",)
    predicate: ExprIR


class GroupBy(SingleInput):
    __slots__ = ("aggs", "keys")
    keys: Seq[ExprIR]
    aggs: Seq[ExprIR]


# `DslPlan::Distinct`
class Unique(SingleInput):
    __slots__ = ("options", "subset")
    subset: Seq[SelectorIR] | None
    options: UniqueOptions


class Sort(SingleInput):
    __slots__ = ("by", "options")
    by: Seq[SelectorIR]
    options: SortMultipleOptions


class Slice(SingleInput):
    __slots__ = ("length", "offset")
    offset: int
    length: int | None


class Join(LogicalPlan):
    __slots__ = ("how", "input_left", "input_right", "left_on", "right_on", "suffix")
    # NOTE: Might be nicer to have `inputs: tuple[LogicalPlan, LogicalPlan]`
    input_left: LogicalPlan
    input_right: LogicalPlan
    left_on: Seq[str]
    right_on: Seq[str]
    options: JoinOptions


class MapFunction(SingleInput):
    # `polars` says this is for UDFs, but uses it for: `Rename`, `RowIndex`, `Unnest`, `Explode`
    __slots__ = ("function",)
    function: LpFunction


# `DslPlan::Union`
class VConcat(LogicalPlan):
    """`concat(how= "vertical" | "diagonal")`."""

    __slots__ = ("inputs", "options")
    inputs: Seq[LogicalPlan]
    options: VConcatOptions


class HConcat(LogicalPlan):
    """`concat(how="horizontal")`."""

    __slots__ = ("inputs", "strict")
    inputs: Seq[LogicalPlan]
    strict: bool
    """Require all `inputs` to be the same height, raising an error if not, default False."""


# NOTE: `DslFunction`
class LpFunction(Immutable): ...


class Explode(LpFunction):
    __slots__ = ("columns", "options")
    columns: SelectorIR
    options: ExplodeOptions


class Unnest(LpFunction):
    __slots__ = ("columns",)
    columns: SelectorIR


class RowIndex(LpFunction):
    __slots__ = ("name",)
    name: str


class Rename(LpFunction):
    __slots__ = ("new", "old")
    old: Seq[str]
    new: Seq[str]

    @property
    def mapping(self) -> dict[str, str]:
        # Trying to avoid adding mutable fields
        return dict(zip_strict(self.old, self.new))
