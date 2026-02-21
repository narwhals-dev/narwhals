"""Query plan representation that has a fully resolved schema.

Roughly based on [`polars_plan::plans::ir::IR`].

Each sub-plan that modifies the schema will have a `output_schema` field,
representing the **input for the next plan**.

Note:
    `polars` uses a mix of name(s): `output_schema`, `columns`, `schema`
    to refer to what I'm going to exclusively call `output_schema`.

[`polars_plan::plans::ir::IR`]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/ir/mod.rs#L36-L164
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from narwhals._plan import expressions as ir
from narwhals._plan._immutable import Immutable
from narwhals._plan.plans._base import _BasePlan
from narwhals._plan.schema import freeze_schema
from narwhals._plan.typing import Seq
from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import TypeAlias

    from narwhals._plan._expr_ir import NamedIR
    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.options import (
        ExplodeOptions,
        JoinAsofOptions,
        JoinOptions,
        SortMultipleOptions,
        UniqueOptions,
        UnpivotOptions,
    )
    from narwhals._plan.schema import FrozenSchema


_Fwd: TypeAlias = "ResolvedPlan"
_InputsT = TypeVar("_InputsT", bound="Seq[ResolvedPlan]")
ResolvedFunctionT = TypeVar(
    "ResolvedFunctionT", bound="ResolvedFunction", default="ResolvedFunction"
)


class ResolvedPlan(_BasePlan[_Fwd], _root=True):
    def iter_left(self) -> Iterator[ResolvedPlan]:
        for input in self.iter_inputs():
            yield from input.iter_left()
        yield self

    # NOTE: Could probably de-dup `return self.output_schema` w/ metaprogramming
    # but will need to make do for now
    @property
    def schema(self) -> FrozenSchema:
        """Get the schema at this stage in the plan.

        Nodes that change the schema store the result in `output_schema`.

        All others refer to the schema of their first input.
        """
        return next(self.iter_inputs()).schema

    def rename(self, mapping: Mapping[str, str]) -> Select:
        schema = self.schema
        exprs = tuple(ir.named_ir(mapping.get(old, old), ir.col(old)) for old in schema)
        output_schema = freeze_schema(zip((e.name for e in exprs), schema.values()))
        return Select(input=self, exprs=exprs, output_schema=output_schema)


class Scan(ResolvedPlan, has_inputs=False):
    def iter_right(self) -> Iterator[ResolvedPlan]:
        yield self

    def iter_inputs(self) -> Iterator[ResolvedPlan]:
        yield from ()


class SingleInput(ResolvedPlan, has_inputs=True):
    __slots__ = ("input",)
    input: ResolvedPlan

    def iter_right(self) -> Iterator[ResolvedPlan]:
        yield self
        yield from self.input.iter_right()

    def iter_inputs(self) -> Iterator[ResolvedPlan]:
        yield self.input


class MultipleInputs(ResolvedPlan, Generic[_InputsT], has_inputs=True):
    __slots__ = ("inputs",)
    inputs: _InputsT

    def iter_right(self) -> Iterator[ResolvedPlan]:
        yield self
        for input in reversed(self.inputs):
            yield from input.iter_right()

    def iter_inputs(self) -> Iterator[ResolvedPlan]:
        yield from self.inputs


class Sink(SingleInput): ...


class Collect(Sink): ...


class SinkFile(Sink):
    __slots__ = ("target",)
    target: str


class SinkParquet(SinkFile): ...


class ScanFile(Scan):
    __slots__ = ("output_schema", "source")
    source: str
    output_schema: FrozenSchema
    """Schema of the file.

    Equivalent to `IR::Scan.file_info.schema`
    """

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema


class ScanCsv(ScanFile): ...


# NOTE: Not sure about the impl variant
# If we already have the schema, then no need to propagate?
class ScanParquet(ScanFile): ...


class ScanDataFrame(Scan):
    __slots__ = ("frame", "output_schema")
    frame: DataFrame[Any, Any]
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        yield from (id(self.frame), self.output_schema)

    def __str__(self) -> str:
        # not redoing, just avoiding `DataFrame.__repr__`
        return f"<{type(self).__module__}.{type(self).__name__} todo>"


# `IR::SimpleProjection`
class SelectNames(SingleInput):
    __slots__ = ("output_schema",)
    output_schema: FrozenSchema

    @property
    def names(self) -> tuple[str, ...]:
        return self.output_schema.names

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema


class Select(SingleInput):
    """Roughly `prepare_projection` + FrozenSchema update."""

    __slots__ = ("exprs", "output_schema")
    exprs: Seq[NamedIR]
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema


# `IR::HStack`
class WithColumns(SingleInput):
    __slots__ = ("exprs", "output_schema")
    exprs: Seq[NamedIR]
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema


class Filter(SingleInput):
    __slots__ = ("predicate",)
    predicate: NamedIR


class Slice(SingleInput):
    __slots__ = ("length", "offset")
    offset: int
    length: int | None  # probably should have `int`


class Sort(SingleInput):
    __slots__ = ("by", "options")
    by: Seq[str]
    """Deviation from polars, resolved selector names."""
    options: SortMultipleOptions


# `UniqueBy` might be composable from other steps?
class Unique(SingleInput):
    __slots__ = ("options", "subset")
    subset: Seq[str] | None
    options: UniqueOptions


class MapFunction(SingleInput, Generic[ResolvedFunctionT]):
    __slots__ = ("function",)
    function: ResolvedFunctionT

    @property
    def schema(self) -> FrozenSchema:
        return self.function.output_schema


class Join(MultipleInputs[tuple[ResolvedPlan, ResolvedPlan]]):
    __slots__ = ("left_on", "options", "output_schema", "right_on")
    left_on: Seq[str]
    right_on: Seq[str]
    options: JoinOptions
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema


class JoinAsof(MultipleInputs[tuple[ResolvedPlan, ResolvedPlan]]):
    __slots__ = ("left_on", "options", "output_schema", "right_on")
    left_on: str
    right_on: str
    options: JoinAsofOptions
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema


class GroupBy(SingleInput):
    """`GroupByResolver._from_grouper`."""

    __slots__ = ("aggs", "keys", "output_schema")
    keys: Seq[NamedIR]
    aggs: Seq[NamedIR]
    output_schema: FrozenSchema  # GroupByResolver._schema

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema


class GroupByNames(SingleInput):
    """`DataFrameGroupBy.by_names`/`not resolver.requires_projection()`."""

    __slots__ = ("aggs", "key_names", "output_schema")
    key_names: Seq[str]
    aggs: Seq[NamedIR]
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema


class VConcat(MultipleInputs[Seq[ResolvedPlan]]):
    # - Only 1 attribute we have carries over from UnionArgs -> UnionOptions
    #   - https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/dsl/options/mod.rs#L417-L429
    # - interestingly, polars doesn't store a schema
    #   - the schemas of inputs are rewritten, based on `VConcatOptions`
    #   - so the schema of this node is the same schema all of the others have (after)
    __slots__ = ("maintain_order",)
    maintain_order: bool


class HConcat(MultipleInputs[Seq[ResolvedPlan]]):
    __slots__ = ("output_schema",)
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema


# `MapFunction.function: FunctionIR`
class ResolvedFunction(Immutable):
    __slots__ = ("output_schema",)
    output_schema: FrozenSchema


class RowIndex(ResolvedFunction):
    __slots__ = ("name",)
    name: str


class Unnest(ResolvedFunction):
    # polars doesn't store the schema on this one,
    # but implementing for pyarrow made we wish we had it
    __slots__ = ("columns",)
    columns: Seq[str]


class Explode(ResolvedFunction):
    __slots__ = ("columns", "options")
    columns: Seq[str]
    options: ExplodeOptions


class Unpivot(ResolvedFunction):
    __slots__ = ("index", "on", "options")
    on: Seq[str]
    index: Seq[str]
    options: UnpivotOptions
