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
from narwhals._plan.compliant.typing import Native
from narwhals._plan.plans._base import _BasePlan
from narwhals._plan.plans.typing import FrameT
from narwhals._plan.schema import FrozenSchema
from narwhals._plan.typing import ClosedKwds, Seq
from narwhals._typing_compat import TypeVar
from narwhals._utils import zip_strict
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import TypeAlias

    from narwhals._plan._expr_ir import NamedIR
    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
    from narwhals._plan.dataframe import DataFrame  # noqa: F401
    from narwhals._plan.options import (
        ExplodeOptions,
        JoinAsofOptions,
        JoinOptions,
        SortMultipleOptions,
        UniqueOptions,
        UnpivotOptions,
    )
    from narwhals._plan.plans.visitors import ResolvedToCompliant
    from narwhals._utils import Implementation

Incomplete: TypeAlias = Any
_Fwd: TypeAlias = "ResolvedPlan"
_InputsT = TypeVar("_InputsT", bound="Seq[ResolvedPlan]")
RpFunctionT_co = TypeVar(
    "RpFunctionT_co", bound="RpFunction", default="RpFunction", covariant=True
)

# TODO @dangotbanned: Figure out how to integrate this *without* obliterating typing
NonSink: TypeAlias = "Scan | SingleInput | MultipleInputs[Any]"
"""Any `ResolvedPlan` node which evaluates to a `CompliantLazyFrame`."""


class ResolvedPlan(_BasePlan[_Fwd], _root=True):
    def iter_left(self) -> Iterator[ResolvedPlan]:  # pragma: no cover
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

    def rename(self, mapping: Mapping[str, str]) -> Select:  # pragma: no cover
        schema = self.schema
        exprs = tuple(ir.NamedIR(mapping.get(old, old), ir.col(old)) for old in schema)
        output_schema = FrozenSchema(zip((e.name for e in exprs), schema.values()))
        return Select(input=self, exprs=exprs, output_schema=output_schema)

    # or maybe `execute`?
    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:
        """Evaluate this `ResolvedPlan`.

        This is not the same as materializing, unless called on a `Sink` node
        (e.g. `Collect`).

        For example, you could call evaluate on the resolved plan of `lf.select().with_columns()`
        which should not materialize.
        """
        msg = f"TODO: `{type(self).__name__}.evaluate`"
        raise NotImplementedError(msg)


class Scan(ResolvedPlan, has_inputs=False):
    def iter_right(self) -> Iterator[ResolvedPlan]:  # pragma: no cover
        yield self

    def iter_inputs(self) -> Iterator[ResolvedPlan]:  # pragma: no cover
        yield from ()


# TODO @dangotbanned: The inputs need to be typed as not allowing a `Sink`
# This is enforced at `LogicalPlan._sink`, but not having the typing means everything needs to check
# if it got a lazyframe or none or dataframe
class SingleInput(ResolvedPlan, has_inputs=True):
    __slots__ = ("input",)
    input: ResolvedPlan

    def iter_right(self) -> Iterator[ResolvedPlan]:  # pragma: no cover
        yield self
        yield from self.input.iter_right()

    def iter_inputs(self) -> Iterator[ResolvedPlan]:
        yield self.input


class MultipleInputs(ResolvedPlan, Generic[_InputsT], has_inputs=True):
    __slots__ = ("inputs",)
    inputs: _InputsT

    def iter_right(self) -> Iterator[ResolvedPlan]:  # pragma: no cover
        yield self
        for input in reversed(self.inputs):
            yield from input.iter_right()

    def iter_inputs(self) -> Iterator[ResolvedPlan]:  # pragma: no cover
        yield from self.inputs


# TODO @dangotbanned: Don't define `ResolvePlan.evaluate`
# - Also can't do `SingleInput.evaluate`, unless this stops inheriting
# - Possibly add the concept of *transforming* nodes?
class Sink(SingleInput):
    def evaluate(
        self, evaluator: ResolvedToCompliant[Any], /
    ) -> Incomplete:  # pragma: no cover
        from narwhals._plan._dispatch import _pascal_to_snake_case

        evaluator_name = type(evaluator).__name__
        node_name = type(self).__name__
        method = _pascal_to_snake_case(node_name)
        msg = (
            f"`Sink` nodes do not support `evaluate()`.\n"
            f"Instead of calling: `{node_name}.evaluate({evaluator_name})\n"
            f"Try: `{evaluator_name}.{method}({node_name}, ...)"
        )
        raise InvalidOperationError(msg)


class Collect(Sink):
    __slots__ = ("kwds",)
    kwds: ClosedKwds


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


class ScanCsv(ScanFile):
    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.scan_csv(self)


class ScanParquet(ScanFile):
    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.scan_parquet(self)


class ScanFrame(Scan, Generic[FrameT]):
    __slots__ = ("frame", "output_schema")
    frame: FrameT
    output_schema: FrozenSchema

    @property
    def implementation(self) -> Implementation:  # pragma: no cover
        return self.frame.implementation

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema

    def __str__(self) -> str:
        # not redoing, just avoiding `*Frame.__repr__`
        return f"<{type(self).__module__}.{type(self).__name__} todo>"


class ScanDataFrame(ScanFrame["DataFrame[Any, Any]"]):
    @property
    def __immutable_values__(self) -> Iterator[Any]:  # pragma: no cover
        yield from (id(self.frame), self.output_schema)

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.scan_dataframe(self)


# TODO @dangotbanned: `[override]` is because this is the **only correct** node
# - All of the others need to propagate `Native` from `LogicalPlan`
# - Which requires making `LogicalPlan` generic
#   - This also removes the need for the `LazyFrame._compiant: CompliantLazyFrame[Native]` hack
class ScanLazyFrame(ScanFrame["CompliantLazyFrame[Native]"], Generic[Native]):
    def evaluate(  # type: ignore[override]
        self, evaluator: ResolvedToCompliant[Native] | Incomplete, /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.scan_lazyframe(self)


# `IR::SimpleProjection`
class SelectNames(SingleInput):
    __slots__ = ("output_schema",)
    output_schema: FrozenSchema

    @property
    def names(self) -> tuple[str, ...]:  # pragma: no cover
        return self.output_schema.names

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.select_names(self)


class Select(SingleInput):
    """Roughly `prepare_projection` + FrozenSchema update."""

    __slots__ = ("exprs", "output_schema")
    exprs: Seq[NamedIR]
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.select(self)


# `IR::HStack`
class WithColumns(SingleInput):
    __slots__ = ("exprs", "output_schema")
    exprs: Seq[NamedIR]
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:
        return self.output_schema

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.with_columns(self)


class Filter(SingleInput):
    __slots__ = ("predicate",)
    predicate: NamedIR

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.filter(self)


class Slice(SingleInput):
    __slots__ = ("length", "offset")
    offset: int
    length: int | None  # probably should have `int`

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.slice(self)


class Sort(SingleInput):
    __slots__ = ("by", "options")
    by: Seq[str]
    """Deviation from polars, resolved selector names."""
    options: SortMultipleOptions

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.sort(self)


class Unique(SingleInput):
    __slots__ = ("options", "subset")
    subset: Seq[str] | None
    options: UniqueOptions

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.unique(self)


class UniqueBy(Unique):
    # NOTE: May want to provide a rewrite *from* this
    __slots__ = ("order_by",)
    order_by: Seq[str]

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.unique_by(self)


class MapFunction(SingleInput, Generic[RpFunctionT_co]):
    __slots__ = ("function",)
    # NOTE: https://discuss.python.org/t/make-replace-stop-interfering-with-variance-inference/96092
    function: RpFunctionT_co  # type: ignore[misc]

    @property
    def schema(self) -> FrozenSchema:  # pragma: no cover
        return self.function.output_schema

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.map_function(self)


class Join(MultipleInputs[tuple[ResolvedPlan, ResolvedPlan]]):
    __slots__ = ("left_on", "options", "output_schema", "right_on")
    left_on: Seq[str]
    right_on: Seq[str]
    options: JoinOptions
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:  # pragma: no cover
        return self.output_schema

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.join(self)


class JoinAsof(MultipleInputs[tuple[ResolvedPlan, ResolvedPlan]]):
    __slots__ = ("left_on", "options", "output_schema", "right_on")
    left_on: str
    right_on: str
    options: JoinAsofOptions
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:  # pragma: no cover
        return self.output_schema

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.join_asof(self)


class GroupBy(SingleInput):
    """`GroupByResolver._from_grouper`."""

    __slots__ = ("aggs", "keys", "output_schema")
    keys: Seq[NamedIR]
    aggs: Seq[NamedIR]
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:  # pragma: no cover
        return self.output_schema

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.group_by(self)


class GroupByNames(SingleInput):
    """`DataFrameGroupBy.by_names`/`not resolver.requires_projection()`."""

    __slots__ = ("aggs", "key_names", "output_schema")
    key_names: Seq[str]
    aggs: Seq[NamedIR]
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:  # pragma: no cover
        return self.output_schema

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.group_by_names(self)


class VConcat(MultipleInputs[Seq[ResolvedPlan]]):
    # - Only 1 attribute we have carries over from UnionArgs -> UnionOptions
    #   - https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/dsl/options/mod.rs#L417-L429
    # - interestingly, polars doesn't store a schema
    #   - the schemas of inputs are rewritten, based on `VConcatOptions`
    #   - so the schema of this node is the same schema all of the others have (after)
    __slots__ = ("maintain_order",)
    maintain_order: bool

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.concat_vertical(self)


class HConcat(MultipleInputs[Seq[ResolvedPlan]]):
    __slots__ = ("output_schema",)
    output_schema: FrozenSchema

    @property
    def schema(self) -> FrozenSchema:  # pragma: no cover
        return self.output_schema

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], /
    ) -> CompliantLazyFrame[Native]:  # pragma: no cover
        return evaluator.concat_horizontal(self)


# `MapFunction.function: FunctionIR`
class RpFunction(Immutable):
    __slots__ = ("output_schema",)
    output_schema: FrozenSchema

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], plan: MapFunction[Incomplete], /
    ) -> CompliantLazyFrame[Native]:
        """Evaluate this `ResolvedPlan`."""
        msg = f"TODO: `{type(self).__name__}.evaluate`"
        raise NotImplementedError(msg)


class RowIndex(RpFunction):
    __slots__ = ("name",)
    name: str

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], plan: MapFunction[Incomplete], /
    ) -> CompliantLazyFrame[Native]:
        return evaluator.with_row_index(plan)  # pragma: no cover


class RowIndexBy(RowIndex):
    __slots__ = ("order_by",)
    order_by: Seq[str]

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], plan: MapFunction[RowIndexBy], /
    ) -> CompliantLazyFrame[Native]:
        return evaluator.with_row_index_by(plan)  # pragma: no cover


class Rename(RpFunction):
    """Rename when the backend supports this operation natively."""

    __slots__ = ("new", "old")
    old: Seq[str]
    new: Seq[str]

    @property
    def mapping(self) -> dict[str, str]:
        """Return a new dictionary representing `{old: new}`."""
        return dict(zip_strict(self.old, self.new))  # pragma: no cover

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], plan: MapFunction[Rename], /
    ) -> CompliantLazyFrame[Native]:
        return evaluator.rename(plan)  # pragma: no cover


class Unnest(RpFunction):
    # polars doesn't store the schema on this one,
    # but implementing for pyarrow made we wish we had it
    __slots__ = ("columns",)
    columns: Seq[str]

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], plan: MapFunction[Unnest], /
    ) -> CompliantLazyFrame[Native]:
        return evaluator.unnest(plan)  # pragma: no cover


class Explode(RpFunction):
    __slots__ = ("columns", "options")
    columns: Seq[str]
    options: ExplodeOptions

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], plan: MapFunction[Explode], /
    ) -> CompliantLazyFrame[Native]:
        return evaluator.explode(plan)  # pragma: no cover


class Unpivot(RpFunction):
    __slots__ = ("index", "on", "options")
    on: Seq[str]
    index: Seq[str]
    options: UnpivotOptions

    def evaluate(
        self, evaluator: ResolvedToCompliant[Native], plan: MapFunction[Unpivot], /
    ) -> CompliantLazyFrame[Native]:
        return evaluator.unpivot(plan)  # pragma: no cover
