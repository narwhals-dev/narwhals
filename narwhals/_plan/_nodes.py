"""Traversal of `ExprIR`, maybe `*Plan` eventually too.

## Implementation Notes
There's a couple *big ideas* in here, but they could broadly fit into:

> Try to encode as much as possible **into the classes**

With a helping hand from [field specifiers] ([PEP 681]), all expressions can be fully traversed
**and** transformed without a single runtime type check.

[Field specifiers]: https://typing.python.org/en/latest/spec/dataclasses.html#field-specifiers
[PEP 681]: https://peps.python.org/pep-0681/#field-specifiers
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeVar, final, overload

from narwhals._plan.exceptions import combination_mixed_multi_output_error

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from typing_extensions import Self, TypeAlias

    from narwhals._plan._expansion import Expander
    from narwhals._plan._expr_ir import ExprIR
    from narwhals._plan.typing import MapIR, OneOrSeq, Seq


class IsScalar(enum.Enum):
    OBSERVE = enum.auto()
    SKIP = enum.auto()


_OBSERVE: Final = IsScalar.OBSERVE
_SKIP: Final = IsScalar.SKIP
_Observe: TypeAlias = Literal[IsScalar.OBSERVE]
_Skip: TypeAlias = Literal[IsScalar.SKIP]

NodeT = TypeVar("NodeT")
# TODO @dangotbanned: Rename `Get` -> `Storage`?
Get = TypeVar("Get", covariant=True)  # noqa: PLC0105
IsScalarT = TypeVar(  # noqa: PLC0105
    "IsScalarT", _Observe, _Skip, covariant=True
)


def node(*, observe_scalar: bool = True) -> Any:
    """Declare that a field stores a single node.

    Arguments:
        observe_scalar: If True (default), include the node when evaluating `is_scalar()`.

            The result is a logical AND over observed nodes.

    Examples:
        If an expression contains another expression, we mark the field with a [field specifier]:
        >>> from narwhals._plan import expressions as ir
        >>> class Alias(ir.ExprIR):
        ...     __slots__ = ("expr", "name")
        ...     expr: ir.ExprIR = node()
        ...     name: str
        ...
        ...     def __repr__(self) -> str:
        ...         return f"{self.expr!r}.alias({self.name!r})"

        Which populates `__expr_ir_nodes__` at the class-level:
        >>> Alias.__expr_ir_nodes__
        ExprTraverser[1]
            expr: ExprIR = node()

        An instance can use the field as normal:
        >>> aliased = Alias(expr=ir.col("a"), name="b")
        >>> aliased
        col('a').alias('b')
        >>> aliased.expr
        col('a')

        But the field is understood as being traversable:
        >>> first, second = aliased.iter_left()
        >>> first
        col('a')
        >>> second
        col('a').alias('b')

        By default a `node` is *observed* in `is_scalar()`:
        >>> aliased.is_scalar()
        False

        Meaning we depend on it to answer that question:
        >>> lit_aliased = aliased.__replace__(expr=ir.lit(1))
        >>> lit_aliased
        lit(int: 1).alias('b')
        >>> lit_aliased.is_scalar()
        True

    [field specifier]: https://typing.python.org/en/latest/spec/dataclasses.html#field-specifiers
    """
    return SingleExpr(_OBSERVE) if observe_scalar else SingleExpr(_SKIP)


def nodes() -> Any:
    """Declare that a field stores multiple nodes.

    Unlike `node`, these are never used for `is_scalar()`.

    ## Examples
    If an expression accepts a variable number of expressions, we mark the variadic field(s)
    with a [field specifier]:

    >>> from narwhals._plan import expressions as ir
    >>> from narwhals._plan.typing import Seq
    >>> class Over(ir.ExprIR):
    ...     __slots__ = ("expr", "partition_by")
    ...     expr: ir.ExprIR = node(observe_scalar=False)
    ...     partition_by: Seq[ir.ExprIR] = nodes()
    ...
    ...     def __repr__(self) -> str:
    ...         return f"{self.expr!r}.over({list(self.partition_by)!r})"

    Which populates `__expr_ir_nodes__` at the class-level:
    >>> Over.__expr_ir_nodes__
    ExprTraverser[2]
        expr: ExprIR = node(observe_scalar=False)
        partition_by: Seq[ExprIR] = nodes()

    An instance can use the field as normal:
    >>> from narwhals._plan.expressions.aggregation import First
    >>> root = First(expr=ir.col("a"))
    >>> variadic = tuple(ir.col(s) for s in "bcdef")
    >>> over = Over(expr=root, partition_by=variadic)
    >>> over
    col('a').first().over([col('b'), col('c'), col('d'), col('e'), col('f')])

    >>> over.partition_by
    (col('b'), col('c'), col('d'), col('e'), col('f'))

    But the field is understood as being traversable:
    >>> it = over.iter_right()
    >>> next(it) is over
    True
    >>> next(it) == ir.col("f")
    True
    >>> e, d, c, b, *rest = it
    >>> e, d, c, b
    (col('e'), col('d'), col('c'), col('b'))
    >>> rest == list(over.expr.iter_right())
    True

    [field specifier]: https://typing.python.org/en/latest/spec/dataclasses.html#field-specifiers
    """
    return MultipleExpr()


def into_expr_node(
    assigned_name: str, node: ExprNodeAny, cls_name: str
) -> MultipleExpr | SingleExpr[_Observe] | SingleExpr[_Skip]:
    if not isinstance(node, (SingleExpr, MultipleExpr)):
        raise _into_expr_node_error(assigned_name, node, cls_name)
    return node.with_name(assigned_name)


# TODO @dangotbanned: (low-prio): Migrate the `*Plan` bits to use this instead
# (https://github.com/narwhals-dev/narwhals/blob/c5a624885b845a1a9fcab849296424879197f6e5/narwhals/_plan/plans/_base.py#L13-L27)
class Node(Protocol[NodeT, Get]):
    """A field that participates in graph traversal.

    Describes the field in the class (`type[NodeT]`), and how we iterate over
    instances of it (`NodeT`).
    """

    __slots__ = ("name",)
    name: str
    """The name of the field."""

    def iter_left(self, instance: NodeT, /) -> Iterator[NodeT]:
        """Yield nodes recursively from root->leaf."""
        ...

    def iter_right(self, instance: NodeT, /) -> Iterator[NodeT]:
        """Yield nodes recursively from leaf->root."""
        ...

    def with_name(self, name: str, /) -> Self:
        """Set the name the field specifier refers to.

        Arguments:
            name: The name used in the assignment of the class body.

        Notes:
            Serves a similar purpose to [`__set_name__`], but:
            - Is called at an [earlier stage] in class creation
            - Points at an instance slot (for data)

        [`__set_name__`]: https://docs.python.org/3/reference/datamodel.html#object.__set_name__
        [earlier stage]: https://docs.python.org/3/reference/datamodel.html#class-object-creation
        """
        self.name = name
        return self

    def get(self, instance: NodeT, /) -> Get:
        """Get the implementation of this node from `instance`."""
        node: Get = getattr(instance, self.name)
        return node

    def map_nodes(
        self, instance: NodeT, function: Callable[[NodeT], NodeT], /
    ) -> Get | None:
        """Call `function` on this node and return the result *iff* changed.

        The idea is to limit the size of `Immutable.__replace__(**changes)`, or skip it
        entirely if we didn't need it.
        """
        ...


class ExprNode(Node["ExprIR", Get], Protocol[Get, IsScalarT]):
    """Extensions to `Node` for `ExprIR`."""

    __slots__ = ()

    @property
    def observe_scalar(self) -> IsScalarT:
        """Include the node when evaluating `is_scalar()`."""
        ...

    def iter_output_name(self, instance: ExprIR, /) -> Iterator[ExprIR]:
        """Follow the **left-hand-side** of the graph until we can derive an output name.

        See `ExprIR.iter_output_name` for examples.
        """
        ...

    def iter_expand(self, instance: ExprIR, ctx: Expander, /) -> Iterator[ExprIR]:
        """Yield the expression(s) that the current node expands into.

        Exposes expansion - *without constraints* - as a building block for `expand_as_*`.
        """
        ...

    def expand_as_first_child(self, instance: ExprIR, ctx: Expander, /) -> Iterator[Get]:
        """Expand the first field in an expression.

        Note:
            Related to `expand_as_next_sibling`, see [first child-next sibling].

        ## Examples
        >>> import narwhals as nw
        >>> import narwhals._plan as nwp
        >>> import narwhals._plan.selectors as ncs
        >>> from tests.plan.utils import Frame
        >>> schema = {"a": nw.Int64(), "b": nw.Float64(), "c": nw.Int64()}
        >>> df = Frame.from_mapping(schema)

        When the first field is a `node()`, multiple inputs expand into multiple outputs:
        >>> list(df.project(ncs.float().sum()))
        [b=col('b').sum()]
        >>> list(df.project(ncs.integer().sum()))
        [a=col('a').sum(), c=col('c').sum()]

        Whereas `nodes()` expands within container they describe, reducing to a single output:
        >>> list(df.project(nwp.sum_horizontal(ncs.float(), ncs.integer())))
        [b=col('b').sum_horizontal([col('a'), col('c')])]

        [first child-next sibling]: https://xlinux.nist.gov/dads/HTML/binaryTreeRepofTree.html
        """
        ...

    def expand_as_next_sibling(self, instance: ExprIR, ctx: Expander, /) -> Get:
        r"""Expand a field in an expression (excluding the first).

        Note:
            Related to `expand_as_first_child`, see [first child-next sibling].

        ## Examples
        >>> import narwhals as nw
        >>> import narwhals._plan as nwp
        >>> import narwhals._plan.selectors as ncs
        >>> from tests.plan.utils import Frame
        >>> i64 = nw.Int64()
        >>> schema = {"a": i64, "b": nw.Float64(), "c": i64, "d": nw.String()}
        >>> df = Frame.from_mapping(schema)
        >>> a = nwp.col("a")

        A `node()` supports selectors in this position *iff* they select a single column:
        >>> list(df.project(a.filter(ncs.float())))
        [a=col('a').filter(col('b'))]

        >>> list(df.project(a.filter(ncs.integer())))
        Traceback (most recent call last):
        narwhals.exceptions.MultiOutputExpressionError: ...
        Got `col('a').filter(ncs.integer())`, but `ncs.integer()` expanded into 2 outputs:
          col('a')
          col('c')

        Whereas `nodes()` expand within the container they describe:
        >>> agg = nwp.col("a").first()
        >>> floats, ints = ncs.float(), ncs.integer()
        >>> expanded = df.project(
        ...     agg.over(floats).alias("p1"),
        ...     agg.over(floats, ints).alias("p3"),
        ...     agg.over(floats, ints, order_by=[ncs.string(), "a"]).alias("p3_o2"),
        ... )
        >>> print(*map(repr, expanded), sep="\n")
        p1=col('a').first().over([col('b')])
        p3=col('a').first().over([col('b'), col('a'), col('c')])
        p3_o2=col('a').first().over(partition_by=[col('b'), col('a'), col('c')], order_by=[col('d'), col('a')])

        [first child-next sibling]: https://xlinux.nist.gov/dads/HTML/binaryTreeRepofTree.html
        """
        ...


class SingleExpr(ExprNode["ExprIR", IsScalarT]):
    """Representation for `node(...)`."""

    __slots__ = ("_observe_scalar",)
    _observe_scalar: IsScalarT

    def __init__(self, observe_scalar: IsScalarT) -> None:
        # Called immediately after:
        #     `<field-name> = <field-specifier-function>(**kwds)`
        self._observe_scalar = observe_scalar

    def __repr__(self) -> str:
        if not hasattr(self, "name"):
            return f"{type(self).__name__}(observe_scalar={self.observe_scalar})"
        args = "observe_scalar=False" if self.observe_scalar is _SKIP else ""
        return f"{self.name}: ExprIR = {node.__name__}({args})"

    @property
    def observe_scalar(self) -> IsScalarT:
        return self._observe_scalar

    def iter_left(self, instance: ExprIR) -> Iterator[ExprIR]:
        yield from self.get(instance).iter_left()

    def iter_right(self, instance: ExprIR) -> Iterator[ExprIR]:
        yield from self.get(instance).iter_right()

    def iter_output_name(self, instance: ExprIR) -> Iterator[ExprIR]:
        yield from self.get(instance).iter_output_name()

    def iter_expand(self, instance: ExprIR, ctx: Expander, /) -> Iterator[ExprIR]:
        yield from self.get(instance).iter_expand(ctx)

    def expand_as_first_child(
        self, instance: ExprIR, ctx: Expander, /
    ) -> Iterator[ExprIR]:
        yield from self.iter_expand(instance, ctx)

    def expand_as_next_sibling(self, instance: ExprIR, ctx: Expander, /) -> ExprIR:
        return ctx.only(instance, self.get(instance))

    def map_nodes(self, instance: ExprIR, function: MapIR, /) -> ExprIR | None:
        before = self.get(instance)
        after = before.map_ir(function)
        return None if after == before else after


class MultipleExpr(ExprNode["Seq[ExprIR]", Literal[IsScalar.SKIP]]):
    """Representation for `nodes()`."""

    __slots__ = ()

    def __repr__(self) -> str:
        if not hasattr(self, "name"):
            return f"{type(self).__name__}()"
        return f"{self.name}: Seq[ExprIR] = {nodes.__name__}()"

    @property
    def observe_scalar(self) -> Literal[IsScalar.SKIP]:
        return _SKIP

    def iter_left(self, instance: ExprIR) -> Iterator[ExprIR]:
        for expr in self.get(instance):
            yield from expr.iter_left()

    def iter_right(self, instance: ExprIR) -> Iterator[ExprIR]:
        for expr in reversed(self.get(instance)):
            yield from expr.iter_right()

    def iter_output_name(self, instance: ExprIR) -> Iterator[ExprIR]:
        for lhs in self.get(instance)[:1]:
            yield from lhs.iter_output_name()

    def iter_expand(self, instance: ExprIR, ctx: Expander, /) -> Iterator[ExprIR]:
        for expr in self.get(instance):
            yield from expr.iter_expand(ctx)

    def expand_as_first_child(
        self, instance: ExprIR, ctx: Expander, /
    ) -> Iterator[Seq[ExprIR]]:
        yield tuple(self.iter_expand(instance, ctx))

    def expand_as_next_sibling(self, instance: ExprIR, ctx: Expander, /) -> Seq[ExprIR]:
        return tuple(self.iter_expand(instance, ctx))

    def map_nodes(self, instance: ExprIR, function: MapIR, /) -> Seq[ExprIR] | None:
        if before := self.get(instance):
            after = tuple(e.map_ir(function) for e in before)
            return None if after == before else after
        return None


# NOTE: This is very intentional typing, be careful if changing in the future:
# - Do docstrings for `ExprNode` methods show in an IDE, when viewing from `ExprTraverser`?
# - Are we able to safely call this, after narrowing with an enum?
#     `node.get(instance).is_scalar()`
ExprNodeAny: TypeAlias = "ExprNode[ExprIR, _Observe] | ExprNode[OneOrSeq[ExprIR], _Skip]"
IntoTraverser: TypeAlias = "Iterable[ExprNodeAny]"


@final
class ExprTraverser:
    """Field specifier-based iteration backend for `ExprIR`.

    ## Notes
    - Methods *accepting* an `instance` correspond to those in `ExprIR` with the same name
    - The rest expose a subset of `Sequence[ExprNode]`
        - Excluding the `value`-based methods
    """

    __slots__ = ("_names", "_nodes")
    _nodes: Seq[ExprNodeAny]

    def __init__(self, nodes: Seq[ExprNodeAny], /) -> None:
        self._nodes = nodes
        self._names: Seq[str] | None = None

    def __repr__(self) -> str:
        return self._repr_with("\n", " ")

    def _repr_html_(self) -> str:
        return self._repr_with("<br>", "&nbsp;")

    def _repr_with(
        self, new_line: Literal["\n", "<br>"], indent: Literal[" ", "&nbsp;"], /
    ) -> str:
        tp_name = type(self).__name__
        if self:
            members = new_line.join(f"{indent * 4}{n!r}" for n in self)
            return f"{tp_name}[{len(self)}]{new_line}{members}"
        return f"{tp_name}[]"

    def extend_with(self, nodes: Iterable[ExprNodeAny], /) -> ExprTraverser:
        """Create a new traverser, extending with `nodes`.

        Arguments:
            nodes: New nodes extracted from a subclass `__dict__`.
                - The order of the current (parent) nodes is preserved.
                - `nodes` follow immediately after
        """
        return ExprTraverser((*self, *nodes))

    # NOTE: `ExprIR` API
    def iter_left(self, instance: ExprIR, /) -> Iterator[ExprIR]:
        """Yield from nodes recursively from root->leaf."""
        for node in self:
            yield from node.iter_left(instance)
        yield instance

    def iter_right(self, instance: ExprIR, /) -> Iterator[ExprIR]:
        """Yield from nodes recursively from leaf->root."""
        yield instance
        for node in reversed(self):
            yield from node.iter_right(instance)

    def iter_output_name(self, instance: ExprIR, /) -> Iterator[ExprIR]:
        """Follow the **left-hand-side** of the graph until we can derive an output name.

        See `ExprIR.iter_output_name` for examples.
        """
        if self:
            yield from self[0].iter_output_name(instance)
        else:
            yield instance

    # TODO @dangotbanned: (low priority) integrate `FunctionExpr`
    def iter_expand(self, instance: ExprIR, ctx: Expander, /) -> Iterator[ExprIR]:
        """Expand an expression, using the default strategy.

        Arguments:
            instance: The expression to expand.
            ctx: The expansion context to resolve the operation in.

        Notes:
            This strategy applies the following rules:

            - `node()` expands into multiple independent expressions
                - Which is permitted only at `self[0]`
            - `nodes()` expands within the container it describes
            - Expressions without inputs (e.g. `Column`) have nothing to expand
            - Selectors expand into `Column`(s), and then follow the above

        See Also:
            `ExprNode.expand_as_first_child`
            `ExprNode.expand_as_next_sibling`
        """
        if not self:
            yield instance
            return

        if len(self) > 1 and (
            changes := {
                sibling.name: expanded
                for sibling in reversed(self[1:])
                if (expanded := sibling.expand_as_next_sibling(instance, ctx))
            }
        ):
            instance = instance.__replace__(**changes)

        child = self[0]
        name = child.name
        for expanded in child.expand_as_first_child(instance, ctx):
            yield instance.__replace__(**{name: expanded})

    def iter_expand_by_combination(
        self, instance: ExprIR, ctx: Expander, /
    ) -> Iterator[ExprIR]:
        r"""Expand an expression, with broadcasting and/or zipping of it's inputs.

        Arguments:
            instance: The expression to expand.
            ctx: The expansion context to resolve the operation in.

        Important:
            Adapted from [upstream].
            The algorithm is similar to [`more_itertools.zip_broadcast`].

        Examples:
            >>> import narwhals._plan as nw
            >>> from narwhals import Boolean, Float64, Int64, UInt8
            >>> from narwhals._plan import selectors as ncs
            >>> from narwhals._plan._expansion import Expander
            >>> from tests.plan.utils import Frame

            >>> def show(nodes: Iterable[Any]) -> None:
            ...     print(*(map(repr, nodes)), sep="\n")

            >>> i64, u8, f64, bool = Int64(), UInt8(), Float64(), Boolean()
            >>> schema = {"a": bool, "b": i64, "c": u8, "d": f64, "e": bool, "f": f64}
            >>> ctx = Expander(schema)
            >>> df = Frame.from_mapping(schema)

            Combination expansion is used for `BinaryExpr` and `TernaryExpr` [^1].

            Selectors are supported in each position, with nothing fancy on single-outputs:
            >>> singles = (nw.when(ncs.first()).then(ncs.last()).otherwise(nw.nth(2)))._ir
            >>> singles
            .when(ncs.first()).then(ncs.last()).otherwise(ncs.by_index([2]))
            >>> show(singles.iter_expand(ctx))
            .when(col('a')).then(col('f')).otherwise(col('c'))

            We combine single/multi-outputs via **broadcasting**:
            >>> broadcasting = (nw.col("b") + nw.col("c", "d", "f"))._ir
            >>> broadcasting
            [(col('b')) + (ncs.by_name('c', 'd', 'f'))]
            >>> show(broadcasting.iter_expand(ctx))
            [(col('b')) + (col('c'))]
            [(col('b')) + (col('d'))]
            [(col('b')) + (col('f'))]

            And this is *not restricted* to any particular sub-expression:
            >>> show(df.project(nw.when("a").then(ncs.numeric())))
            b=.when(col('a')).then(col('b')).otherwise(lit(null))
            c=.when(col('a')).then(col('c')).otherwise(lit(null))
            d=.when(col('a')).then(col('d')).otherwise(lit(null))
            f=.when(col('a')).then(col('f')).otherwise(lit(null))

            >>> show(df.project(nw.when(ncs.boolean()).then(1).otherwise(0).name.keep()))
            a=.when(col('a')).then(lit(int: 1)).otherwise(lit(int: 0))
            e=.when(col('e')).then(lit(int: 1)).otherwise(lit(int: 0))

            We combine equal-sized multi-outputs by **zipping** them together:
            >>> zipping = (nw.col("b", "c") + nw.col("d", "f"))._ir
            >>> zipping
            [(ncs.by_name('b', 'c')) + (ncs.by_name('d', 'f'))]
            >>> show(zipping.iter_expand(ctx))
            [(col('b')) + (col('d'))]
            [(col('c')) + (col('f'))]

            We can broadcast *and* zip in the same expression:
            >>> show(
            ...     nw.when(ncs.boolean())
            ...     .then(ncs.float())
            ...     .otherwise(ncs.last().min())
            ...     ._ir.iter_expand(ctx)
            ... )
            .when(col('a')).then(col('d')).otherwise(col('f').min())
            .when(col('e')).then(col('f')).otherwise(col('f').min())

            [`fill_nan`] makes good use of that under-the-covers:
            >>> zip_broadcast_2 = ncs.float().fill_nan(0.0)._ir
            >>> zip_broadcast_2
            .when([(ncs.float().is_not_nan()) | (ncs.float().is_null())]).then(ncs.float()).otherwise(lit(float: 0.0))
            >>> show(zip_broadcast_2.iter_expand(ctx))
            .when([(col('d').is_not_nan()) | (col('d').is_null())]).then(col('d')).otherwise(...)
            .when([(col('f').is_not_nan()) | (col('f').is_null())]).then(col('f')).otherwise(...)

            This is not magical, and still requires the zipping rule to be followed:
            >>> bad = nw.when(ncs.boolean()).then(nw.nth(1, 3)).otherwise(nw.all())._ir
            >>> next(bad.iter_expand(ctx))  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            MultiOutputExpressionError: Cannot combine selectors that produce a different number of columns (2 != 2 != 6).

        [^1]: `polars` uses this strategy for `over`, `sort_by`, `filter` + others.

              Not convinced that those are intuitive (see [polars#25022], [polars#25317]).

        [upstream]: https://github.com/pola-rs/polars/blob/bb93ba8e67a1f38951506ce044245560009fe55a/crates/polars-plan/src/plans/conversion/dsl_to_ir/expr_expansion.rs#L129-L197
        [polars#25022]: https://github.com/pola-rs/polars/issues/25022
        [polars#25317]: https://github.com/pola-rs/polars/issues/25317
        [`more_itertools.zip_broadcast`]: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.zip_broadcast
        [`fill_nan`]: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/dsl/mod.rs#L908-L916
        """
        if not instance.meta.has_multiple_outputs():
            changes = {e.name: next(e.iter_expand(instance, ctx)) for e in self}
            yield instance.__replace__(**changes)
            return
        # **First act**: Expand each node, observing how many exprs plopped out ():
        # 1. `size==1`: We can broadcast replacement(s) through by overwriting our `in_progress` temporary
        # 2. `size>=2`: The first one we see becomes the target (`expansion_size`) that all others need to get in line with
        # 3. `size==?`: Raise on anything new that *tries* to join the party
        in_progress = instance
        expansion_size: int = 0
        expansions: dict[str, Seq[ExprIR]] = {}
        for node, name in zip(self, self.names):
            expanded = tuple(node.iter_expand(instance, ctx))
            if (size := len(expanded)) == 1:
                in_progress = in_progress.__replace__(**{name: expanded[0]})
            elif not expansion_size or size == expansion_size:
                expansions[name] = expanded
                expansion_size = size
            else:
                expand_all = (tuple(e.iter_expand(instance, ctx)) for e in self)
                mixed_sizes = tuple(len(exprs) for exprs in expand_all)
                raise combination_mixed_multi_output_error(instance, mixed_sizes)

        # **Intermission**: Fast paths for all single-output, or only 1 multi-output expansion
        if not expansion_size:
            yield in_progress
            return
        as_expansion = in_progress.__replace__
        if len(expansions) == 1:
            name, expanded = next(iter(expansions.items()))
            yield from (as_expansion(**{name: expr}) for expr in expanded)
        else:
            # **Second act**: Unzipping expanded values together, yielding the combinations
            names = tuple(expansions)
            values_per_expansion: zip[Seq[ExprIR]] = zip(*expansions.values())
            yield from (as_expansion(**dict(zip(names, v))) for v in values_per_expansion)

    def is_scalar(self, instance: ExprIR, /) -> bool:
        return _all_empty_false(
            node.get(instance).is_scalar()
            for node in self
            if node.observe_scalar is _OBSERVE
        )

    def map_ir(self, instance: ExprIR, function: MapIR, /) -> ExprIR:
        if self and (
            changes := {
                node.name: change
                for node in self
                if (change := node.map_nodes(instance, function))
            }
        ):
            instance = instance.__replace__(**changes)
        return function(instance)

    @property
    def names(self) -> Seq[str]:
        """The name of each field that stores an expression."""
        if self._names is None:
            self._names = tuple(node.name for node in self)
        return self._names

    # NOTE: `Sequence[ExprNode]` API
    @overload
    def __getitem__(self, key: int, /) -> ExprNodeAny: ...
    @overload
    def __getitem__(self, key: slice, /) -> Seq[ExprNodeAny]: ...
    def __getitem__(self, key: int | slice, /) -> ExprNodeAny | Seq[ExprNodeAny]:
        return self._nodes.__getitem__(key)

    def __iter__(self) -> Iterator[ExprNodeAny]:
        yield from self._nodes

    def __len__(self) -> int:
        return self._nodes.__len__()

    def __reversed__(self) -> Iterator[ExprNodeAny]:
        if self:
            yield from reversed(self._nodes)


def _all_empty_false(iterable: Iterable[bool], /) -> bool:
    """Return True if bool(x) is True for all values x in the iterable.

    If the iterable is empty, return False.
    """
    it = iter(iterable)
    return (next(it, False)) and all(it)


_EXPR_FIELD_SPECIFIER_NAMES: Final = (node.__name__, nodes.__name__)


def _into_expr_node_error(assigned_name: str, node: Any, cls_name: str) -> TypeError:
    from narwhals._utils import qualified_type_name

    name = assigned_name
    value = repr(node)
    value = value[:10] + "..." if len(value) > 10 else value
    value = f"`{value}`"
    hints = (
        f"if {name!r} is a class variable,\n  remove it from __slots__",
        f"if {value} is a default, for `{cls_name}({name}=...)`,"
        "\n  define the it in a classmethod, staticmethod or function instead of the class body",
        f"if {value} came from a new fancy field specifier,\n  fix the check that raised this!",
    )
    node_examples = ", ".join(f"`{nm}()`" for nm in _EXPR_FIELD_SPECIFIER_NAMES)
    what = f"Incompatible assignment in ExprIR subclass {cls_name!r}."
    cause = f"The class body tried to assign a {qualified_type_name(node)!r} to {name!r}, while also declaring {name!r} in __slots__."
    nl = "\n"
    return TypeError(
        f"{what}\n\n{cause}\n"
        f"This syntax is reserved for field specifiers: {node_examples}\n\n"
        f"Hints:\n{nl.join(f'- {hint}' for hint in hints)}"
    )
