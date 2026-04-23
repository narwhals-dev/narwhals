"""Core expression intermediate representations.

## Implementation Notes
The design is *based on* (rust) polars, with *deviations* where:
- `rust != python`
  - Lack of (typing) support for non-trivial [algebraic data types]
  - (*Unsurprisingly*) memory management architecture
- If our subset of polars doesn't/can't support things
- A more faithful adaptation would require considerably more code

### `ExprIR`
Variants of [`dsl::expr::Expr`] are subclasses of `ExprIR`.

### `SelectorIR`
[`dsl::expr::Expr::Selector`] is `class SelectorIR(ExprIR)`.

*Some* variants of [`dsl::selector::Selector`] relate to *subclasses of* `SelectorIR`.

### `NamedIR`
[`plans::expr_ir::ExprIR`] introduced storing a single output name.

While that and [`plans::aexpr::AExpr`] have a focus on memory management (*not here*).

You could think of these sharing a role in lowering an IR:

    # Polars
    Expr   -> AExpr

    # Narwhals
    ExprIR -> NamedIR[ExprIR]

[`dsl::expr::Expr`]: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/dsl/expr/mod.rs#L88-L223
[`dsl::expr::Expr::Selector`]: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/dsl/expr/mod.rs#L104
[`dsl::selector::Selector`]: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/dsl/selector.rs#L148-L152
[algebraic data types]: https://github.com/jspahrsummers/adt
[`plans::expr_ir::ExprIR`]: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/plans/expr_ir.rs#L63-L74
[`plans::aexpr::AExpr`]: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/plans/aexpr/mod.rs#L179-L269
"""

from __future__ import annotations

import functools

# ruff: noqa: N806
from functools import reduce
from typing import TYPE_CHECKING, Generic, Literal, final

from narwhals._plan._dispatch import Dispatcher, DispatcherOptions
from narwhals._plan._dtype import IntoResolveDType, ResolveDType
from narwhals._plan._immutable import _OBJ_SETATTR, Immutable
from narwhals._plan._meta import ExprIRMeta
from narwhals._plan._nodes import ExprTraverser
from narwhals._plan._version import into_version
from narwhals._plan.typing import ExprIRT_co
from narwhals._utils import Version, unstable
from narwhals.dtypes import DType
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from typing import Any, ClassVar

    from typing_extensions import Self

    from narwhals._plan._expansion import Expander
    from narwhals._plan.compliant import CompliantNamespace, typing as ct
    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions.expr import Alias, BinaryExpr, Cast, Column
    from narwhals._plan.expressions.operators import Eq
    from narwhals._plan.meta import MetaNamespace
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.selectors import Selector
    from narwhals._plan.typing import Ignored, MapIR
    from narwhals.typing import IntoDType


class ExprIR(Immutable, metaclass=ExprIRMeta):
    r"""An immutable representation of an expression.

    All functions and methods that return an `Expr` are backed by an `ExprIR`.

    That may be a single node:
    >>> import narwhals._plan as nw
    >>> column = nw.col("howdy")
    >>> column._ir
    col('howdy')
    >>> print(column._ir)
    Column(name='howdy')

    Or something more deeply nested:
    >>> bigger = (column + 10.5).alias("more")
    >>> bigger_ir = bigger._ir
    >>> print(bigger_ir)
    Alias(expr=BinaryExpr(left=..., op=Add(), right=Lit(..., value=10.5)), name='more')

    An `ExprIR` is an easily traversable graph:
    >>> def show_order(nodes: Iterator[ExprIR]) -> None:
    ...     print("\n".join(f"{idx}: {node}" for idx, node in enumerate(nodes)))

    Supporting iteration from both *root to leaf*:
    >>> show_order(bigger_ir.iter_left())
    0: Column(name='howdy')
    1: Lit(dtype=Float64, value=10.5)
    2: BinaryExpr(left=..., op=Add(), right=...)
    3: Alias(expr=BinaryExpr(...), name='more')

    And *leaf to root* - for the same cost:
    >>> show_order(bigger_ir.iter_right())
    0: Alias(expr=BinaryExpr(...), name='more')
    1: BinaryExpr(left=..., op=Add(), right=...)
    2: Lit(dtype=Float64, value=10.5)
    3: Column(name='howdy')

    That comes in handy for [`meta`] operations, which are available for both `Expr` and `ExprIR`:
    >>> bigger_ir.meta.root_names()
    ['howdy']
    >>> bigger_ir.meta.output_name()
    'more'

    See `ExprIR.map_ir` for other superpowers this gives us.

    [`meta`]: https://docs.pola.rs/api/python/stable/reference/expressions/meta.html
    """

    __expr_ir_nodes__: ClassVar[ExprTraverser] = ExprTraverser(())
    """Graph traversal backend.

    Populated through the use of the `node`, `nodes` field specifiers in the class body:
    >>> from narwhals._plan._nodes import node, nodes
    >>> from narwhals._plan.typing import Seq
    >>> class Over(ExprIR):
    ...     __slots__ = ("expr", "partition_by")
    ...     expr: ExprIR = node(observe_scalar=False)
    ...     partition_by: Seq[ExprIR] = nodes()

    Aaaaand there it is:
    >>> Over.__expr_ir_nodes__
    ExprTraverser[2]
        expr: ExprIR = node(observe_scalar=False)
        partition_by: Seq[ExprIR] = nodes()

    The order they are defined is meaningful, and extends to subclasses:
    >>> from narwhals._plan.options import SortOptions
    >>> class OverOrdered(Over):
    ...     __slots__ = ("order_by", "sort_options")
    ...     order_by: Seq[ExprIR] = nodes()
    ...     sort_options: SortOptions

    Note how `sort_options` does not use a field specifier, and is ignored for traversal:
    >>> OverOrdered.__expr_ir_nodes__
    ExprTraverser[3]
        expr: ExprIR = node(observe_scalar=False)
        partition_by: Seq[ExprIR] = nodes()
        order_by: Seq[ExprIR] = nodes()

    This is what provides the default implementation of:
    - `iter_left`
    - `iter_right`
    - `iter_output_name`
    - `is_scalar`
    - `map_ir`
    """

    __expr_ir_dispatch__: ClassVar[Dispatcher[Self]] = Dispatcher()
    """Callable that dispatches to the appropriate compliant-level method.

    See `Dispatcher` and `DispatcherOptions` for examples.

    To customize the behavior, use the `dispatch` **parameter** [when subclassing]:

        class Column(ExprIR, dispatch=DispatcherOptions.namespaced("col")):
            __slots__ = ("name",)
            name: str

    If nothing there *quite* scratches the itch, override the `dispatch` **method** instead:

        from narwhals._plan._function import Function
        from narwhals._plan._nodes import nodes
        from narwhals._plan.compliant import ExprDispatch

        class FunctionExpr(ExprIR):
            __slots__ = ("input", "function")
            input: tuple[ExprIR, ...] = nodes()
            function: Function

            def dispatch(self, ctx: ExprDispatch[T, R], frame: T, name: str) -> R:
                return self.function.__expr_ir_dispatch__(self, ctx, frame, name)

    Notes:
        Each class has their own `Dispatcher` instance, and inheritance is only on the `options` property.

    [when subclassing]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
    """

    __expr_ir_dtype__: ClassVar[ResolveDType[Self]] = ResolveDType()
    """Callable defining how a `DType` is derived when `resolve_dtype` is called.

    If the logic fits an existing pattern, use the `dtype` **parameter** [when subclassing]:

        class Slice(ExprIR, dtype=ResolveDType.expr_ir.same_dtype()):
            __slots__ = ("expr", "length", "offset")
            expr: ExprIR = node()
            offset: int
            length: int | None

    See `IntoResolveDType` and `ResolveDType` for more examples.

    If nothing there *quite* scratches the itch, override `resolve_dtype` instead:

        from narwhals.dtypes import Array, List

        class Explode(ExprIR):
            __slots__ = ("expr",)
            expr: ExprIR = node()

            def resolve_dtype(self, schema: FrozenSchema) -> DType:
                dtype = self.expr.resolve_dtype(schema)
                if not isinstance(dtype, (Array, List)):
                    raise NotImplementedError(dtype)
                inner = dtype.inner
                return inner if not isinstance(inner, type) else inner()

    [when subclassing]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
    """

    def __init_subclass__(
        cls: type[Self],
        *,
        dispatch: DispatcherOptions | Literal["no_dispatch"] | None = None,
        dtype: IntoResolveDType[Self] | None = None,
        **_: Any,
    ) -> None:
        """Hook to [customize a new subclass] of `ExprIR`.

        All parameters are optional and will be inherited when not provided to `__init_subclass__`.

        Arguments:
            dispatch: Defines how to build a `Dispatcher`.
                Stored in `__expr_ir_dispatch__.options`.

                *"no_dispatch"* is syntax sugar for `DispatcherOptions(allow_dispatch=False)`.

            dtype: Defines how a `DType` is derived when `resolve_dtype` is called.
                Stored in `__expr_ir_dtype__`.

                See `IntoResolveDType` and `ResolveDType` for usage.

                **Warning**: This functionality is considered **unstable**.
                Full support depends on [#3396].

        [customize a new subclass]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
        [#3396]: https://github.com/narwhals-dev/narwhals/pull/3396
        """
        super().__init_subclass__(**_)
        cls.__expr_ir_dispatch__ = Dispatcher.from_expr_ir(cls, dispatch)
        if dtype is not None:
            if isinstance(dtype, DType):
                dtype = ResolveDType.just_dtype(dtype)
            elif not isinstance(dtype, ResolveDType):
                dtype = ResolveDType.expr_ir.visitor(dtype)  # pragma: no cover
            cls.__expr_ir_dtype__ = dtype

    # TODO @dangotbanned: Come back to this doc after slimming down `Compliant{Expr,Scalar}`
    def dispatch(
        self: Self,
        ctx: ct.DispatchScopeAny[ct.Frame, ct.ET_co, ct.ST_co],
        frame: ct.Frame,
        name: str,
        /,
    ) -> ct.ET_co | ct.ST_co:
        """Evaluate this expression in `frame`, using implementation(s) provided by `ctx`.

        Arguments:
            ctx: Anything that can return a `CompliantExpr`.
            frame: A`*Frame` that shares a namespace with `ctx`.
            name: Output column name, usually `NamedIR.name`.

        Notes:
            - `ctx`/`ExprDispatch` is intended to be typed in a permissive way
              but seems to have gone too far in that direction
            - `_ArrowDispatch` utilizes it to define a common base for `*Expr`, `*Scalar`
              while refining `R_co` -> `StoresNativeT_co`
            - Neither of those types are `CompliantExpr`
        """
        return self.__expr_ir_dispatch__(self, ctx, frame, name)

    def resolve_dtype(self: Self, schema: FrozenSchema) -> DType:
        """Get the data type of an expanded expression.

        Arguments:
            schema: The same schema used to project this expression.
        """
        return self.__expr_ir_dtype__(self, schema)

    def to_narwhals(self, version: Version = Version.MAIN) -> Expr:
        """Convert this `ExprIR` into an narwhals-level `Expr`.

        Arguments:
            version: API version to export into.
        """
        return into_version(version).expr._from_ir(self)

    def to_selector_ir(self) -> SelectorIR:
        """Try to convert this `ExprIR` into a `SelectorIR`.

        >>> import narwhals._plan as nw

        The only valid conversion is for a `Column`:
        >>> column = nw.col("a")._ir
        >>> column
        col('a')

        >>> selector = column.to_selector_ir()
        >>> selector
        ncs.by_name('a')

        For a `SelectorIR`, this is a noop:
        >>> selector.to_selector_ir()
        ncs.by_name('a')

        Everything else will raise:
        >>> alias = nw.col("a").alias("bad")._ir
        >>> alias.to_selector_ir()
        Traceback (most recent call last):
        narwhals.exceptions.InvalidOperationError: cannot turn `col('a').alias('bad')` into a selector
        """
        msg = f"cannot turn `{self!r}` into a selector"
        raise InvalidOperationError(msg)

    # NOTE: Quite aware of how these seem to be at odds:
    # - `changes_length`, `is_length_preserving`, `is_scalar`
    def changes_length(self) -> bool:
        """Return True if this leaf changes the length of input columns, without reducing to a scalar.

        Literals (including series) and aggregations are not considered to be length-changing.
        """
        return not self.is_scalar() and not self.is_length_preserving()

    def is_length_preserving(self) -> bool:
        """Return True if this leaf maintains the length of input columns.

        Literals (including series) and aggregations are not length-preserving.

        Implementations adapted from [upstream].

        [upstream]: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-stream/src/physical_plan/lower_expr.rs#L301-L402
        """
        # NOTE: Many can piggyback off the mutual exclusive relationship with `is_scalar`
        return not self.is_scalar()

    def is_scalar(self) -> bool:
        """Return True if this leaf produces a single value.

        Some expressions are always scalar:
        >>> import narwhals._plan as nw
        >>> length = nw.len()
        >>> length._ir.is_scalar()
        True
        >>> nw.lit(123)._ir.is_scalar()
        True

        Others always output a column:
        >>> column = nw.col("a")
        >>> column._ir.is_scalar()
        False
        >>> nw.int_range(0, 10)._ir.is_scalar()
        False

        Many depend on the scalar-ness of child expressions, and require traversal:
        >>> (column + length)._ir.is_scalar()
        False
        >>> (column.first() + length)._ir.is_scalar()
        True

        ## Notes
        Subclasses should override in 2 cases:
        1. They are unconditionally scalar (`Len`, `AggExpr`)
        2. They answer the question using non-node fields (`FunctionExpr.function`, `Lit.value`)
        """
        return self.__expr_ir_nodes__.is_scalar(self)

    def needs_expansion(self) -> bool:
        """Return True if this expression contains selectors.

        >>> import narwhals._plan as nw
        >>> a = nw.col("a")
        >>> bc = nw.col("b", "c")
        >>> a._ir.needs_expansion()
        False
        >>> bc._ir.needs_expansion()
        True
        >>> (a * bc)._ir.needs_expansion()
        True
        """
        return any(isinstance(e, SelectorIR) for e in self.iter_left())

    def map_ir(self, function: MapIR, /) -> ExprIR:
        """Transform an expression by applying a function to all nodes in the graph.

        Arguments:
            function: A single argument [idempotent] function.

                Called *recursively* on any inputs and then the current node.

        Tip:
            Use `NamedIR.map_ir` if `function` requires selector expansion.

        Returns:
            Either
            - A new `ExprIR`, with any changes made as a result of `function`
            - The same `ExprIR` (by identity)

        Examples:
            >>> import narwhals._plan as nw
            >>> import narwhals._plan.expressions as ir
            >>> import narwhals._plan.expressions.functions as F

            `nw.*_horizontal` functions allow us to aggregate *across* columns:
            >>> expr = nw.sum_horizontal("a", "b", "c").alias("sum")

            However, most backends **do not** have a direct analogue to this concept.

            `map_ir` helps us rewrite an `ExprIR` *in terms of* others that they **do** support:
            >>> from collections import deque
            >>> def sum_horizontal_to_add(expr: ir.ExprIR) -> ir.ExprIR:
            ...     if isinstance(expr, ir.FunctionExpr) and isinstance(
            ...         expr.function, F.SumHorizontal
            ...     ):
            ...         inputs = deque(expr.input)
            ...         left = inputs.popleft()
            ...         add = ir.operators.Add()
            ...         while inputs:
            ...             left = ir.BinaryExpr(
            ...                 left=left, op=add, right=inputs.popleft()
            ...             )
            ...         return left
            ...     # Anything else, we return unchanged
            ...     return expr

            So while this version works for `polars`:
            >>> before = expr._ir
            >>> before
            col('a').sum_horizontal([col('b'), col('c')]).alias('sum')

            Any backend that supports `+` can understand this guy and we only needed to write it once:
            >>> after = before.map_ir(sum_horizontal_to_add)
            >>> after
            [([(col('a')) + (col('b'))]) + (col('c'))].alias('sum')

        Notes:
            - The name `map_ir` is a nod to [`plans::iterator::Expr.map_expr`]
            - The iteration pattern is adapted from [`plans::iterator::!push_expr`]

        [idempotent]: https://en.wikipedia.org/wiki/Idempotence#Computer_science_meaning
        [`plans::iterator::Expr.map_expr`]: https://github.com/pola-rs/polars/blob/3ea81c45e0c184af2cf5a93f8378cf330e4658c9/crates/polars-plan/src/plans/iterator.rs#L166-L169
        [`plans::iterator::!push_expr`]: https://github.com/pola-rs/polars/blob/3ea81c45e0c184af2cf5a93f8378cf330e4658c9/crates/polars-plan/src/plans/iterator.rs#L10-L124
        """
        return self.__expr_ir_nodes__.map_ir(self, function)

    # NOTE: Pylance renders "Examples:" sections poorly if there isn't "Arguments:" as well
    # This style still runs in `doctest` and looks better in vscode
    def iter_left(self) -> Iterator[ExprIR]:
        r"""Yield nodes recursively from root->leaf.

        `iter_left` will always yield the current node **last**.

        >>> import narwhals._plan as nw
        >>> def show(nodes: Iterator[ExprIR]) -> None:
        ...     print("\n".join(repr(e) for e in nodes))

        A root node just yields itself:
        >>> show(nw.col("a")._ir.iter_left())
        col('a')

        A binary expression is a little more interesting:
        >>> binary = nw.col("a").alias("b") + nw.col("c").min().alias("d")
        >>> show(binary._ir.iter_left())
        col('a')
        col('a').alias('b')
        col('c')
        col('c').min()
        col('c').min().alias('d')
        [(col('a').alias('b')) + (col('c').min().alias('d'))]

        And we can go fancier:
        >>> window = nw.col("e").first().over("f", order_by=["g", "h"])
        >>> show(window._ir.iter_left())
        col('e')
        col('e').first()
        col('f')
        col('g')
        col('h')
        col('e').first().over(partition_by=[col('f')], order_by=[col('g'), col('h')])
        """
        yield from self.__expr_ir_nodes__.iter_left(self)

    def iter_right(self) -> Iterator[ExprIR]:
        r"""Yield nodes recursively from leaf->root.

        `iter_right` will always yield the current node **first**.

        >>> import narwhals._plan as nw
        >>> def show(nodes: Iterator[ExprIR]) -> None:
        ...     print("\n".join(repr(e) for e in nodes))

        A root node just yields itself:
        >>> show(nw.col("a")._ir.iter_right())
        col('a')

        A binary expression is a little more interesting:
        >>> binary = nw.col("a").alias("b") + nw.col("c").min().alias("d")
        >>> show(binary._ir.iter_right())
        [(col('a').alias('b')) + (col('c').min().alias('d'))]
        col('c').min().alias('d')
        col('c').min()
        col('c')
        col('a').alias('b')
        col('a')

        And we can go fancier:
        >>> window = nw.col("e").first().over("f", order_by=["g", "h"])
        >>> show(window._ir.iter_right())
        col('e').first().over(partition_by=[col('f')], order_by=[col('g'), col('h')])
        col('h')
        col('g')
        col('f')
        col('e').first()
        col('e')
        """
        yield from self.__expr_ir_nodes__.iter_right(self)

    def iter_output_name(self) -> Iterator[ExprIR]:
        """Follow the **left-hand-side** of the graph until we can derive an output name.

        Used for `ExprIR.meta.output_name` and will stop as soon as we see one of:

        A root node with a `name`:

            Column
            Lit
            LitSeries
            Len

        A branch node with a `name`:

            Alias

        A special case where we can navigate to a name:

            ByName(names=("name",), ...)
            #              ^^^^
            #              Equivalent to `col("name")`

            StructExpr(function=FieldByName(name="name"), ...)
            #                                     ^^^^
            #    Same idea, but with a `Struct` schema

        A branch node that transforms the name of the above:

            RenameAlias

        A branch node that requires schema context for expansion, raising
        instead of recursing further:

            KeepName
        """
        yield from self.__expr_ir_nodes__.iter_output_name(self)

    def iter_expand(self, ctx: Expander, /) -> Iterator[ExprIR]:
        r"""Yield the expression(s) that the current node expands into.

        Arguments:
            ctx: The expansion context to resolve the operation in.

        Examples:
            >>> import narwhals as nw
            >>> from narwhals._plan import col, selectors as ncs, sum_horizontal
            >>> from narwhals._plan._expansion import Expander

            >>> def show(nodes: Iterator[ExprIR], *, use_repr: bool = False) -> None:
            ...     print(*(map(repr, nodes) if use_repr else nodes), sep="\n")

            >>> i64 = nw.Int64()
            >>> schema_1 = {"a": i64, "b": i64, "c": i64, "d": nw.Float64()}
            >>> schema_2 = {"a": nw.Datetime(), "b": i64, "c": nw.Datetime("ns")}

            When we say *expansion*, we're talking about taking a single expression:
            >>> cast_ab = col("a", "b").cast(nw.String)._ir
            >>> print(cast_ab)
            Cast(expr=ByName(names=['a', 'b'], require_all=True), dtype=String)

            And transforming it into *zero or more* new expressions:
            >>> ctx_1 = Expander(schema_1)
            >>> show(cast_ab.iter_expand(ctx_1))
            Cast(expr=Column(name='a'), dtype=String)
            Cast(expr=Column(name='b'), dtype=String)

            Yep, zero is possible:
            >>> cast_datetime = ncs.datetime().cast(nw.String)._ir
            >>> show(cast_datetime.iter_expand(ctx_1))
            <BLANKLINE>

            It's all contextual:
            >>> ctx_2 = Expander(schema_2)
            >>> show(cast_datetime.iter_expand(ctx_2))
            Cast(expr=Column(name='a'), dtype=String)
            Cast(expr=Column(name='c'), dtype=String)

            There are a few different kinds of expansion:
            >>> standard = ncs.all().sum()._ir
            >>> into_input = sum_horizontal(ncs.all())._ir
            >>> broadcasting = (col("a") + col("b", "c", "d"))._ir
            >>> zipping = (col("a", "b") + col("c", "d"))._ir
            >>> mixed = (
            ...     ncs.all()
            ...     .cum_sum()
            ...     .over(ncs.integer() - ncs.first(), order_by=ncs.last())
            ...     ._ir
            ... )

            The one you all know and love:
            >>> standard
            ncs.all().sum()
            >>> show(standard.iter_expand(ctx_1), use_repr=True)
            col('a').sum()
            col('b').sum()
            col('c').sum()
            col('d').sum()

            Another that might feel familiar:
            >>> into_input
            ncs.all().sum_horizontal()
            >>> show(into_input.iter_expand(ctx_1), use_repr=True)
            col('a').sum_horizontal([col('b'), col('c'), col('d')])

            One for the more adventurous:
            >>> broadcasting
            [(col('a')) + (ncs.by_name('b', 'c', 'd'))]
            >>> show(broadcasting.iter_expand(ctx_1), use_repr=True)
            [(col('a')) + (col('b'))]
            [(col('a')) + (col('c'))]
            [(col('a')) + (col('d'))]

            Did you know about this one though?:
            >>> zipping
            [(ncs.by_name('a', 'b')) + (ncs.by_name('c', 'd'))]
            >>> show(zipping.iter_expand(ctx_1), use_repr=True)
            [(col('a')) + (col('c'))]
            [(col('b')) + (col('d'))]

            Some fancy expressions use different strategies per-field [^1]:
            >>> mixed
            ncs.all().cum_sum().over(partition_by=[[ncs.integer() - ncs.first()]], order_by=[ncs.last()])
            >>> show(mixed.iter_expand(ctx_1), use_repr=True)
            col('a').cum_sum().over(partition_by=[col('b'), col('c')], order_by=[col('d')])
            col('b').cum_sum().over(partition_by=[col('b'), col('c')], order_by=[col('d')])
            col('c').cum_sum().over(partition_by=[col('b'), col('c')], order_by=[col('d')])
            col('d').cum_sum().over(partition_by=[col('b'), col('c')], order_by=[col('d')])


        [^1]: This is an intentional deviation from polars (see [polars#25022]).

        [polars#25022]: https://github.com/pola-rs/polars/issues/25022
        """
        yield from self.__expr_ir_nodes__.iter_expand(self, ctx)

    @property
    def meta(self) -> MetaNamespace:
        """Methods to traverse and introspect existing expressions."""
        from narwhals._plan.meta import MetaNamespace

        return MetaNamespace(_ir=self)

    def cast(self, dtype: DType) -> Cast:
        """Syntax sugar for `Cast(expr=self, dtype=dtype)`."""
        from narwhals._plan.expressions.expr import Cast

        return Cast(expr=self, dtype=dtype)

    def alias(self, name: str) -> Alias:
        """Syntax sugar for `Alias(expr=self, name=name)`."""
        from narwhals._plan.expressions.expr import Alias

        return Alias(expr=self, name=name)

    def eq(self, other: ExprIR) -> BinaryExpr[Self, Eq, ExprIR]:
        """Syntax sugar for `BinaryExpr(left=self, op=Eq(), right=other)`."""
        from narwhals._plan.expressions.operators import Eq

        return Eq().to_binary_expr(self, other)

    def _repr_html_(self) -> str:
        """Return a html representation of this expression, used by [IPython].

        Although this is identical to `__repr__`; a notebook will render the string in a prettier way.

        [IPython]: https://ipython.readthedocs.io/en/stable/config/integrating.html#custom-methods
        """
        return self.__repr__()


class SelectorIR(ExprIR, dispatch="no_dispatch"):
    """An expression that selects zero or more columns.

    All functions that return a `Selector` are backed by a `SelectorIR`.

    The `selectors` module is the usual entry point:
    >>> import narwhals._plan as nw
    >>> import narwhals._plan.selectors as ncs

    And selectors created this way support set operations:
    >>> selector = ncs.first() | ncs.boolean()
    >>> isinstance(selector, nw.Selector)
    True
    >>> selector._ir
    [ncs.first() | ncs.boolean()]

    Some expressions are simply selectors in disguise:
    >>> expr = nw.col("a", "b", "c")
    >>> isinstance(expr, nw.Selector)
    False
    >>> expr._ir
    ncs.by_name('a', 'b', 'c')

    We can use selectors almost anywhere that column names are expected:
    >>> data = {
    ...     "A": ["dog", "cat", "dog", "cat", "dog"],
    ...     "B": [1, 2, 1, 4, 0],
    ...     "C": [30.0, 40.5, 10.0, 20.5, 2.0],
    ...     "D": ["a", "b", "a", "a", "b"],
    ... }
    >>> df = nw.DataFrame.from_dict(data, backend="pyarrow")
    >>> result = (
    ...     df.group_by(ncs.string())
    ...     .agg(ncs.float().sum(), ncs.integer())
    ...     .with_columns(ncs.matches("C") / ncs.list().list.len())
    ...     .explode(ncs.list())
    ...     .sort(ncs.by_index(0, 1), descending=True)
    ... )
    >>> result.to_polars()
    shape: (5, 4)
    ┌─────┬─────┬──────┬─────┐
    │ A   ┆ D   ┆ C    ┆ B   │
    │ --- ┆ --- ┆ ---  ┆ --- │
    │ str ┆ str ┆ f64  ┆ i64 │
    ╞═════╪═════╪══════╪═════╡
    │ dog ┆ b   ┆ 2.0  ┆ 0   │
    │ dog ┆ a   ┆ 20.0 ┆ 1   │
    │ dog ┆ a   ┆ 20.0 ┆ 1   │
    │ cat ┆ b   ┆ 40.5 ┆ 2   │
    │ cat ┆ a   ┆ 20.5 ┆ 4   │
    └─────┴─────┴──────┴─────┘

    Selectors are schema-dependent, with column names determined during expression expansion:
    >>> middle = (~(ncs.first() | ncs.last()))._ir
    >>> list(middle.iter_expand_selector(result.schema))
    ['D', 'C']
    >>> list(middle.iter_expand_selector(df.schema))
    ['B', 'C']

    Unlike `col`, a selector can still be valid if it matches zero columns:
    >>> result.drop(ncs.struct()).columns == ["A", "D", "C", "B"]
    True
    """

    def to_narwhals(self, version: Version = Version.MAIN) -> Selector:
        return into_version(version).selector._from_ir(self)

    # TODO @dangotbanned: (low-priority) Add `SelectorIR.expand_selector`
    # - Would accept the flags for duplicates & any, but skip `ignored_columns`
    # - Return a tuple, while still getting validation (if needed)
    def iter_expand_selector(
        self, schema: Mapping[str, DType], ignored_columns: Ignored = (), /
    ) -> Iterator[str]:
        """Yield column names that match the selector in schema order [^1].

        Arguments:
            schema: Target scope to expand the selector in.
            ignored_columns: Names of `group_by` key columns.
                These columns will be excluded [^1] from the result.

        Notes:
            [^1]: Except `ByName`, `ByIndex`.

        Examples:
            >>> import narwhals as nw
            >>> import narwhals._plan.selectors as ncs
            >>> from narwhals._plan.schema import FrozenSchema
            >>> schema = FrozenSchema(x=nw.String(), y=nw.Int64(), z=nw.Float64())

            >>> s = ncs.numeric()._ir
            >>> list(s.iter_expand_selector(schema))
            ['y', 'z']
            >>> s = (ncs.first() | ncs.last())._ir
            >>> list(s.iter_expand_selector(schema))
            ['x', 'z']
            >>> s = (~(ncs.first() | ncs.last()))._ir
            >>> list(s.iter_expand_selector(schema))
            ['y']

            Most selectors are non-strict and yield empty without a match:
            >>> s = ncs.boolean()._ir
            >>> list(s.iter_expand_selector(schema))
            []

            By default, `by_name` and `by_index` will raise without a full match:
            >>> s = ncs.by_name("oops", "z")._ir
            >>> list(s.iter_expand_selector(schema))
            Traceback (most recent call last):
            narwhals.exceptions.ColumnNotFoundError: The following columns were not found: ['oops']...

            But that can be relaxed if needed:
            >>> s = ncs.by_name("oops", "z", require_all=False)._ir
            >>> list(s.iter_expand_selector(schema))
            ['z']
        """
        msg = f"{self.iter_expand_selector.__qualname__}"
        raise NotImplementedError(msg)

    def iter_expand(self, ctx: Expander, /) -> Iterator[Column]:
        Column = _import_column()
        yield from (
            Column(name=name)
            for name in self.iter_expand_selector(ctx.schema, ctx.ignored)
        )

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield from ()

    @final
    def matches(self, dtype: IntoDType, /) -> bool:
        """Return True if this selector matches the data type.

        Arguments:
            dtype: Anything that can be converted into a Narwhals DType.

        Important:
            This method **must not be overridden**, as it ensures the result is cached
            *without* storing it on the instance.
            Instead, override `SelectorIR._matches_dtype` to customize the check.

        Examples:
            >>> import narwhals as nw
            >>> import narwhals._plan.selectors as ncs
            >>> ncs.numeric()._ir.matches(nw.Float32)
            True
            >>> ncs.enum()._ir.matches(nw.String)
            False
            >>> (ncs.numeric() | ncs.temporal())._ir.matches(nw.Datetime())
            True
        """
        return _matches_dtype(self, dtype)

    def _matches_dtype(self, dtype: IntoDType) -> bool:
        """Return True if this selector matches the data type.

        Arguments:
            dtype: Anything that can be converted into a Narwhals DType.
        """
        msg = f"{type(self).__name__}._matches_dtype"
        raise NotImplementedError(msg)

    def to_dtype_selector(self) -> SelectorIR:
        """Try to convert this `SelectorIR` into a `DTypeSelector`.

        This helps us enforce what you can use inside a nested selector:
        >>> import narwhals as nw
        >>> import narwhals._plan.selectors as ncs

        We allow existing dtype selectors:
        >>> ncs.list(ncs.integer())._ir
        ncs.list(ncs.integer())

        Selectors that can be converted into dtype selectors:
        >>> ncs.list(ncs.all())._ir
        ncs.list(ncs.all())
        >>> ncs.list(ncs.empty())._ir
        ncs.list(ncs.empty())

        Compositions where each sub-selector satisfies the above:
        >>> ncs.list(~ncs.float())._ir
        ncs.list(~ncs.float())
        >>> ncs.list(ncs.enum() | ncs.list(ncs.struct()))._ir
        ncs.list([ncs.enum() | ncs.list(ncs.struct())])

        Everything else will raise:
        >>> ncs.list(ncs.float() | ncs.matches("inner"))
        Traceback (most recent call last):
        TypeError: expected data type based selector got `ncs.matches('inner')`
        """
        msg = f"expected data type based selector got `{self!r}`"
        raise TypeError(msg)

    def to_selector_ir(self) -> Self:
        return self

    def needs_expansion(self) -> bool:
        return True

    def or_(self: SelectorIR, *others: SelectorIR) -> SelectorIR:
        """Syntax sugar for `BinarySelector(left=self, op=Or(), right=others[0])`.

        - Noop when `others` is empty
        - Reduction when `len(others) > 1`
        """
        # NOTE: `Self@SelectorIR.or_(*SelectorIR)` is obliterating pylance
        from narwhals._plan.expressions.operators import Or

        return reduce(Or().to_binary_selector, others, self)

    def invert(self) -> SelectorIR:
        """Return the complement of this selector."""
        # NOTE: We have a few special simplifications:
        # - All                        -> Empty
        # - Empty                      -> All
        # - InvertSelector[SelectorIR] -> SelectorIR
        from narwhals._plan.expressions.selectors import InvertSelector

        return InvertSelector(selector=self)


# TODO @dangotbanned: Final polish on class-level doc
@final
class NamedIR(Immutable, Generic[ExprIRT_co]):
    """A post-expansion representation of an expression.

    Each *top-level* `ExprIR` goes on a journey of [expression expansion].

    That begins in a [projection context], like `select`:

        df.select(nw.col("weight", "height").mean().name.prefix("avg_"))

    `NamedIR` *wraps* a `ExprIR` as a promise we've resolved it against a schema and:
    1. Converted selectors into column references
    2. Checked all column references exist in the schema
    3. Finished any renaming operations without producing duplicates

    Each *expanded* expression produces one or more `NamedIR[ExprIR]`s, stripped of all:

        SelectorIR
        Alias
        RenameAlias
        KeepName

    And in their place, we get a *single* output column name, bound to the schema we expanded with.

    ## Examples
    >>> import narwhals as nw
    >>> import narwhals._plan as nwp
    >>> import narwhals._plan.selectors as ncs
    >>> from tests.plan.utils import Frame
    >>> schema = {
    ...     "name": nw.String(),
    ...     "birthdate": nw.String(),
    ...     "weight": nw.Float64(),
    ...     "height": nw.String(),
    ... }
    >>> df = Frame.from_mapping(schema)

    Suppose we have this expression:
    >>> expr = (
    ...     nwp.col("weight", "height")
    ...     .mean()
    ...     .name.prefix("avg_")
    ...     .over(ncs.matches(r"date").str.slice(0, 3))
    ... )

    Before expansion we have 1 `ExprIR` with:
    - a root selector
    - a renaming operation
    - another selector inside a window
    >>> expr._ir
    ncs.by_name('weight', 'height').mean().name.prefix('avg_').over([ncs.matches('date').str.slice()])

    After expansion, its `col`(s) all the way down, multiple outputs + the names are ready too:
    >>> df.project(expr)  # doctest: +NORMALIZE_WHITESPACE
    (avg_weight=col('weight').mean().over([col('birthdate').str.slice()]),
     avg_height=col('height').mean().over([col('birthdate').str.slice()]))

    [projection context]: https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/#contexts
    [expression expansion]: https://docs.pola.rs/user-guide/expressions/expression-expansion/
    """

    __slots__ = ("expr", "name")
    # NOTE: https://discuss.python.org/t/make-replace-stop-interfering-with-variance-inference/96092
    expr: ExprIRT_co  # type: ignore[misc]
    """The expanded expression.

    For an expression with multiple outputs:
    >>> import narwhals._plan as nw
    >>> expr_ir = nw.col("a", "b").first()._ir
    >>> expr_ir
    col('a', 'b').first()

    We expand each output into a new `NamedIR`:
    >>> from narwhals._plan import expressions as ir
    >>> for name in expr_ir.expr.selector.names:
    ...     expanded = expr_ir.__replace__(expr=ir.col(name))
    ...     print(f"{ir.NamedIR(expr=expanded, name=name)!r}")
    ...     #                   ^^^^
    a=col('a').first()
    b=col('b').first()
    """

    name: str
    """The resolved output column name.

    When an expression contains one or more renaming operations, like:
    >>> import narwhals._plan as nw
    >>> expr_ir = (
    ...     nw.col("a")
    ...     .cum_sum()
    ...     .alias("b")
    ...     .over(order_by="c")
    ...     .name.suffix("_cum_sum")
    ...     ._ir
    ... )
    >>> expr_ir
    col('a').cum_sum().alias('b').over(order_by=[col('c')]).name.suffix('_cum_sum')

    `name` represents the column name *produced by* the original expression:
    >>> expr_ir.meta.output_name()
    'b_cum_sum'
    """

    def __init__(self, name: str, expr: ExprIRT_co) -> None:
        _OBJ_SETATTR(self, "name", name)
        _OBJ_SETATTR(self, "expr", expr)

    def map_ir(self, function: MapIR, /) -> NamedIR[ExprIR]:
        """Transform the wrapped expression by applying a function to all nodes in it's graph.

        See `ExprIR.map_ir` for examples.

        Arguments:
            function: A single argument [idempotent] function.

                Called *recursively* on any inputs to `self.expr` and then `self.expr` itself.

        Warning:
            If `function` performs any kind of renaming operation, use `ExprIR.map_ir` **before**
            selector expansion instead.

        Returns:
            Either
            - A new `NamedIR`, with any changes made as a result of `function`
            - The same `NamedIR` (by identity)

        [idempotent]: https://en.wikipedia.org/wiki/Idempotence#Computer_science_meaning
        """
        return NamedIR.__replace__(self, expr=function(self.expr.map_ir(function)))

    def __replace__(
        self, *, expr: ExprIR | None = None, name: str | None = None
    ) -> NamedIR[ExprIR]:
        name = self.name if name is None else name
        if (changed := expr) and changed != self.expr:
            return NamedIR(name, changed)
        if name != self.name:
            return NamedIR(name, self.expr)
        return self

    def __repr__(self) -> str:
        return f"{self.name}={self.expr!r}"

    def _repr_html_(self) -> str:  # pragma: no cover
        return f"<b>{self.name}</b>={self.expr._repr_html_()}"

    def is_column(self, *, allow_aliasing: bool = False) -> bool:
        """Return True if wrapping a single `Column` node.

        Arguments:
            allow_aliasing: If False (default), any aliasing is not considered to be column selection.
        """
        Column = _import_column()
        ir = self.expr
        return isinstance(ir, Column) and ((self.name == ir.name) or allow_aliasing)

    @unstable
    def resolve_dtype(self, schema: FrozenSchema) -> DType:  # pragma: no cover
        """Get the data type of an expanded expression.

        Arguments:
            schema: The same schema used to project this expression.

        Warning:
            Most `ExprIR`(s) and `Function`(s) support this operation, but
            may be composed of others that cannot until [#3396] is merged.

        [#3396]: https://github.com/narwhals-dev/narwhals/pull/3396
        """
        return self.expr.resolve_dtype(schema)

    def dispatch(
        self, ctx: CompliantNamespace[ct.Frame, ct.ET_co, ct.ST_co], frame: ct.Frame, /
    ) -> ct.ET_co | ct.ST_co:
        # NOTE: Has an intentionally narrower type, to hint that it's top-level
        return self.expr.dispatch(ctx, frame, self.name)


@functools.lru_cache(maxsize=128)
def _matches_dtype(selector: SelectorIR, dtype: IntoDType, /) -> bool:
    return selector._matches_dtype(dtype)


@functools.cache
def _import_column() -> type[Column]:
    from narwhals._plan.expressions import Column

    return Column
