from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, final

from narwhals._plan._dispatch import Dispatcher, DispatcherOptions
from narwhals._plan._dtype import IntoResolveDType, ResolveDType
from narwhals._plan._immutable import Immutable
from narwhals._plan._meta import ExprIRMeta
from narwhals._plan._nodes import ExprTraverser
from narwhals._plan.typing import ExprIRT_co
from narwhals._utils import Version, unstable
from narwhals.dtypes import DType
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, ClassVar

    from typing_extensions import Self

    from narwhals._plan.compliant.column import ExprDispatch
    from narwhals._plan.compliant.typing import FrameT_contra, R_co
    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions.expr import Alias, Cast
    from narwhals._plan.meta import MetaNamespace
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.selectors import Selector
    from narwhals._plan.typing import ExprIRT, Ignored, MapIR
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

        class When(ExprIR, dispatch=<something-different>): ...

    If nothing there *quite* scratches the itch, override the `dispatch` **method** instead:

        class When(ExprIR):
            def dispatch(self, ctx, frame, name, /):
                return ctx.when_then(self, frame, name)

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
        ctx: ExprDispatch[FrameT_contra, R_co],
        frame: FrameT_contra,
        name: str,
        /,
    ) -> R_co:
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
        from narwhals._plan import expr

        tp = expr.Expr if version is Version.MAIN else expr.ExprV1
        return tp._from_ir(self)

    def to_selector_ir(self) -> SelectorIR:
        """Try to convert this `ExprIR` into a `SelectorIR`.

        >>> import narwhals._plan as nw

        The only valid conversion is for a `Column`:
        >>> column = nw.col("a")._ir
        >>> column
        col('a')

        >>> selector = column.to_selector_ir()
        >>> selector
        ncs.by_name('a', require_all=True)

        For a `SelectorIR`, this is a noop:
        >>> selector.to_selector_ir()
        ncs.by_name('a', require_all=True)

        Everything else will raise:
        >>> alias = nw.col("a").alias("bad")._ir
        >>> alias.to_selector_ir()
        Traceback (most recent call last):
        narwhals.exceptions.InvalidOperationError: cannot turn `col('a').alias('bad')` into a selector
        """
        msg = f"cannot turn `{self!r}` into a selector"
        raise InvalidOperationError(msg)

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
        2. They answer the question using non-node fields (`FunctionExpr.function`, `Literal.value`)
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
            Literal
            Len

        A leaf node with a `name`:

            Alias

        A special case where we can navigate to a name:

            RootSelector(selector=cs.ByName(names=("name",), ...))
            #                                       ^^^^
            #                 Equivalent to `col("name")`

            StructExpr(function=FieldByName(name="name"), ...)
            #                                     ^^^^
            #    Same idea, but with a `Struct` schema

        A leaf node that transforms the name of the above:

            RenameAlias

        A leaf node that requires schema context for expansion, raising
        instead of recursing further:

            KeepName
        """
        yield from self.__expr_ir_nodes__.iter_output_name(self)

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

    def _repr_html_(self) -> str:
        """Return a html representation of this expression, used by [IPython].

        Although this is identical to `__repr__`; a notebook will render the string in a prettier way.

        [IPython]: https://ipython.readthedocs.io/en/stable/config/integrating.html#custom-methods
        """
        return self.__repr__()


class SelectorIR(ExprIR, dispatch="no_dispatch"):
    def to_narwhals(self, version: Version = Version.MAIN) -> Selector:
        from narwhals._plan.selectors import Selector, SelectorV1

        tp = Selector if version is Version.MAIN else SelectorV1
        return tp._from_ir(self)

    # NOTE: Corresponds with `Selector.iter_expand`
    # A longer name is used here to distinguish expression and name-only expansion
    def iter_expand_names(
        self, schema: FrozenSchema, ignored_columns: Ignored
    ) -> Iterator[str]:
        """Yield column names that match the selector, in `schema` order[^1].

        Adapted from [upstream].

        Arguments:
            schema: Target scope to expand the selector in.
            ignored_columns: Names of `group_by` columns, which are excluded[^2] from the result.

        Note:
            [^1]: `ByName`, `ByIndex` return their inputs in given order not in schema order.

        Note:
            [^2]: `ByName`, `ByIndex` will never be ignored.

        [upstream]: https://github.com/pola-rs/polars/blob/2b241543851800595efd343be016b65cdbdd3c9f/crates/polars-plan/src/dsl/selector.rs#L188-L198
        """
        msg = f"{type(self).__name__}.iter_expand_names"
        raise NotImplementedError(msg)

    def matches(self, dtype: IntoDType) -> bool:
        """Return True if we can select this dtype."""
        msg = f"{type(self).__name__}.matches"
        raise NotImplementedError(msg)

    def to_dtype_selector(self) -> Self:
        msg = f"{type(self).__name__}.to_dtype_selector"
        raise NotImplementedError(msg)

    def to_selector_ir(self) -> Self:
        return self

    def needs_expansion(self) -> bool:
        return True


@final
class NamedIR(Immutable, Generic[ExprIRT_co]):
    """Post-projection expansion wrapper for `ExprIR`.

    - Somewhat similar to [`polars_plan::plans::expr_ir::ExprIR`].
    - The [`polars_plan::plans::aexpr::AExpr`] stage has been skipped (*for now*)
      - Parts of that will probably be in here too
      - `AExpr` seems like too much duplication when we won't get the memory allocation benefits in python

    [`polars_plan::plans::expr_ir::ExprIR`]: https://github.com/pola-rs/polars/blob/2c7a3e77f0faa37c86a3745db4ef7707ae50c72e/crates/polars-plan/src/plans/expr_ir.rs#L63-L74
    [`polars_plan::plans::aexpr::AExpr`]: https://github.com/pola-rs/polars/blob/2c7a3e77f0faa37c86a3745db4ef7707ae50c72e/crates/polars-plan/src/plans/aexpr/mod.rs#L145-L231
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

    def __repr__(self) -> str:
        return f"{self.name}={self.expr!r}"

    def _repr_html_(self) -> str:  # pragma: no cover
        return f"<b>{self.name}</b>={self.expr._repr_html_()}"

    def is_column(self, *, allow_aliasing: bool = False) -> bool:
        """Return True if wrapping a single `Column` node.

        Note:
            Multi-output (including selectors) expressions have been expanded at this stage.

        Arguments:
            allow_aliasing: If False (default), any aliasing is not considered to be column selection.
        """
        from narwhals._plan.expressions import Column

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


def named_ir(name: str, expr: ExprIRT, /) -> NamedIR[ExprIRT]:
    """Positional-only `NamedIR` constructor.

    Arguments:
        name: The resolved output column name.
        expr: The expanded expression.

    Examples:
        >>> from narwhals._plan.expressions import col
        >>> short = named_ir("b", col("a"))
        >>> longer = NamedIR(name="b", expr=col("a"))
        >>> short == longer
        True
    """
    return NamedIR(expr=expr, name=name)
