from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, final

from narwhals._plan._dispatch import Dispatcher, DispatcherOptions
from narwhals._plan._dtype import IntoResolveDType, ResolveDType
from narwhals._plan._immutable import Immutable
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
    from narwhals._plan.expressions.expr import Alias, Cast, Column
    from narwhals._plan.meta import MetaNamespace
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.selectors import Selector
    from narwhals._plan.typing import ExprIRT, Ignored, MapIR, Seq
    from narwhals.typing import IntoDType


# TODO @dangotbanned: Docs
class ExprIR(Immutable):
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
        Alias(expr=BinaryExpr(left=..., op=Add(), right=Literal(..., value=10.5))), name='more')

    An `ExprIR` is an easily traversable graph, supporting iteration from *root to leaf*:

        >>> root_to_leaf = bigger_ir.iter_left()
        >>> print("\n".join(f"{idx}: {node}" for idx, node in enumerate(root_to_leaf)))
        0: Column(name='howdy')
        1: Literal(value=ScalarLiteral(dtype=Float64, value=10.5))
        2: BinaryExpr(left=..., op=Add(), right=...)
        3: Alias(expr=BinaryExpr(...), name='more')

    And *leaf to root* - both have the same cost:

        >>> leaf_to_root = bigger_ir.iter_right()
        >>> print("\n".join(f"{idx}: {node}" for idx, node in enumerate(leaf_to_root)))
        0: Alias(expr=BinaryExpr(...), name='more')
        1: BinaryExpr(left=..., op=Add(), right=...)
        2: Literal(value=ScalarLiteral(dtype=Float64, value=10.5))
        3: Column(name='howdy')

    That comes in handy for [`meta`] operations, which are available for both `Expr` and `ExprIR`:

        >>> bigger_ir.meta.root_names()
        ['howdy']
        >>> bigger_ir.meta.output_name()
        'more'

    See `ExprIR.map_ir` for other superpowers this gives us.

    [`meta`]: https://docs.pola.rs/api/python/stable/reference/expressions/meta.html
    """

    # TODO @dangotbanned: How this relates to:
    # - `__init_subclass__(child)`
    # NOTE: Idea for removing this as boilerplate:
    # - https://github.com/narwhals-dev/narwhals/pull/3066#issuecomment-3242037939
    # - https://github.com/narwhals-dev/narwhals/commit/4b0431a234808450a61d8b5260c8769f8cebff7b
    _child: ClassVar[Seq[str]] = ()
    """Nested node names, in iteration order."""

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

    # TODO @dangotbanned: How this relates to:
    # - `__init_subclass__(dtype)`
    # - `resolve_dtype`
    # - `dtypes_mapper.py`
    __expr_ir_dtype__: ClassVar[ResolveDType[Self]] = ResolveDType()

    def __init_subclass__(
        cls: type[Self],
        *,
        child: Seq[str] = (),
        dispatch: DispatcherOptions | Literal["no_dispatch"] | None = None,
        dtype: IntoResolveDType[Self] | None = None,
        **_: Any,
    ) -> None:  # TODO @dangotbanned: Do another pass on the short description phrasing
        """[Subclass-definition time] hook for customizing an `ExprIR` class.

        All parameters are optional and will be inherited when not provided to `__init_subclass__`.

        Arguments:
            child: Name(s) of fields that store one or more `ExprIR`(s).
                Stored in `_child`.

                The order of `child` defines iteration order in `iter_left`.

                **Note**: Unlike `__slots__`, a subclass that needs to extend a non-empty `_child`
                must use:

                    child=(*<parent-field-names>, *<more-names>)  # (sorry, need to fix this!)

            dispatch: Instructions defining how to build a `Dispatcher`.
                Stored in `__expr_ir_dispatch__.options`.

            dtype: Defines how a `DType` is derived when `resolve_dtype` is called.
                Stored in `__expr_ir_dtype__`.

                See `IntoResolveDType` and `ResolveDType` for usage.

                **Warning**: This functionality is considered **unstable**.
                Full support depends on [#3396].

        [Subclass-definition time]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
        [#3396]: https://github.com/narwhals-dev/narwhals/pull/3396
        """
        super().__init_subclass__(**_)
        if child:
            cls._child = child
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

        This is a noop for `SelectorIR`, and raises for all `ExprIR` *except* `Column`.
        """
        msg = f"cannot turn `{self!r}` into a selector"
        raise InvalidOperationError(msg)

    # TODO @dangotbanned: Rewrite `is_scalar` as a method?
    # Not really an expensive check, but `Function.is_scalar` leads to `FunctionOptions.returns_scalar()`
    # so it would be more consistent
    @property
    def is_scalar(self) -> bool:
        """Return True if this leaf produces a single value.

        Some expressions are always scalar:

            >>> import narwhals._plan as nw
            >>> length = nw.len()
            >>> length._ir.is_scalar
            True
            >>> nw.lit(123)._ir.is_scalar
            True

        Others always output a column:

            >>> column = nw.col("a")
            >>> column._ir.is_scalar
            False
            >>> nw.int_range(0, 10)._ir.is_scalar
            False

        Many depend on the scalar-ness of child expressions, and require traversal:

            >>> (column + length)._ir.is_scalar
            False
            >>> (column.first() + length)._ir.is_scalar
            True
        """
        return False

    def needs_expansion(self) -> bool:
        """Return True if this expression contains selectors.

        Examples:
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
        if not self._child:
            return function(self)
        children = ((name, getattr(self, name)) for name in self._child)
        changed = {name: _map_ir_child(child, function) for name, child in children}
        return function(self.__replace__(**changed))

    def iter_left(self) -> Iterator[ExprIR]:
        """Yield nodes root->leaf.

        Examples:
            >>> from narwhals import _plan as nw
            >>>
            >>> a = nw.col("a")
            >>> b = a.alias("b")
            >>> c = b.min().alias("c")
            >>> d = c.over(nw.col("e"), nw.col("f"))
            >>>
            >>> list(a._ir.iter_left())
            [col('a')]
            >>>
            >>> list(b._ir.iter_left())
            [col('a'), col('a').alias('b')]
            >>>
            >>> list(c._ir.iter_left())
            [col('a'), col('a').alias('b'), col('a').alias('b').min(), col('a').alias('b').min().alias('c')]
            >>>
            >>> list(d._ir.iter_left())
            [col('a'), col('a').alias('b'), col('a').alias('b').min(), col('a').alias('b').min().alias('c'), col('e'), col('f'), col('a').alias('b').min().alias('c').over([col('e'), col('f')])]
        """
        for name in self._child:
            child: ExprIR | Seq[ExprIR] = getattr(self, name)
            if isinstance(child, ExprIR):
                yield from child.iter_left()
            else:
                for node in child:
                    yield from node.iter_left()
        yield self

    def iter_right(self) -> Iterator[ExprIR]:
        """Yield nodes leaf->root.

        Note:
            Identical to `iter_left` for root nodes.

        Examples:
            >>> from narwhals import _plan as nw
            >>>
            >>> a = nw.col("a")
            >>> b = a.alias("b")
            >>> c = b.min().alias("c")
            >>> d = c.over(nw.col("e"), nw.col("f"))
            >>>
            >>> list(a._ir.iter_right())
            [col('a')]
            >>>
            >>> list(b._ir.iter_right())
            [col('a').alias('b'), col('a')]
            >>>
            >>> list(c._ir.iter_right())
            [col('a').alias('b').min().alias('c'), col('a').alias('b').min(), col('a').alias('b'), col('a')]
            >>>
            >>> list(d._ir.iter_right())
            [col('a').alias('b').min().alias('c').over([col('e'), col('f')]), col('f'), col('e'), col('a').alias('b').min().alias('c'), col('a').alias('b').min(), col('a').alias('b'), col('a')]
        """
        yield self
        for name in reversed(self._child):
            child: ExprIR | Seq[ExprIR] = getattr(self, name)
            if isinstance(child, ExprIR):
                yield from child.iter_right()
            else:
                for node in reversed(child):  # pragma: no cover
                    yield from node.iter_right()

    # TODO @dangotbanned: Can this be factored out?
    # - Only `FunctionExpr`, `StructExpr` do anything fancy here
    # - Everything else just iterates the first child
    #   - `FunctionExpr` stores `input: Seq[ExprIR]` there, and need to stop on the first element
    def iter_output_name(self) -> Iterator[ExprIR]:
        """Override for different iteration behavior in `ExprIR.meta.output_name`.

        Note:
            Identical to `iter_right` by default.
        """
        yield from self.iter_right()

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


def _map_ir_child(obj: ExprIR | Seq[ExprIR], fn: MapIR, /) -> ExprIR | Seq[ExprIR]:
    return obj.map_ir(fn) if isinstance(obj, ExprIR) else tuple(e.map_ir(fn) for e in obj)


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
    name: str

    @staticmethod
    def from_name(name: str, /) -> NamedIR[Column]:
        """Construct as a simple, unaliased `col(name)` expression.

        Intended to be used in `with_columns` from a `FrozenSchema`'s keys.
        """
        from narwhals._plan.expressions.expr import col

        return NamedIR(expr=col(name), name=name)

    @staticmethod
    def from_ir(expr: ExprIRT, /) -> NamedIR[ExprIRT]:
        """Construct from an already expanded `ExprIR`.

        Should be cheap to get the output name from cache, but will raise if used
        without care.
        """
        return NamedIR(expr=expr, name=expr.meta.output_name(raise_if_undetermined=True))

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
    return NamedIR(expr=expr, name=name)
