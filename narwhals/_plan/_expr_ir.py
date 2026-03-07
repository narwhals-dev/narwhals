from __future__ import annotations

from typing import TYPE_CHECKING, Generic, final

from narwhals._plan._dispatch import Dispatcher
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._guards import is_function_expr, is_literal
from narwhals._plan._immutable import Immutable
from narwhals._plan.options import ExprIROptions
from narwhals._plan.typing import ExprIRT_co
from narwhals._utils import Version, unstable
from narwhals.dtypes import DType
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any, ClassVar

    from typing_extensions import Self

    from narwhals._plan.compliant.typing import Ctx, FrameT_contra, R_co
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
        Alias(expr=BinaryExpr(left=Column(name='howdy'), op=Add(), right=Literal(value=ScalarLiteral(dtype=Float64, value=10.5))), name='more')

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

    We can apply functions to transform each node in the graph:

        # TODO: show `map_ir`
        # example(s) ...

    [`meta`]: https://docs.pola.rs/api/python/stable/reference/expressions/meta.html
    """

    # TODO @dangotbanned: How this relates to:
    # - `__init_subclass__(child)`
    # NOTE: Idea for removing this as boilerplate:
    # - https://github.com/narwhals-dev/narwhals/pull/3066#issuecomment-3242037939
    # - https://github.com/narwhals-dev/narwhals/commit/4b0431a234808450a61d8b5260c8769f8cebff7b
    _child: ClassVar[Seq[str]] = ()
    """Nested node names, in iteration order."""

    __expr_ir_config__: ClassVar[ExprIROptions] = ExprIROptions.default()
    """Class-level configuration for how a `Dispatcher` should be built.

    Defined via the (optional) `config` parameter at [subclass-definition time].

    Many expressions simply use the default:

        >>> from narwhals._plan import expressions as ir
        >>> from narwhals._plan.options import ExprIROptions
        >>>
        >>> class Explode(ir.ExprIR, child=("expr",)):
        ... #                                      ^^ # default `config`
        ...     __slots__ = ("expr",)
        ...     expr: ir.ExprIR

        >>> Explode.__expr_ir_config__
        ExprIROptions(is_namespaced=False, override_name='', allow_dispatch=True)

        >>> Explode.__expr_ir_dispatch__
        Dispatcher<explode>

    `config` provides a bit more flexibility when you want it:

        >>> class Explode2(Explode, config=ExprIROptions.renamed("explodier")): ...
        >>> #                       ^^^^^^ custom `config`

        >>> Explode2.__expr_ir_config__
        ExprIROptions(is_namespaced=False, override_name='explodier', allow_dispatch=True)

        >>> Explode2.__expr_ir_dispatch__
        Dispatcher<explodier>

    Keep in mind that `__expr_ir_config__` is inherited:

        >>> class Explode21(Explode2): ...
        >>> Explode21.__expr_ir_dispatch__
        Dispatcher<explodier>

    So we'd need another override to get the default back:

        >>> from narwhals._plan.options import ExplodeOptions
        >>>
        >>> class ExplodeWithOptions(Explode2, config=ExprIROptions.default()):
        ...     __slots__ = ("options",)
        ...     options: ExplodeOptions

        >>> ExplodeWithOptions.__expr_ir_dispatch__
        Dispatcher<explode_with_options>

    Warning:
        This attribute should be considered immutable once [`__init_subclass__`] finishes executing.

    [`__init_subclass__`]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
    [subclass-definition time]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
    """

    # TODO @dangotbanned: How this relates to:
    # - `dispatch`
    # - `ExprDispatch` (compliant)
    __expr_ir_dispatch__: ClassVar[Dispatcher[Self]]

    # TODO @dangotbanned: How this relates to:
    # - `__init_subclass__(dtype)`
    # - `resolve_dtype`
    # - `dtypes_mapper.py`
    __expr_ir_dtype__: ClassVar[ResolveDType[Self]] = ResolveDType()

    # NOTE: May need to add docs, even though it won't show in an IDE
    def __init_subclass__(
        cls: type[Self],
        *,
        child: Seq[str] = (),
        config: ExprIROptions | None = None,
        dtype: DType | ResolveDType[Any] | Callable[[Self], DType] | None = None,
        **kwds: Any,
    ) -> None:
        super().__init_subclass__(**kwds)
        if child:
            cls._child = child
        if config:
            cls.__expr_ir_config__ = config
        cls.__expr_ir_dispatch__ = Dispatcher.from_expr_ir(cls)
        if dtype is not None:
            if isinstance(dtype, DType):
                dtype = ResolveDType.just_dtype(dtype)
            elif not isinstance(dtype, ResolveDType):
                dtype = ResolveDType.expr_ir.visitor(dtype)  # pragma: no cover
            cls.__expr_ir_dtype__ = dtype

    # TODO @dangotbanned: Really deserves a good doc
    def dispatch(
        self: Self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str, /
    ) -> R_co:
        """Evaluate this expression in `frame`, using implementation(s) provided by `ctx`."""
        return self.__expr_ir_dispatch__(self, ctx, frame, name)

    def resolve_dtype(self: Self, schema: FrozenSchema) -> DType:
        """Get the data type of an expanded expression.

        Arguments:
            schema: The same schema used to project this expression.
        """
        return self.__expr_ir_dtype__(self, schema)

    def to_narwhals(self, version: Version = Version.MAIN) -> Expr:
        """Convert this `ExprIR` into an `Expr`."""
        from narwhals._plan import expr

        tp = expr.Expr if version is Version.MAIN else expr.ExprV1
        return tp._from_ir(self)

    def to_selector_ir(self) -> SelectorIR:
        """Try to convert this `ExprIR` into a `SelectorIR`.

        This is a noop for `SelectorIR`, and raises for all `ExprIR` *except* `Column`.
        """
        msg = f"cannot turn `{self!r}` into a selector"
        raise InvalidOperationError(msg)

    # TODO @dangotbanned: do another pass on the phrasing
    @property
    def is_scalar(self) -> bool:
        """Return True if this leaf produces a single value.

        Some expressions are always scalar:

            >>> import narwhals._plan as nw
            >>> nw.len()._ir.is_scalar
            True
            >>> nw.lit(123)._ir.is_scalar
            True

        Others are never scalar:

            >>> nw.col("a")._ir.is_scalar
            False
            >>> nw.int_range(0, 10)._ir.is_scalar
            False

        Many depend on the scalar-ness of child expressions:

            >>> (nw.col("a") + nw.len())._ir.is_scalar
            False
            >>> (nw.col("a").first() + nw.len())._ir.is_scalar
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

    # TODO @dangotbanned: Docs should explain a bit + give an example
    # E.g. rewrite `sum_horizontal("a", "b", "c")` -> `col("a") + col("b") + col("c")`
    # We have multiple versions of that in various backends, but it would be easy to write a backend-agnostic version
    def map_ir(self, function: MapIR, /) -> ExprIR:
        """Apply `function` to each child node, returning a new `ExprIR`.

        See [`polars_plan::plans::iterator::Expr.map_expr`] and [`polars_plan::plans::visitor::visitors`].

        [`polars_plan::plans::iterator::Expr.map_expr`]: https://github.com/pola-rs/polars/blob/0fa7141ce718c6f0a4d6ae46865c867b177a59ed/crates/polars-plan/src/plans/iterator.rs#L152-L159
        [`polars_plan::plans::visitor::visitors`]: https://github.com/pola-rs/polars/blob/0fa7141ce718c6f0a4d6ae46865c867b177a59ed/crates/polars-plan/src/plans/visitor/visitors.rs
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


# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
no_dispatch = ExprIROptions.no_dispatch


class SelectorIR(ExprIR, config=no_dispatch()):
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
        """**WARNING**: don't use renaming ops here, or `self.name` is invalid."""
        return NamedIR.__replace__(self, expr=function(self.expr.map_ir(function)))

    def __repr__(self) -> str:
        return f"{self.name}={self.expr!r}"

    def _repr_html_(self) -> str:  # pragma: no cover
        return f"<b>{self.name}</b>={self.expr._repr_html_()}"

    def is_elementwise_top_level(self) -> bool:  # pragma: no cover
        """Return True if the outermost node is elementwise.

        Based on [`polars_plan::plans::aexpr::properties::AExpr.is_elementwise_top_level`]

        This check:
        - Is not recursive
        - Is not valid on `ExprIR` *prior* to being expanded

        [`polars_plan::plans::aexpr::properties::AExpr.is_elementwise_top_level`]: https://github.com/pola-rs/polars/blob/2c7a3e77f0faa37c86a3745db4ef7707ae50c72e/crates/polars-plan/src/plans/aexpr/properties.rs#L16-L44
        """
        from narwhals._plan.expressions import expr

        ir = self.expr
        if is_function_expr(ir):
            return ir.options.is_elementwise()
        if is_literal(ir):
            return ir.is_scalar
        return isinstance(ir, (expr.BinaryExpr, expr.Column, expr.TernaryExpr, expr.Cast))

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
