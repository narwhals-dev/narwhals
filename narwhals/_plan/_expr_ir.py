from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from narwhals._plan._dispatch import Dispatcher
from narwhals._plan._guards import is_function_expr, is_literal
from narwhals._plan._immutable import Immutable
from narwhals._plan.common import replace
from narwhals._plan.options import ExprIROptions
from narwhals._plan.typing import ExprIRT
from narwhals.exceptions import InvalidOperationError
from narwhals.utils import Version

if TYPE_CHECKING:
    from collections.abc import Container, Iterator
    from typing import Any, ClassVar

    from typing_extensions import Self

    from narwhals._plan.compliant.typing import Ctx, FrameT_contra, R_co
    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions.expr import Alias, Cast, Column
    from narwhals._plan.meta import MetaNamespace
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.selectors import Selector
    from narwhals._plan.typing import ExprIRT2, MapIR, Seq
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType


class ExprIR(Immutable):
    """Anything that can be a node on a graph of expressions."""

    _child: ClassVar[Seq[str]] = ()
    """Nested node names, in iteration order."""

    __expr_ir_config__: ClassVar[ExprIROptions] = ExprIROptions.default()
    __expr_ir_dispatch__: ClassVar[Dispatcher[Self]]

    def __init_subclass__(
        cls: type[Self],
        *args: Any,
        child: Seq[str] = (),
        config: ExprIROptions | None = None,
        **kwds: Any,
    ) -> None:
        super().__init_subclass__(*args, **kwds)
        if child:
            cls._child = child
        if config:
            cls.__expr_ir_config__ = config
        cls.__expr_ir_dispatch__ = Dispatcher.from_expr_ir(cls)

    def dispatch(
        self: Self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str, /
    ) -> R_co:
        """Evaluate this expression in `frame`, using implementation(s) provided by `ctx`."""
        return self.__expr_ir_dispatch__(self, ctx, frame, name)

    def to_narwhals(self, version: Version = Version.MAIN) -> Expr:
        from narwhals._plan import expr

        tp = expr.Expr if version is Version.MAIN else expr.ExprV1
        return tp._from_ir(self)

    def to_selector_ir(self) -> SelectorIR:
        msg = f"cannot turn `{self!r}` into a selector"
        raise InvalidOperationError(msg)

    @property
    def is_scalar(self) -> bool:
        return False

    def needs_expansion(self) -> bool:
        return any(isinstance(e, SelectorIR) for e in self.iter_left())

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
        return function(replace(self, **changed))

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
                for node in reversed(child):
                    yield from node.iter_right()

    def iter_root_names(self) -> Iterator[ExprIR]:
        """Override for different iteration behavior in `ExprIR.meta.root_names`.

        Note:
            Identical to `iter_left` by default.
        """
        yield from self.iter_left()

    def iter_output_name(self) -> Iterator[ExprIR]:
        """Override for different iteration behavior in `ExprIR.meta.output_name`.

        Note:
            Identical to `iter_right` by default.
        """
        yield from self.iter_right()

    @property
    def meta(self) -> MetaNamespace:
        from narwhals._plan.meta import MetaNamespace

        return MetaNamespace(_ir=self)

    def cast(self, dtype: DType) -> Cast:
        from narwhals._plan.expressions.expr import Cast

        return Cast(expr=self, dtype=dtype)

    def alias(self, name: str) -> Alias:
        from narwhals._plan.expressions.expr import Alias

        return Alias(expr=self, name=name)

    def _repr_html_(self) -> str:
        return self.__repr__()


def _map_ir_child(obj: ExprIR | Seq[ExprIR], fn: MapIR, /) -> ExprIR | Seq[ExprIR]:
    return obj.map_ir(fn) if isinstance(obj, ExprIR) else tuple(e.map_ir(fn) for e in obj)


class SelectorIR(ExprIR, config=ExprIROptions.no_dispatch()):
    def to_narwhals(self, version: Version = Version.MAIN) -> Selector:
        from narwhals._plan.selectors import Selector, SelectorV1

        if version is Version.MAIN:
            return Selector._from_ir(self)
        return SelectorV1._from_ir(self)

    def into_columns(
        self, schema: FrozenSchema, ignored_columns: Container[str]
    ) -> Iterator[str]:
        msg = f"{type(self).__name__}.into_columns"
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


class NamedIR(Immutable, Generic[ExprIRT]):
    """Post-projection expansion wrapper for `ExprIR`.

    - Somewhat similar to [`polars_plan::plans::expr_ir::ExprIR`].
    - The [`polars_plan::plans::aexpr::AExpr`] stage has been skipped (*for now*)
      - Parts of that will probably be in here too
      - `AExpr` seems like too much duplication when we won't get the memory allocation benefits in python

    [`polars_plan::plans::expr_ir::ExprIR`]: https://github.com/pola-rs/polars/blob/2c7a3e77f0faa37c86a3745db4ef7707ae50c72e/crates/polars-plan/src/plans/expr_ir.rs#L63-L74
    [`polars_plan::plans::aexpr::AExpr`]: https://github.com/pola-rs/polars/blob/2c7a3e77f0faa37c86a3745db4ef7707ae50c72e/crates/polars-plan/src/plans/aexpr/mod.rs#L145-L231
    """

    __slots__ = ("expr", "name")
    expr: ExprIRT
    name: str

    @staticmethod
    def from_name(name: str, /) -> NamedIR[Column]:
        """Construct as a simple, unaliased `col(name)` expression.

        Intended to be used in `with_columns` from a `FrozenSchema`'s keys.
        """
        from narwhals._plan.expressions.expr import col

        return NamedIR(expr=col(name), name=name)

    @staticmethod
    def from_ir(expr: ExprIRT2, /) -> NamedIR[ExprIRT2]:
        """Construct from an already expanded `ExprIR`.

        Should be cheap to get the output name from cache, but will raise if used
        without care.
        """
        return NamedIR(expr=expr, name=expr.meta.output_name(raise_if_undetermined=True))

    def map_ir(self, function: MapIR, /) -> Self:
        """**WARNING**: don't use renaming ops here, or `self.name` is invalid."""
        return replace(self, expr=function(self.expr.map_ir(function)))

    def __repr__(self) -> str:
        return f"{self.name}={self.expr!r}"

    def _repr_html_(self) -> str:
        return f"<b>{self.name}</b>={self.expr._repr_html_()}"

    def is_elementwise_top_level(self) -> bool:
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


def named_ir(name: str, expr: ExprIRT, /) -> NamedIR[ExprIRT]:
    return NamedIR(expr=expr, name=name)
