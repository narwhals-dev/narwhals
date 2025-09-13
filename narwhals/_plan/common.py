from __future__ import annotations

import datetime as dt
import re
import sys
from collections.abc import Iterable
from decimal import Decimal
from operator import attrgetter
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast, overload

from narwhals._plan._guards import is_function_expr, is_iterable_reject, is_literal
from narwhals._plan._immutable import Immutable
from narwhals._plan.options import ExprIROptions
from narwhals._plan.typing import (
    DTypeT,
    ExprIRT,
    ExprIRT2,
    FunctionT,
    MapIR,
    NonNestedDTypeT,
    OneOrIterable,
    Seq,
)
from narwhals.dtypes import DType
from narwhals.utils import Version

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Callable

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.expr import Expr, Selector
    from narwhals._plan.expressions.expr import Alias, Cast, Column
    from narwhals._plan.meta import MetaNamespace
    from narwhals._plan.protocols import Ctx, FrameT_contra, R_co
    from narwhals.typing import NonNestedDType, NonNestedLiteral


if sys.version_info >= (3, 13):
    from copy import replace as replace  # noqa: PLC0414
else:

    def replace(obj: T, /, **changes: Any) -> T:
        cls = obj.__class__
        func = getattr(cls, "__replace__", None)
        if func is None:
            msg = f"replace() does not support {cls.__name__} objects"
            raise TypeError(msg)
        return func(obj, **changes)  # type: ignore[no-any-return]


T = TypeVar("T")
Incomplete: TypeAlias = "Any"


def pascal_to_snake_case(s: str) -> str:
    """Convert a PascalCase, camelCase string to snake_case.

    Adapted from https://github.com/pydantic/pydantic/blob/f7a9b73517afecf25bf898e3b5f591dffe669778/pydantic/alias_generators.py#L43-L62
    """
    # Handle the sequence of uppercase letters followed by a lowercase letter
    snake = _PATTERN_UPPER_LOWER.sub(_re_repl_snake, s)
    # Insert an underscore between a lowercase letter and an uppercase letter
    return _PATTERN_LOWER_UPPER.sub(_re_repl_snake, snake).lower()


_PATTERN_UPPER_LOWER = re.compile(r"([A-Z]+)([A-Z][a-z])")
_PATTERN_LOWER_UPPER = re.compile(r"([a-z])([A-Z])")


def _re_repl_snake(match: re.Match[str], /) -> str:
    return f"{match.group(1)}_{match.group(2)}"


def _dispatch_method_name(tp: type[ExprIRT | FunctionT]) -> str:
    config = tp.__expr_ir_config__
    name = config.override_name or pascal_to_snake_case(tp.__name__)
    return f"{ns}.{name}" if (ns := getattr(config, "accessor_name", "")) else name


def _dispatch_getter(tp: type[ExprIRT | FunctionT]) -> Callable[[Any], Any]:
    getter = attrgetter(_dispatch_method_name(tp))
    if tp.__expr_ir_config__.origin == "expr":
        return getter
    return lambda ctx: getter(ctx.__narwhals_namespace__())


def _dispatch_generate(
    tp: type[ExprIRT], /
) -> Callable[[Incomplete, ExprIRT, Incomplete, str], Incomplete]:
    if not tp.__expr_ir_config__.allow_dispatch:

        def _(ctx: Any, /, node: ExprIRT, _: Any, name: str) -> Any:
            msg = (
                f"{tp.__name__!r} should not appear at the compliant-level.\n\n"
                f"Make sure to expand all expressions first, got:\n{ctx!r}\n{node!r}\n{name!r}"
            )
            raise TypeError(msg)

        return _
    getter = _dispatch_getter(tp)

    def _(ctx: Any, /, node: ExprIRT, frame: Any, name: str) -> Any:
        return getter(ctx)(node, frame, name)

    return _


def _map_ir_child(obj: ExprIR | Seq[ExprIR], fn: MapIR, /) -> ExprIR | Seq[ExprIR]:
    return obj.map_ir(fn) if isinstance(obj, ExprIR) else tuple(e.map_ir(fn) for e in obj)


class ExprIR(Immutable):
    """Anything that can be a node on a graph of expressions."""

    _child: ClassVar[Seq[str]] = ()
    """Nested node names, in iteration order."""

    __expr_ir_config__: ClassVar[ExprIROptions] = ExprIROptions.default()
    __expr_ir_dispatch__: ClassVar[
        staticmethod[[Incomplete, Self, Incomplete, str], Incomplete]
    ]

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
        cls.__expr_ir_dispatch__ = staticmethod(_dispatch_generate(cls))

    def dispatch(
        self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str, /
    ) -> R_co:
        """Evaluate expression in `frame`, using `ctx` for implementation(s)."""
        return self.__expr_ir_dispatch__(ctx, cast("Self", self), frame, name)  # type: ignore[no-any-return]

    def to_narwhals(self, version: Version = Version.MAIN) -> Expr:
        from narwhals._plan import expr

        tp = expr.Expr if version is Version.MAIN else expr.ExprV1
        return tp._from_ir(self)

    @property
    def is_scalar(self) -> bool:
        return False

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


class SelectorIR(ExprIR, config=ExprIROptions.no_dispatch()):
    def to_narwhals(self, version: Version = Version.MAIN) -> Selector:
        from narwhals._plan import expr

        if version is Version.MAIN:
            return expr.Selector._from_ir(self)
        return expr.SelectorV1._from_ir(self)

    def matches_column(self, name: str, dtype: DType) -> bool:
        """Return True if we can select this column.

        - Thinking that we could get more cache hits on an individual column basis.
        - May also be more efficient to not iterate over the schema for every selector
          - Instead do one pass, evaluating every selector against a single column at a time
        """
        raise NotImplementedError(type(self))


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


def py_to_narwhals_dtype(obj: NonNestedLiteral, version: Version = Version.MAIN) -> DType:
    dtypes = version.dtypes
    mapping: dict[type[NonNestedLiteral], type[NonNestedDType]] = {
        int: dtypes.Int64,
        float: dtypes.Float64,
        str: dtypes.String,
        bool: dtypes.Boolean,
        dt.datetime: dtypes.Datetime,
        dt.date: dtypes.Date,
        dt.time: dtypes.Time,
        dt.timedelta: dtypes.Duration,
        bytes: dtypes.Binary,
        Decimal: dtypes.Decimal,
        type(None): dtypes.Unknown,
    }
    return mapping.get(type(obj), dtypes.Unknown)()


@overload
def into_dtype(dtype: type[NonNestedDTypeT], /) -> NonNestedDTypeT: ...
@overload
def into_dtype(dtype: DTypeT, /) -> DTypeT: ...
def into_dtype(dtype: DTypeT | type[NonNestedDTypeT], /) -> DTypeT | NonNestedDTypeT:
    # NOTE: `mypy` needs to learn intersections
    if isinstance(dtype, type) and issubclass(dtype, DType):
        return cast("NonNestedDTypeT", dtype())
    return dtype


# TODO @dangotbanned: Review again and try to work around (https://github.com/microsoft/pyright/issues/10673#issuecomment-3033789021)
# The issue is `T` possibly being `Iterable`
# Ignoring here still leaks the issue to the caller, where you need to annotate the base case
def flatten_hash_safe(iterable: Iterable[OneOrIterable[T]], /) -> Iterator[T]:
    """Fully unwrap all levels of nesting.

    Aiming to reduce the chances of passing an unhashable argument.
    """
    for element in iterable:
        if isinstance(element, Iterable) and not is_iterable_reject(element):
            yield from flatten_hash_safe(element)
        else:
            yield element  # type: ignore[misc]
