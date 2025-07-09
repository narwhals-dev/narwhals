from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, overload

from narwhals._plan.typing import (
    DTypeT,
    ExprIRT,
    ExprIRT2,
    ExprT,
    IRNamespaceT,
    MapIR,
    NamedOrExprIRT,
    NativeSeriesT,
    NonNestedDTypeT,
    Ns,
    Seq,
)
from narwhals._utils import _hasattr_static
from narwhals.dtypes import DType
from narwhals.utils import Version

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Callable, Literal

    from typing_extensions import Never, Self, TypeIs, dataclass_transform

    from narwhals._plan import expr
    from narwhals._plan.dummy import DummyExpr, DummySelector, DummySeries
    from narwhals._plan.expr import (
        Agg,
        BinaryExpr,
        Cast,
        Column,
        FunctionExpr,
        WindowExpr,
    )
    from narwhals._plan.meta import IRMetaNamespace
    from narwhals._plan.options import FunctionOptions
    from narwhals._plan.protocols import DummyCompliantSeries
    from narwhals.typing import NonNestedDType, NonNestedLiteral

else:
    # NOTE: This isn't important to the proposal, just wanted IDE support
    # for the **temporary** constructors.
    # It is interesting how much boilerplate this avoids though 🤔
    # https://docs.python.org/3/library/typing.html#typing.dataclass_transform
    def dataclass_transform(
        *,
        eq_default: bool = True,
        order_default: bool = False,
        kw_only_default: bool = False,
        frozen_default: bool = False,
        field_specifiers: tuple[type[Any] | Callable[..., Any], ...] = (),
        **kwargs: Any,
    ) -> Callable[[T], T]:
        def decorator(cls_or_fn: T) -> T:
            cls_or_fn.__dataclass_transform__ = {
                "eq_default": eq_default,
                "order_default": order_default,
                "kw_only_default": kw_only_default,
                "frozen_default": frozen_default,
                "field_specifiers": field_specifiers,
                "kwargs": kwargs,
            }
            return cls_or_fn

        return decorator


T = TypeVar("T")

_IMMUTABLE_HASH_NAME: Literal["__immutable_hash_value__"] = "__immutable_hash_value__"


@dataclass_transform(kw_only_default=True, frozen_default=True)
class Immutable:
    __slots__ = (_IMMUTABLE_HASH_NAME,)
    __immutable_hash_value__: int

    @property
    def __immutable_keys__(self) -> Iterator[str]:
        slots: tuple[str, ...] = self.__slots__
        for name in slots:
            if name != _IMMUTABLE_HASH_NAME:
                yield name

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        for name in self.__immutable_keys__:
            yield getattr(self, name)

    @property
    def __immutable_items__(self) -> Iterator[tuple[str, Any]]:
        for name in self.__immutable_keys__:
            yield name, getattr(self, name)

    @property
    def __immutable_hash__(self) -> int:
        if hasattr(self, _IMMUTABLE_HASH_NAME):
            return self.__immutable_hash_value__
        hash_value = hash((self.__class__, *self.__immutable_values__))
        object.__setattr__(self, _IMMUTABLE_HASH_NAME, hash_value)
        return self.__immutable_hash_value__

    def __setattr__(self, name: str, value: Never) -> Never:
        msg = f"{type(self).__name__!r} is immutable, {name!r} cannot be set."
        raise AttributeError(msg)

    def __init_subclass__(cls, *args: Any, **kwds: Any) -> None:
        super().__init_subclass__(*args, **kwds)
        if cls.__slots__:
            ...
        else:
            cls.__slots__ = ()

    def __hash__(self) -> int:
        return self.__immutable_hash__

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        elif type(self) is not type(other):
            return False
        return all(
            getattr(self, key) == getattr(other, key) for key in self.__immutable_keys__
        )

    def __str__(self) -> str:
        # NOTE: Debug repr, closer to constructor
        fields = ", ".join(f"{_field_str(k, v)}" for k, v in self.__immutable_items__)
        return f"{type(self).__name__}({fields})"

    def __init__(self, **kwds: Any) -> None:
        # NOTE: DUMMY CONSTRUCTOR - don't use beyond prototyping!
        # Just need a quick way to demonstrate `ExprIR` and interactions
        required: set[str] = set(self.__immutable_keys__)
        if not required and not kwds:
            # NOTE: Fastpath for empty slots
            ...
        elif required == set(kwds):
            # NOTE: Everything is as expected
            for name, value in kwds.items():
                object.__setattr__(self, name, value)
        elif missing := required.difference(kwds):
            msg = (
                f"{type(self).__name__!r} requires attributes {sorted(required)!r}, \n"
                f"but missing values for {sorted(missing)!r}"
            )
            raise TypeError(msg)
        else:
            extra = set(kwds).difference(required)
            msg = (
                f"{type(self).__name__!r} only supports attributes {sorted(required)!r}, \n"
                f"but got unknown arguments {sorted(extra)!r}"
            )
            raise TypeError(msg)


def _field_str(name: str, value: Any) -> str:
    if isinstance(value, tuple):
        inner = ", ".join(f"{v}" for v in value)
        return f"{name}=[{inner}]"
    elif isinstance(value, str):
        return f"{name}={value!r}"
    return f"{name}={value}"


class ExprIR(Immutable):
    """Anything that can be a node on a graph of expressions."""

    def to_narwhals(self, version: Version = Version.MAIN) -> DummyExpr:
        from narwhals._plan import dummy

        if version is Version.MAIN:
            return dummy.DummyExpr._from_ir(self)
        return dummy.DummyExprV1._from_ir(self)

    def to_compliant(self, plx: Ns[ExprT], /) -> ExprT:
        raise NotImplementedError

    @property
    def is_scalar(self) -> bool:
        return False

    def map_ir(self, function: MapIR, /) -> ExprIR:
        """Apply `function` to each child node, returning a new `ExprIR`.

        See [`polars_plan::plans::iterator::Expr.map_expr`] and [`polars_plan::plans::visitor::visitors`].

        [`polars_plan::plans::iterator::Expr.map_expr`]: https://github.com/pola-rs/polars/blob/0fa7141ce718c6f0a4d6ae46865c867b177a59ed/crates/polars-plan/src/plans/iterator.rs#L152-L159
        [`polars_plan::plans::visitor::visitors`]: https://github.com/pola-rs/polars/blob/0fa7141ce718c6f0a4d6ae46865c867b177a59ed/crates/polars-plan/src/plans/visitor/visitors.rs
        """
        msg = f"Need to handle recursive visiting first for {type(self).__qualname__!r}!\n\n{self!r}"
        raise NotImplementedError(msg)

    def iter_left(self) -> Iterator[ExprIR]:
        """Yield nodes root->leaf.

        Examples:
            >>> from narwhals._plan import demo as nwd
            >>>
            >>> a = nwd.col("a")
            >>> b = a.alias("b")
            >>> c = b.min().alias("c")
            >>> d = c.over(nwd.col("e"), nwd.col("f"))
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
        yield self

    def iter_right(self) -> Iterator[ExprIR]:
        """Yield nodes leaf->root.

        Note:
            Identical to `iter_left` for root nodes.

        Examples:
            >>> from narwhals._plan import demo as nwd
            >>>
            >>> a = nwd.col("a")
            >>> b = a.alias("b")
            >>> c = b.min().alias("c")
            >>> d = c.over(nwd.col("e"), nwd.col("f"))
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
    def meta(self) -> IRMetaNamespace:
        from narwhals._plan.meta import IRMetaNamespace

        return IRMetaNamespace(_ir=self)

    def cast(self, dtype: DType) -> Cast:
        from narwhals._plan.expr import Cast

        return Cast(expr=self, dtype=dtype)

    def _repr_html_(self) -> str:
        return self.__repr__()


class SelectorIR(ExprIR):
    def to_narwhals(self, version: Version = Version.MAIN) -> DummySelector:
        from narwhals._plan import dummy

        if version is Version.MAIN:
            return dummy.DummySelector._from_ir(self)
        return dummy.DummySelectorV1._from_ir(self)

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
        from narwhals._plan.expr import col

        return NamedIR(expr=col(name), name=name)

    def map_ir(self, function: MapIR, /) -> NamedIR[ExprIR]:
        """**WARNING**: don't use renaming ops here, or `self.name` is invalid."""
        return self.with_expr(function(self.expr.map_ir(function)))

    def with_expr(self, expr: ExprIRT2, /) -> NamedIR[ExprIRT2]:
        if expr == self.expr:
            return cast("NamedIR[ExprIRT2]", self)
        return NamedIR(expr=expr, name=self.name)

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
        from narwhals._plan import expr

        ir = self.expr
        if is_function_expr(ir):
            return ir.options.is_elementwise()
        if is_literal(ir):
            return ir.is_scalar
        return isinstance(ir, (expr.BinaryExpr, expr.Column, expr.Ternary, expr.Cast))


class IRNamespace(Immutable):
    __slots__ = ("_ir",)
    _ir: ExprIR

    @classmethod
    def from_expr(cls, expr: DummyExpr, /) -> Self:
        return cls(_ir=expr._ir)


class ExprNamespace(Immutable, Generic[IRNamespaceT]):
    __slots__ = ("_expr",)
    _expr: DummyExpr

    @property
    def _ir_namespace(self) -> type[IRNamespaceT]:
        raise NotImplementedError

    @property
    def _ir(self) -> IRNamespaceT:
        return self._ir_namespace.from_expr(self._expr)

    def _to_narwhals(self, ir: ExprIR, /) -> DummyExpr:
        return self._expr._from_ir(ir)


class Function(Immutable):
    """Shared by expr functions and namespace functions.

    Only valid in `FunctionExpr.function`

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L114
    """

    @property
    def function_options(self) -> FunctionOptions:
        from narwhals._plan.options import FunctionOptions

        return FunctionOptions.default()

    @property
    def is_scalar(self) -> bool:
        return self.function_options.returns_scalar()

    def to_function_expr(self, *inputs: ExprIR) -> FunctionExpr[Self]:
        from narwhals._plan.expr import FunctionExpr

        # NOTE: Still need to figure out if using a closure is needed
        options = self.function_options
        # https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L442-L450.
        return FunctionExpr(input=inputs, function=self, options=options)


_NON_NESTED_LITERAL_TPS = (
    int,
    float,
    str,
    dt.date,
    dt.time,
    dt.timedelta,
    bytes,
    Decimal,
)


def is_non_nested_literal(obj: Any) -> TypeIs[NonNestedLiteral]:
    return obj is None or isinstance(obj, _NON_NESTED_LITERAL_TPS)


def is_expr(obj: Any) -> TypeIs[DummyExpr]:
    from narwhals._plan.dummy import DummyExpr

    return isinstance(obj, DummyExpr)


def is_column(obj: Any) -> TypeIs[DummyExpr]:
    """Indicate if the given object is a basic/unaliased column.

    https://github.com/pola-rs/polars/blob/a3d6a3a7863b4d42e720a05df69ff6b6f5fc551f/py-polars/polars/_utils/various.py#L164-L168.
    """
    return is_expr(obj) and obj.meta.is_column()


def is_series(
    obj: DummySeries[NativeSeriesT] | Any,
) -> TypeIs[DummySeries[NativeSeriesT]]:
    from narwhals._plan.dummy import DummySeries

    return isinstance(obj, DummySeries)


def is_compliant_series(
    obj: DummyCompliantSeries[NativeSeriesT] | Any,
) -> TypeIs[DummyCompliantSeries[NativeSeriesT]]:
    return _hasattr_static(obj, "__narwhals_series__")


def is_iterable_reject(
    obj: Any,
) -> TypeIs[str | bytes | DummySeries | DummyCompliantSeries]:
    from narwhals._plan.dummy import DummySeries

    return isinstance(obj, (str, bytes, DummySeries)) or is_compliant_series(obj)


def is_regex_projection(name: str) -> bool:
    return name.startswith("^") and name.endswith("$")


def is_window_expr(obj: Any) -> TypeIs[WindowExpr]:
    from narwhals._plan.expr import WindowExpr

    return isinstance(obj, WindowExpr)


def is_function_expr(obj: Any) -> TypeIs[FunctionExpr[Any]]:
    from narwhals._plan.expr import FunctionExpr

    return isinstance(obj, FunctionExpr)


def is_binary_expr(obj: Any) -> TypeIs[BinaryExpr]:
    from narwhals._plan.expr import BinaryExpr

    return isinstance(obj, BinaryExpr)


# TODO @dangotbanned: Rename `Agg` -> `AggExpr`
def is_agg_expr(obj: Any) -> TypeIs[Agg]:
    from narwhals._plan.expr import Agg

    return isinstance(obj, Agg)


def is_aggregation(obj: Any) -> TypeIs[Agg | FunctionExpr[Any]]:
    """Superset of `ExprIR.is_scalar`, excludes literals & len."""
    return is_agg_expr(obj) or (is_function_expr(obj) and obj.is_scalar)


def is_literal(obj: Any) -> TypeIs[expr.Literal[Any]]:
    from narwhals._plan import expr

    return isinstance(obj, expr.Literal)


def is_horizontal_reduction(obj: FunctionExpr[Any] | Any) -> TypeIs[FunctionExpr[Any]]:
    return is_function_expr(obj) and obj.options.is_input_wildcard_expansion()


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
    if isinstance(dtype, type) and issubclass(dtype, DType):
        # NOTE: `mypy` needs to learn intersections
        return dtype()  # type: ignore[return-value]
    return dtype


def collect(iterable: Seq[T] | Iterable[T], /) -> Seq[T]:
    """Collect `iterable` into a `tuple`, *iff* it is not one already."""
    return iterable if isinstance(iterable, tuple) else tuple(iterable)


def map_ir(
    origin: NamedOrExprIRT, function: MapIR, *more_functions: MapIR
) -> NamedOrExprIRT:
    """Apply one or more functions, sequentially, to all of `origin`'s children."""
    if more_functions:
        result = origin
        for fn in (function, *more_functions):
            result = result.map_ir(fn)
        return result
    return origin.map_ir(function)


# TODO @dangotbanned: Review again and try to work around (https://github.com/microsoft/pyright/issues/10673#issuecomment-3033789021)
# The issue is `T` possibly being `Iterable`
# Ignoring here still leaks the issue to the caller, where you need to annotate the base case
def flatten_hash_safe(iterable: Iterable[T | Iterable[T]], /) -> Iterator[T]:
    """Fully unwrap all levels of nesting.

    Aiming to reduce the chances of passing an unhashable argument.
    """
    for element in iterable:
        if isinstance(element, Iterable) and not is_iterable_reject(element):
            yield from flatten_hash_safe(element)
        else:
            yield element  # type: ignore[misc]
