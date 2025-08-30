from __future__ import annotations

import datetime as dt
import re
import sys
from collections.abc import Iterable
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast, overload

from narwhals._plan.typing import (
    Accessor,
    DTypeT,
    ExprIRT,
    ExprIRT2,
    IRNamespaceT,
    MapIR,
    NamedOrExprIRT,
    NativeSeriesT,
    NonNestedDTypeT,
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
    from narwhals._plan.dummy import Expr, Selector, Series
    from narwhals._plan.expr import (
        AggExpr,
        Alias,
        BinaryExpr,
        Cast,
        Column,
        FunctionExpr,
        WindowExpr,
    )
    from narwhals._plan.meta import IRMetaNamespace
    from narwhals._plan.options import FunctionOptions
    from narwhals._plan.protocols import CompliantSeries
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

    def __replace__(self, **changes: Any) -> Self:
        """https://docs.python.org/3.13/library/copy.html#copy.replace"""  # noqa: D415
        if len(changes) == 1:
            k_new, v_new = next(iter(changes.items()))
            # NOTE: Will trigger an attribute error if invalid name
            if getattr(self, k_new) == v_new:
                return self
            changed = dict(self.__immutable_items__)
            # Now we *don't* need to check the key is valid
            changed[k_new] = v_new
        else:
            changed = dict(self.__immutable_items__)
            changed |= changes
        return type(self)(**changed)

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
        if type(self) is not type(other):
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
    if isinstance(value, str):
        return f"{name}={value!r}"
    return f"{name}={value}"


class ExprIR(Immutable):
    """Anything that can be a node on a graph of expressions."""

    # NOTE: No frills solution *first*
    _child: ClassVar[Seq[str]] = ()
    """Nested node names, in iteration order."""

    def to_narwhals(self, version: Version = Version.MAIN) -> Expr:
        from narwhals._plan import dummy

        if version is Version.MAIN:
            return dummy.Expr._from_ir(self)
        return dummy.ExprV1._from_ir(self)

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
        for name in self._child:
            child: ExprIR | Seq[ExprIR] = getattr(self, name)
            if not isinstance(child, ExprIR):
                for nested in child:
                    yield from nested.iter_left()
            else:
                yield from child.iter_left()
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
        for name in reversed(self._child):
            child: ExprIR | Seq[ExprIR] = getattr(self, name)
            if not isinstance(child, ExprIR):
                for nested in reversed(child):
                    yield from nested.iter_right()
            else:
                yield from child.iter_right()

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

    def alias(self, name: str) -> Alias:
        from narwhals._plan.expr import Alias

        return Alias(expr=self, name=name)

    def _repr_html_(self) -> str:
        return self.__repr__()


class SelectorIR(ExprIR):
    def to_narwhals(self, version: Version = Version.MAIN) -> Selector:
        from narwhals._plan import dummy

        if version is Version.MAIN:
            return dummy.Selector._from_ir(self)
        return dummy.SelectorV1._from_ir(self)

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

    @staticmethod
    def from_ir(expr: ExprIRT2, /) -> NamedIR[ExprIRT2]:
        """Construct from an already expanded `ExprIR`.

        Should be cheap to get the output name from cache, but will raise if used
        without care.
        """
        return NamedIR(expr=expr, name=expr.meta.output_name(raise_if_undetermined=True))

    def map_ir(self, function: MapIR, /) -> NamedIR[ExprIR]:
        """**WARNING**: don't use renaming ops here, or `self.name` is invalid."""
        return self.with_expr(function(self.expr.map_ir(function)))

    def with_expr(self, expr: ExprIRT2, /) -> NamedIR[ExprIRT2]:
        return cast("NamedIR[ExprIRT2]", replace(self, expr=expr))

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
    def from_expr(cls, expr: Expr, /) -> Self:
        return cls(_ir=expr._ir)


class ExprNamespace(Immutable, Generic[IRNamespaceT]):
    __slots__ = ("_expr",)
    _expr: Expr

    @property
    def _ir_namespace(self) -> type[IRNamespaceT]:
        raise NotImplementedError

    @property
    def _ir(self) -> IRNamespaceT:
        return self._ir_namespace.from_expr(self._expr)

    def _to_narwhals(self, ir: ExprIR, /) -> Expr:
        return self._expr._from_ir(ir)

    def _with_unary(self, function: Function, /) -> Expr:
        return self._expr._with_unary(function)


def _function_options_default() -> FunctionOptions:
    from narwhals._plan.options import FunctionOptions

    return FunctionOptions.default()


class Function(Immutable):
    """Shared by expr functions and namespace functions.

    Only valid in `FunctionExpr.function`

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L114
    """

    _accessor: ClassVar[Accessor | None] = None
    """Namespace accessor name, if any."""

    _function_options: ClassVar[staticmethod[[], FunctionOptions]] = staticmethod(
        _function_options_default
    )

    @property
    def function_options(self) -> FunctionOptions:
        return self._function_options()

    @property
    def is_scalar(self) -> bool:
        return self.function_options.returns_scalar()

    def to_function_expr(self, *inputs: ExprIR) -> FunctionExpr[Self]:
        from narwhals._plan.expr import FunctionExpr

        return FunctionExpr(input=inputs, function=self, options=self.function_options)

    def __init_subclass__(
        cls,
        *args: Any,
        accessor: Accessor | None = None,
        options: Callable[[], FunctionOptions] | None = None,
        **kwds: Any,
    ) -> None:
        super().__init_subclass__(*args, **kwds)
        if accessor:
            cls._accessor = accessor
        if options:
            cls._function_options = staticmethod(options)

    def __repr__(self) -> str:
        return _function_repr(type(self))


# TODO @dangotbanned: Add caching strategy?
def _function_repr(tp: type[Function], /) -> str:
    name = _pascal_to_snake_case(tp.__name__)
    return f"{ns_name}.{name}" if (ns_name := tp._accessor) else name


def _pascal_to_snake_case(s: str) -> str:
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


def is_expr(obj: Any) -> TypeIs[Expr]:
    from narwhals._plan.dummy import Expr

    return isinstance(obj, Expr)


def is_column(obj: Any) -> TypeIs[Expr]:
    """Indicate if the given object is a basic/unaliased column.

    https://github.com/pola-rs/polars/blob/a3d6a3a7863b4d42e720a05df69ff6b6f5fc551f/py-polars/polars/_utils/various.py#L164-L168.
    """
    return is_expr(obj) and obj.meta.is_column()


def is_series(obj: Series[NativeSeriesT] | Any) -> TypeIs[Series[NativeSeriesT]]:
    from narwhals._plan.dummy import Series

    return isinstance(obj, Series)


def is_compliant_series(
    obj: CompliantSeries[NativeSeriesT] | Any,
) -> TypeIs[CompliantSeries[NativeSeriesT]]:
    return _hasattr_static(obj, "__narwhals_series__")


def is_iterable_reject(obj: Any) -> TypeIs[str | bytes | Series | CompliantSeries]:
    from narwhals._plan.dummy import Series

    return isinstance(obj, (str, bytes, Series)) or is_compliant_series(obj)


def is_window_expr(obj: Any) -> TypeIs[WindowExpr]:
    from narwhals._plan.expr import WindowExpr

    return isinstance(obj, WindowExpr)


def is_function_expr(obj: Any) -> TypeIs[FunctionExpr[Any]]:
    from narwhals._plan.expr import FunctionExpr

    return isinstance(obj, FunctionExpr)


def is_binary_expr(obj: Any) -> TypeIs[BinaryExpr]:
    from narwhals._plan.expr import BinaryExpr

    return isinstance(obj, BinaryExpr)


def is_agg_expr(obj: Any) -> TypeIs[AggExpr]:
    from narwhals._plan.expr import AggExpr

    return isinstance(obj, AggExpr)


def is_aggregation(obj: Any) -> TypeIs[AggExpr | FunctionExpr[Any]]:
    """Superset of `ExprIR.is_scalar`, excludes literals & len."""
    return is_agg_expr(obj) or (is_function_expr(obj) and obj.is_scalar)


def is_literal(obj: Any) -> TypeIs[expr.Literal[Any]]:
    from narwhals._plan import expr

    return isinstance(obj, expr.Literal)


def is_horizontal_reduction(obj: FunctionExpr[Any] | Any) -> TypeIs[FunctionExpr[Any]]:
    return is_function_expr(obj) and obj.options.is_input_wildcard_expansion()


def is_tuple_of(obj: Any, tp: type[T]) -> TypeIs[Seq[T]]:
    return bool(isinstance(obj, tuple) and obj and isinstance(obj[0], tp))


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
