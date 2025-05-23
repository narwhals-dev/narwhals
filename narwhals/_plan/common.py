from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING, TypeVar

from narwhals.utils import Version

if TYPE_CHECKING:
    from typing import Any, Callable, Iterator

    from typing_extensions import Never, Self, TypeAlias, TypeIs, dataclass_transform

    from narwhals._plan.dummy import (
        DummyCompliantExpr,
        DummyExpr,
        DummySelector,
        DummySeries,
    )
    from narwhals._plan.expr import FunctionExpr
    from narwhals._plan.options import FunctionOptions
    from narwhals.typing import NonNestedLiteral

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

Seq: TypeAlias = "tuple[T,...]"
"""Immutable Sequence.

Using instead of `Sequence`, as a `list` can be passed there (can't break immutability promise).
"""

Udf: TypeAlias = "Callable[[Any], Any]"
"""Placeholder for `map_batches(function=...)`."""

IntoExprColumn: TypeAlias = "DummyExpr | DummySeries | str"
IntoExpr: TypeAlias = "NonNestedLiteral | IntoExprColumn"


@dataclass_transform(kw_only_default=True, frozen_default=True)
class Immutable:
    __slots__ = ()

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
        slots: tuple[str, ...] = self.__slots__
        return hash(tuple(getattr(self, name) for name in slots))

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        elif type(self) is not type(other):
            return False
        slots: tuple[str, ...] = self.__slots__
        return all(getattr(self, name) == getattr(other, name) for name in slots)

    def __str__(self) -> str:
        # NOTE: Debug repr, closer to constructor
        slots: tuple[str, ...] = self.__slots__
        fields = ", ".join(f"{_field_str(name, getattr(self, name))}" for name in slots)
        return f"{type(self).__name__}({fields})"

    def __init__(self, **kwds: Any) -> None:
        # NOTE: DUMMY CONSTRUCTOR - don't use beyond prototyping!
        # Just need a quick way to demonstrate `ExprIR` and interactions
        slots: set[str] = set(self.__slots__)
        if not slots and not kwds:
            # NOTE: Fastpath for empty slots
            ...
        elif slots == set(kwds):
            # NOTE: Everything is as expected
            for name, value in kwds.items():
                object.__setattr__(self, name, value)
        elif missing := slots.difference(kwds):
            msg = (
                f"{type(self).__name__!r} requires attributes {sorted(slots)!r}, \n"
                f"but missing values for {sorted(missing)!r}"
            )
            raise TypeError(msg)
        else:
            extra = set(kwds).difference(slots)
            msg = (
                f"{type(self).__name__!r} only supports attributes {sorted(slots)!r}, \n"
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

    def to_compliant(self, version: Version = Version.MAIN) -> DummyCompliantExpr:
        from narwhals._plan.dummy import DummyCompliantExpr

        return DummyCompliantExpr._from_ir(self, version)

    @property
    def is_scalar(self) -> bool:
        return False

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


class SelectorIR(ExprIR):
    def to_narwhals(self, version: Version = Version.MAIN) -> DummySelector:
        from narwhals._plan import dummy

        if version is Version.MAIN:
            return dummy.DummySelector._from_ir(self)
        return dummy.DummySelectorV1._from_ir(self)


class ExprIRNamespace(Immutable):
    __slots__ = ("_ir",)

    _ir: ExprIR

    @classmethod
    def from_expr(cls, expr: DummyExpr, /) -> Self:
        return cls(_ir=expr._ir)


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


def is_series(obj: Any) -> TypeIs[DummySeries]:
    from narwhals._plan.dummy import DummySeries

    return isinstance(obj, DummySeries)


def is_iterable_reject(obj: Any) -> TypeIs[str | bytes | DummySeries]:
    from narwhals._plan.dummy import DummySeries

    return isinstance(obj, (str, bytes, DummySeries))
