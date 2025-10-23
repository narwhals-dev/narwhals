from __future__ import annotations

import operator
from collections import deque
from functools import reduce
from typing import TYPE_CHECKING, Any, ClassVar, overload

from narwhals._plan import expressions as ir
from narwhals._plan._guards import is_column
from narwhals._plan.common import flatten_hash_safe
from narwhals._plan.expr import Expr
from narwhals._plan.expressions import operators as ops, selectors as s_ir
from narwhals._utils import Version
from narwhals.dtypes import DType

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import timezone

    from typing_extensions import Never, Self

    from narwhals._plan.typing import IntoExpr, IntoExprColumn, OneOrIterable
    from narwhals.typing import TimeUnit

_dtypes = Version.MAIN.dtypes
_dtypes_v1 = Version.V1.dtypes


class Selector(Expr):
    _ir: ir.SelectorIR

    def __repr__(self) -> str:
        return f"nw._plan.Selector({self.version.name.lower()}):\n{self._ir!r}"

    @classmethod
    def _from_ir(cls, selector_ir: ir.SelectorIR, /) -> Self:  # type: ignore[override]
        obj = cls.__new__(cls)
        obj._ir = selector_ir
        return obj

    def as_expr(self) -> Expr:
        if self.version is Version.MAIN:
            return Expr._from_ir(self._ir)
        if self.version is Version.V1:
            from narwhals._plan.expr import ExprV1

            return ExprV1._from_ir(self._ir)
        raise NotImplementedError(self.version)

    # TODO @dangotbanned: Rename to `exclude` (after `Expr.selector` swap)
    def exclude_s(self, *names: OneOrIterable[str]) -> Selector:
        return self - by_name(*names)  # pyright: ignore[reportReturnType]

    @overload  # type: ignore[override]
    def __or__(self, other: Self) -> Self: ...
    @overload
    def __or__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __or__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if isinstance(other, type(self)):
            op = ops.Or()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self.as_expr() | other

    @overload  # type: ignore[override]
    def __and__(self, other: Self) -> Self: ...
    @overload
    def __and__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __and__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if is_column(other) and (name := other.meta.output_name()):
            other = by_name(name)
        if isinstance(other, type(self)):
            op = ops.And()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self.as_expr() & other

    @overload  # type: ignore[override]
    def __sub__(self, other: Self) -> Self: ...
    @overload
    def __sub__(self, other: IntoExpr) -> Expr: ...
    def __sub__(self, other: IntoExpr) -> Self | Expr:
        if isinstance(other, type(self)):
            op = ops.Sub()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self.as_expr() - other

    @overload  # type: ignore[override]
    def __xor__(self, other: Self) -> Self: ...
    @overload
    def __xor__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __xor__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if isinstance(other, type(self)):
            op = ops.ExclusiveOr()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self.as_expr() ^ other

    def __invert__(self) -> Self:
        return self._from_ir(ir.InvertSelector(selector=self._ir))

    def __add__(self, other: Any) -> Expr:  # type: ignore[override]
        if isinstance(other, type(self)):
            msg = "unsupported operand type(s) for op: ('Selector' + 'Selector')"
            raise TypeError(msg)
        return self.as_expr() + other  # type: ignore[no-any-return]

    def __radd__(self, other: Any) -> Never:
        msg = "unsupported operand type(s) for op: ('Expr' + 'Selector')"
        raise TypeError(msg)

    def __rsub__(self, other: Any) -> Never:
        msg = "unsupported operand type(s) for op: ('Expr' - 'Selector')"
        raise TypeError(msg)

    @overload  # type: ignore[override]
    def __rand__(self, other: Self) -> Self: ...
    @overload
    def __rand__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __rand__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if is_column(other) and (name := other.meta.output_name()):
            return by_name(name) & self
        return self.as_expr().__rand__(other)

    @overload  # type: ignore[override]
    def __ror__(self, other: Self) -> Self: ...
    @overload
    def __ror__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __ror__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if is_column(other) and (name := other.meta.output_name()):
            return by_name(name) | self
        return self.as_expr().__ror__(other)

    @overload  # type: ignore[override]
    def __rxor__(self, other: Self) -> Self: ...
    @overload
    def __rxor__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __rxor__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if is_column(other) and (name := other.meta.output_name()):
            return by_name(name) ^ self
        return self.as_expr().__rxor__(other)


class SelectorV1(Selector):
    _version: ClassVar[Version] = Version.V1


def all() -> Selector:
    return s_ir.All().to_selector_ir().to_narwhals()


def array(inner: Selector | None = None, *, size: int | None = None) -> Selector:
    s = inner._ir if inner is not None else None
    return s_ir.Array(inner=s, size=size).to_selector_ir().to_narwhals()


def by_dtype(*dtypes: OneOrIterable[DType | type[DType]]) -> Selector:
    selectors: deque[Selector] = deque()
    dtypes_: deque[DType | type[DType]] = deque()
    for tp in flatten_hash_safe(dtypes):
        if isinstance(tp, type) and issubclass(tp, DType):
            if constructor := _HASH_SENSITIVE_TO_SELECTOR.get(tp):
                selectors.append(constructor())
            else:
                dtypes_.append(tp)
        elif isinstance(tp, DType):
            dtypes_.append(tp)
        else:
            msg = f"invalid dtype: {tp!r}"
            raise TypeError(msg)
    if dtypes_:
        dtype_selector = (
            s_ir.ByDType(dtypes=frozenset(dtypes_)).to_selector_ir().to_narwhals()
        )
        selectors.appendleft(dtype_selector)
    it = iter(selectors)
    if first := next(it, None):
        return reduce(operator.or_, it, first)
    return s_ir.ByDType.empty().to_selector_ir().to_narwhals()


def by_index(*indices: OneOrIterable[int], require_all: bool = True) -> Selector:
    if len(indices) == 1 and isinstance(indices[0], int):
        sel = s_ir.ByIndex.from_index(indices[0], require_all=require_all)
    else:
        sel = s_ir.ByIndex.from_indices(*indices, require_all=require_all)
    return sel.to_selector_ir().to_narwhals()


def by_name(*names: OneOrIterable[str], require_all: bool = True) -> Selector:
    if len(names) == 1 and isinstance(names[0], str):
        sel = s_ir.ByName.from_name(names[0], require_all=require_all)
    else:
        sel = s_ir.ByName.from_names(*names, require_all=require_all)
    return sel.to_selector_ir().to_narwhals()


def boolean() -> Selector:
    return s_ir.Boolean().to_selector_ir().to_narwhals()


def categorical() -> Selector:
    return s_ir.Categorical().to_selector_ir().to_narwhals()


def datetime(
    time_unit: OneOrIterable[TimeUnit] | None = None,
    time_zone: OneOrIterable[str | timezone | None] = ("*", None),
) -> Selector:
    return (
        s_ir.Datetime.from_time_unit_and_time_zone(time_unit, time_zone)
        .to_selector_ir()
        .to_narwhals()
    )


def list(inner: Selector | None = None) -> Selector:
    s = inner._ir if inner is not None else None
    return s_ir.List(inner=s).to_selector_ir().to_narwhals()


def duration(time_unit: OneOrIterable[TimeUnit] | None = None) -> Selector:
    return s_ir.Duration.from_time_unit(time_unit).to_selector_ir().to_narwhals()


def enum() -> Selector:
    return s_ir.Enum().to_selector_ir().to_narwhals()


def matches(pattern: str) -> Selector:
    return s_ir.Matches.from_string(pattern).to_selector_ir().to_narwhals()


def numeric() -> Selector:
    return s_ir.Numeric().to_selector_ir().to_narwhals()


def string() -> Selector:
    return s_ir.String().to_selector_ir().to_narwhals()


def struct() -> Selector:
    return s_ir.Struct().to_selector_ir().to_narwhals()


_HASH_SENSITIVE_TO_SELECTOR: Mapping[type[DType], Callable[[], Selector]] = {
    _dtypes.Datetime: datetime,
    _dtypes_v1.Datetime: datetime,
    _dtypes.Duration: duration,
    _dtypes_v1.Duration: duration,
    _dtypes.Enum: enum,
    _dtypes_v1.Enum: enum,
    _dtypes.Array: array,
    _dtypes.List: list,
    _dtypes.Struct: struct,
}
