"""Field descriptor for customizing `ExprIR`.

- https://github.com/narwhals-dev/narwhals/pull/3066#issuecomment-3242037939
- https://typing.python.org/en/latest/spec/dataclasses.html#field-specifiers
- https://github.com/python/typing/blob/f0decdcb56d0dc9bd2d928cabda7e4462d098b1f/conformance/tests/dataclasses_transform_field.py
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from narwhals._plan._immutable import dataclass_transform  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from typing_extensions import Never

    from narwhals._plan.common import ExprIR
    from narwhals._plan.typing import Seq


class _Node:
    # TODO @dangotbanned: Utilize the spec to build an iteration pattern
    def __init__(self, *, nested: bool, include_root_names: bool) -> None:
        self._nested = nested
        self._include_root_names = include_root_names

    def __set_name__(self, owner: type[Any], name: str) -> None:
        # https://docs.python.org/3/howto/descriptor.html#customized-names
        self._name_owner: str = owner.__name__
        self._name: str = name
        self._name_instance: str = f"_{name}"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}: {self._name_owner}.{self._name}\n"
            f"  nested={self._nested}\n  include_root_names={self._include_root_names!r}"
        )

    def __get__(self, instance: object | None, owner: type[Any] | None) -> Any:
        return self if instance is None else getattr(instance, self._name_instance)

    def __set__(self, instance: object, value: Any) -> None:
        object.__setattr__(instance, self._name_instance, value)


# - `iter_output_name`
#   - Call on first child
#     - AggExpr, BinaryExpr, Cast, Sort, SortBy, Filter, WindowExpr, TernaryExpr
#   - Custom
#     - FunctionExpr (call on first element of first (and only))
#       - Technically the same as the others, just happens to be the only one with a tuple in first slot
#   - iter_right (self, then reverse everything)
#     - Everything else
#     - Regardless of whether or not they have children
#       - Has children: KeepName, RenameAlias, Alias, Exclude (is also _ColumnSelection)
#       - No children: SelectorIR, _ColumnSelection, Column, Len, Literal
def node(*, nested: bool = False, include_root_names: bool = True) -> Any:
    """Config should be evaluated once per class, each instance then doesn't need to branch."""
    return _Node(nested=nested, include_root_names=include_root_names)


@dataclass_transform(kw_only_default=True, frozen_default=True, field_specifiers=(node,))
class Base:
    """Fully experimenting.

    Trimmed version of `Immutable`, focusing on new stuff for `ExprIR`.

    The idea is the `cls` namespace is used for specifying how each field should be traversed.

    Doing things this way as `__annotations__` is brittle and slow.
    """

    __slots__ = ()
    __nw_nodes__: ClassVar[MappingProxyType[str, _Node]]

    def __setattr__(self, name: str, value: Never) -> Never:
        msg = f"{type(self).__name__!r} is immutable, {name!r} cannot be set."
        raise AttributeError(msg)

    def __init__(self, **kwds: Any) -> None:
        for name, value in kwds.items():
            object.__setattr__(self, name, value)

    def __init_subclass__(cls, *args: Any, **kwds: Any) -> None:
        super().__init_subclass__(*args, **kwds)
        m = {
            name: value
            for name, value in cls.__dict__.items()
            if isinstance(value, _Node)
        }
        if m and (existing := getattr(cls, "__nw_nodes__", None)):
            m = dict(existing, **m)
        if m:
            cls.__nw_nodes__ = MappingProxyType(m)


class Child1(Base):
    __slots__ = ("_expr", "_order_by", "non_node")
    expr: ExprIR = node()
    order_by: Seq[ExprIR] = node(nested=True)
    non_node: str


class Child2(Child1): ...


class Child21(Child2):
    __slots__ = ("_by", "_expr", "_order_by", "non_node")
    by: Seq[ExprIR] = node(nested=True)


class Child22(Child2):
    __slots__ = (*Child2.__slots__, "_bye")
    bye: ExprIR = node()
