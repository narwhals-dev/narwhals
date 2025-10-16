from __future__ import annotations

import re
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Callable, Generic, final, overload

from narwhals._plan._guards import is_function_expr
from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Never, TypeAlias

    from narwhals._plan.expressions import ExprIR, FunctionExpr
    from narwhals._plan.typing import ExprIRT, FunctionT

__all__ = ["Dispatcher", "get_dispatch_name"]

Incomplete: TypeAlias = "Any"

Node = TypeVar("Node")


Getter: TypeAlias = Callable[[Any], Any]
Raiser: TypeAlias = Callable[..., "Never"]


@final
class Dispatcher(Generic[Node]):
    __slots__ = ("_method_getter", "_name")
    _method_getter: Getter
    _name: str

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"{type(self).__name__}<{self.name}>"

    def __call__(
        self, ctx: Incomplete, node: Node, frame: Any, name: str, /
    ) -> Incomplete:
        # raises when the method isn't implemented on `CompliantExpr`, but exists as a method on `Expr`
        # gives a more helpful error for things that are namespaced like `col("a").str.replace`
        try:
            bound_method = self._method_getter(ctx)
        except AttributeError:
            raise self._not_implemented_error(ctx) from None

        if result := bound_method(node, frame, name):
            return result
        # here if is defined on `CompliantExpr`, but not on ctx
        raise self._not_implemented_error(ctx)

    @staticmethod
    def from_expr_ir(tp: type[ExprIRT], /) -> Dispatcher[ExprIRT]:
        if not tp.__expr_ir_config__.allow_dispatch:
            return Dispatcher._no_dispatch(tp)
        return Dispatcher._from_configured_type(tp)

    @staticmethod
    def from_function(tp: type[FunctionT], /) -> Dispatcher[FunctionExpr[FunctionT]]:
        return Dispatcher._from_configured_type(tp)

    @staticmethod
    @overload
    def _from_configured_type(tp: type[ExprIRT], /) -> Dispatcher[ExprIRT]: ...

    @staticmethod
    @overload
    def _from_configured_type(
        tp: type[FunctionT], /
    ) -> Dispatcher[FunctionExpr[FunctionT]]: ...

    # TODO @dangotbanned: Can this be done without overloads?
    @staticmethod
    def _from_configured_type(
        tp: type[ExprIRT | FunctionT], /
    ) -> Dispatcher[ExprIRT] | Dispatcher[FunctionExpr[FunctionT]]:
        obj = Dispatcher.__new__(Dispatcher)
        obj._name = _method_name(tp)
        getter = attrgetter(obj._name)
        is_namespaced = tp.__expr_ir_config__.is_namespaced
        obj._method_getter = _via_namespace(getter) if is_namespaced else getter
        return obj

    @staticmethod
    def _no_dispatch(tp: type[ExprIRT], /) -> Dispatcher[ExprIRT]:
        obj = Dispatcher.__new__(Dispatcher)
        obj._name = tp.__name__
        obj._method_getter = obj._make_no_dispatch_error()
        return obj

    def _make_no_dispatch_error(self) -> Callable[[Any], Raiser]:
        def _no_dispatch_error(node: Node, *_: Any) -> Never:
            msg = (
                f"{self.name!r} should not appear at the compliant-level.\n\n"
                f"Make sure to expand all expressions first, got:\n{node!r}"
            )
            raise TypeError(msg)

        def getter(_: Any, /) -> Raiser:
            return _no_dispatch_error

        return getter

    def _not_implemented_error(self, ctx: object, /) -> NotImplementedError:
        msg = f"`{self.name}` is not yet implemented for {type(ctx).__name__!r}"
        return NotImplementedError(msg)


def _via_namespace(getter: Getter, /) -> Getter:
    def _(ctx: Any, /) -> Any:
        return getter(ctx.__narwhals_namespace__())

    return _


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


def _method_name(tp: type[ExprIRT | FunctionT]) -> str:
    config = tp.__expr_ir_config__
    name = config.override_name or _pascal_to_snake_case(tp.__name__)
    return f"{ns}.{name}" if (ns := getattr(config, "accessor_name", "")) else name


def get_dispatch_name(expr: ExprIR, /) -> str:
    """Return the synthesized method name for `expr`."""
    return (
        repr(expr.function) if is_function_expr(expr) else expr.__expr_ir_dispatch__.name
    )
