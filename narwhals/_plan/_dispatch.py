from __future__ import annotations

import re
from collections.abc import Callable
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, final, overload

from narwhals._plan._guards import is_function_expr
from narwhals._plan.compliant.typing import FrameT_contra, R_co
from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Never, TypeAlias

    from narwhals._plan.compliant.typing import Ctx
    from narwhals._plan.expressions import ExprIR, FunctionExpr
    from narwhals._plan.typing import ExprIRT, FunctionT

__all__ = ["Dispatcher", "get_dispatch_name"]


Node = TypeVar("Node", bound="ExprIR | FunctionExpr[Any]")
Node_contra = TypeVar(
    "Node_contra", bound="ExprIR | FunctionExpr[Any]", contravariant=True
)
Raiser: TypeAlias = Callable[..., "Never"]


class Binder(Protocol[Node_contra]):
    def __call__(
        self, ctx: Ctx[FrameT_contra, R_co], /
    ) -> BoundMethod[Node_contra, FrameT_contra, R_co]: ...


class BoundMethod(Protocol[Node_contra, FrameT_contra, R_co]):
    def __call__(self, node: Node_contra, frame: FrameT_contra, name: str, /) -> R_co: ...


@final
class Dispatcher(Generic[Node]):
    """Translate class definitions into error-wrapped method calls.

    Operates over `ExprIR` and `Function` nodes.
    By default, we dispatch to the compliant-level by calling a method that is the
    **snake_case**-equivalent of the class name:

        class BinaryExpr(ExprIR): ...

        class CompliantExpr(Protocol):
            def binary_expr(self, *args: Any): ...
    """

    __slots__ = ("_bind", "_name")
    _bind: Binder[Node]
    _name: str

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"{type(self).__name__}<{self.name}>"

    def bind(
        self, ctx: Ctx[FrameT_contra, R_co], /
    ) -> BoundMethod[Node, FrameT_contra, R_co]:
        try:
            return self._bind(ctx)
        except AttributeError:
            raise self._not_implemented_error(ctx, "compliant") from None

    def __call__(
        self,
        ctx: Ctx[FrameT_contra, R_co],
        node: Node,
        frame: FrameT_contra,
        name: str,
        /,
    ) -> R_co:
        method = self.bind(ctx)
        if result := method(node, frame, name):
            return result
        raise self._not_implemented_error(ctx, "context")

    @staticmethod
    def from_expr_ir(tp: type[ExprIRT], /) -> Dispatcher[ExprIRT]:
        if not tp.__expr_ir_config__.allow_dispatch:
            return Dispatcher._no_dispatch(tp)
        return Dispatcher._from_type(tp)

    @staticmethod
    def from_function(tp: type[FunctionT], /) -> Dispatcher[FunctionExpr[FunctionT]]:
        return Dispatcher._from_type(tp)

    @staticmethod
    @overload
    def _from_type(tp: type[ExprIRT], /) -> Dispatcher[ExprIRT]: ...
    @staticmethod
    @overload
    def _from_type(tp: type[FunctionT], /) -> Dispatcher[FunctionExpr[FunctionT]]: ...
    @staticmethod
    def _from_type(tp: type[ExprIRT | FunctionT], /) -> Dispatcher[Any]:
        obj = Dispatcher.__new__(Dispatcher)
        obj._name = _method_name(tp)
        getter = attrgetter(obj._name)
        is_namespaced = tp.__expr_ir_config__.is_namespaced
        obj._bind = _via_namespace(getter) if is_namespaced else getter
        return obj

    @staticmethod
    def _no_dispatch(tp: type[ExprIRT], /) -> Dispatcher[ExprIRT]:
        obj = Dispatcher.__new__(Dispatcher)
        obj._name = tp.__name__
        obj._bind = obj._make_no_dispatch_error()
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

    def _not_implemented_error(
        self, ctx: object, /, missing: Literal["compliant", "context"]
    ) -> NotImplementedError:
        if missing == "context":
            msg = f"`{self.name}` is not yet implemented for {type(ctx).__name__!r}"
        else:
            msg = (
                f"`{self.name}` has not been implemented at the compliant-level.\n"
                f"Hint: Try adding `CompliantExpr.{self.name}()` or `CompliantNamespace.{self.name}()`"
            )
        return NotImplementedError(msg)


def _via_namespace(getter: Callable[[Any], Any], /) -> Callable[[Any], Any]:
    def _(ctx: Any, /) -> Any:
        return getter(ctx.__narwhals_namespace__())

    return _


def _pascal_to_snake_case(s: str) -> str:
    """Convert a PascalCase string to snake_case.

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
