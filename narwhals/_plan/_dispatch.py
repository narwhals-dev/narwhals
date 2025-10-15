from __future__ import annotations

import re
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.expressions import FunctionExpr
    from narwhals._plan.typing import ExprIRT, FunctionT

Incomplete: TypeAlias = "Any"


class DispatchGetter:
    __slots__ = ("_fn", "_name")
    _fn: Callable[[Any], Any]
    _name: str

    def __call__(self, ctx: Any, /) -> Any:
        result = self._fn(ctx)
        # this can be `None` iff the method isn't implemented on `CompliantExpr`, but exists as a method on `Expr`
        if result is not None:
            # but the issue I have is `result` exists, but returns `None` when called like `result(node, frame, name)`
            return result
        raise self._not_implemented_error(ctx)

    @classmethod
    def no_dispatch(cls, tp: type[ExprIRT]) -> Self:
        tp_name = tp.__name__
        obj = cls.__new__(cls)
        obj._name = tp_name

        # NOTE: Temp weirdness until fixing the original issue with signature that this had (but never got triggered)
        def _(ctx: Any, /, node: ExprIRT, _: Any, name: str) -> Any:
            raise obj._no_dispatch_error(ctx, node, name)

        obj._fn = lambda _ctx: _
        return obj

    @classmethod
    def from_expr_ir(cls, tp: type[ExprIRT]) -> Self:
        if not tp.__expr_ir_config__.allow_dispatch:
            return cls.no_dispatch(tp)
        return cls._from_configured_type(tp)

    @classmethod
    def from_function(cls, tp: type[FunctionT]) -> Self:
        return cls._from_configured_type(tp)

    @classmethod
    def _from_configured_type(cls, tp: type[ExprIRT | FunctionT]) -> Self:
        name = dispatch_method_name(tp)
        getter = attrgetter(name)
        origin = tp.__expr_ir_config__.origin
        fn = getter if origin == "expr" else _dispatch_via_namespace(getter)
        obj = cls.__new__(cls)
        obj._fn = fn
        obj._name = name
        return obj

    def _not_implemented_error(self, ctx: object, /) -> NotImplementedError:
        msg = f"`{self._name}` is not yet implemented for {type(ctx).__name__!r}"
        return NotImplementedError(msg)

    def _no_dispatch_error(self, ctx: Any, node: ExprIRT, name: str, /) -> TypeError:
        msg = (
            f"{self._name!r} should not appear at the compliant-level.\n\n"
            f"Make sure to expand all expressions first, got:\n{ctx!r}\n{node!r}\n{name!r}"
        )
        return TypeError(msg)


def _dispatch_via_namespace(getter: Callable[[Any], Any], /) -> Callable[[Any], Any]:
    def _(ctx: Any, /) -> Any:
        return getter(ctx.__narwhals_namespace__())

    return _


def dispatch_generate(
    tp: type[ExprIRT], /
) -> Callable[[Incomplete, ExprIRT, Incomplete, str], Incomplete]:
    getter = DispatchGetter.from_expr_ir(tp)

    def _(ctx: Any, /, node: ExprIRT, frame: Any, name: str) -> Any:
        return getter(ctx)(node, frame, name)

    return _


def dispatch_generate_function(
    tp: type[FunctionT], /
) -> Callable[[Incomplete, FunctionExpr[FunctionT], Incomplete, str], Incomplete]:
    getter = DispatchGetter.from_function(tp)

    def _(ctx: Any, /, node: FunctionExpr[FunctionT], frame: Any, name: str) -> Any:
        return getter(ctx)(node, frame, name)

    return _


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


def dispatch_method_name(tp: type[ExprIRT | FunctionT]) -> str:
    config = tp.__expr_ir_config__
    name = config.override_name or pascal_to_snake_case(tp.__name__)
    return f"{ns}.{name}" if (ns := getattr(config, "accessor_name", "")) else name
