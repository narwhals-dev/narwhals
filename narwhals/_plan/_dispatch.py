from __future__ import annotations

import os
import re
from collections.abc import Callable
from operator import attrgetter as _attrgetter, methodcaller as _methodcaller
from typing import TYPE_CHECKING, Any, Final, Generic, Literal, final

from narwhals._plan import common
from narwhals._plan._guards import is_function_expr
from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Never, Self, TypeAlias

    from narwhals._plan.compliant import typing as ct
    from narwhals._plan.expressions import ExprIR, Function, FunctionExpr
    from narwhals._plan.typing import Accessor, Constructs, ExprIRT, FunctionT


__all__ = ("Dispatcher", "DispatcherOptions", "get_dispatch_name")

Incomplete: TypeAlias = Any

Node = TypeVar("Node", bound="ExprIR | FunctionExpr[Any]")
Raiser: TypeAlias = Callable[..., "Never"]


# TODO @dangotbanned: Pick a focus for the docstring and start again
@final
class Dispatcher(Generic[Node]):
    """Dispatch an expression to the compliant-level.

    Translates class definitions into error-wrapped method calls.

    All `ExprIR` and `Function` nodes store one of these guys in `__expr_ir_dispatch__`.

    By default, we dispatch to the compliant-level by calling a method that is the
    **snake_case**-equivalent of the class name:

        class BinaryExpr(ExprIR): ...

        class CompliantExpr(Protocol):
            def binary_expr(self, *args: Any): ...

    ## Rewrite idea
    For a given `Node`, we want to answer:
    1. What is the name of the method we need to call?
    2. Is that name an instance method or do we need to access it from somewhere else?
    3. Once we know where to look, did we actually find the method?
    4. If we found it, did calling it return a value?

    If all goes well - these steps are equivalent to just calling the method directly (if we knew the name at the start).

    If something goes wrong though - we'd like to raise a more helpful error than this:

        AttributeError: "<compliant-something> object has no attribute <method-or-accessor-name>"

    Instead, for a user-facing error we would have:

        NotImplementedError: "`ewm_mean` is not yet implemented for 'ArrowExpr'"
    """

    __slots__ = ("_name", "_options", "bind")

    bind: ct.Binder[Node]
    _options: DispatcherOptions
    _name: str

    @property
    def name(self) -> str:
        """Compliant-level method name.

        They're often the lowercase transform of the class name:

            from narwhals._plan import expressions as ir
            ir.Cast.__expr_ir_dispatch__.name
            'cast'

        *PascalCase* becomes *snake_case*:

            ir.OverOrdered.__expr_ir_dispatch__.name
            'over_ordered'

        Accessor methods reflect the full dotted path:

            ir.lists.NUnique.__expr_ir_dispatch__.name
            'list.n_unique'

        Generated names can always be overridden at class definition time:

            ir.boolean.Not.__expr_ir_dispatch__.name
            'not_'
        """
        return self._name

    # TODO @dangotbanned: Maybe explain in terms of descriptor language?
    # e.g. "when the owner is subclassed ..."
    @property
    def options(self) -> DispatcherOptions:
        """Configuration describing how this instance was built.

        When a dispatchable (...) class is subclassed, this property is inherited
        while each class gets its own instance of `Dispatcher`.

        See `DispatcherOptions` for examples.
        """
        return self._options

    def __repr__(self) -> str:
        return f"{type(self).__name__}<{self.name}>"

    def __call__(
        self,
        node: Node,
        ctx: ct.DispatchScopeAny[ct.Frame, ct.ET_co, ct.ST_co],
        frame: ct.Frame,
        name: str,
        /,
    ) -> ct.ET_co | ct.ST_co:
        """Evaluate this expression in `frame`, using implementation(s) provided by `ctx`."""
        if result := _DISPATCHER_CALL(self, node, ctx, frame, name):
            return result
        msg = f"`{self.name}` is not yet implemented for {type(ctx).__name__!r}"
        raise NotImplementedError(msg)

    @staticmethod
    def from_expr_ir(
        tp: type[ExprIRT], options: DispatcherOptions | None, /
    ) -> Dispatcher[ExprIRT]:
        options = options or tp.__expr_ir_dispatch__.options
        if not options.allow_dispatch:
            return Dispatcher(tp.__name__, options=options)
        return Dispatcher._from_type(tp, options)

    @staticmethod
    def from_function(
        tp: type[FunctionT], options: DispatcherOptions | None, /
    ) -> Dispatcher[FunctionExpr[FunctionT]]:
        """Create a new `Function` dispatcher."""
        options = tp.__expr_ir_dispatch__.options.merge_with(options)
        return Dispatcher._from_type(tp, options)

    @staticmethod
    def _from_type(
        tp: type[Incomplete], options: DispatcherOptions, /
    ) -> Dispatcher[Incomplete]:
        name = options.override_name or _pascal_to_snake_case(tp.__name__)
        if ns := options.accessor_name:
            name = f"{ns}.{name}"
        if constructor := options.constructor_name:
            get_type = _GET_EXPR if constructor == "Expr" else _GET_SCALAR
            bind = _constructor_binder(
                _CALL_NAMESPACE, get_type, _CLASS_METHOD_GETTER(name)
            )
        else:
            bind = _binder(_CALL_EXPR_PREPARE, _ATTR_GETTER(name))
        return Dispatcher(name, bind, options)

    def __get__(self, instance: Any, owner: Any) -> Self:
        return self

    # TODO @dangotbanned: Clean up explanation
    def __set_name__(self, owner: type[Any], name: str) -> None:
        # `Function` and `ExprIR` invoke this by using `Dispatcher()` in the class body
        # This allows special-casing for base class raising an error, but not propagating the behavior (via options)
        self._name = owner.__name__

    def __init__(
        self,
        name: str = "",
        bind: ct.Binder[Node] | None = None,
        options: DispatcherOptions | None = None,
    ) -> None:
        self._name = name
        self.bind = bind or self._make_no_dispatch_error()
        self._options = options or DispatcherOptions()

    def _make_no_dispatch_error(self) -> Callable[[Any], Raiser]:
        def _no_dispatch_error(node: Node, *_: Any) -> Never:
            msg = (
                f"{type(node).__name__!r} should not appear at the compliant-level.\n\n"
                f"Make sure to expand all expressions first, got:\n{node!r}"
            )
            raise TypeError(msg)

        def getter(_: Any, /) -> Raiser:
            return _no_dispatch_error

        return getter


def _dispatch(
    self: Dispatcher[Node],
    node: Node,
    ctx: ct.DispatchScopeAny[ct.Frame, ct.E, ct.SC],
    frame: ct.Frame,
    name: str,
    /,
) -> ct.E | ct.SC | None:
    return self.bind(ctx)(node, frame, name)


def _dispatch_debug(
    self: Dispatcher[Node],
    node: Node,
    ctx: ct.DispatchScopeAny[ct.Frame, ct.E, ct.SC],
    frame: ct.Frame,
    name: str,
    /,
) -> ct.E | ct.SC | None:
    # Provides an opt-in hint for a development-time-only error
    try:
        method = self.bind(ctx)
    except AttributeError as err:
        # This error *looks* like the problem is related to a single backend:
        #   `AttributeError: type object 'ArrowExpr' has no attribute 'lit'`
        name = self.options.constructor_name or "Expr"
        msg = (
            f"`{self.name}` has not been implemented at the compliant-level.\n"
            f"Hint: Try adding `Compliant{name}.{self.name}()`"
        )
        raise NotImplementedError(msg) from err
    return method(node, frame, name)


_DISPATCHER_CALL: Final = (
    _dispatch if not os.environ.get(common.NW_DEV_ENV_NAME) else _dispatch_debug
)


_CALL_NAMESPACE: Final[ct.CallNamespace] = _methodcaller("__narwhals_namespace__")
_CALL_EXPR_PREPARE: Final[ct.CallExprPrepare] = _methodcaller("__narwhals_expr_prepare__")
_ATTR_GETTER: Final[Callable[[str], ct.GetMethod]] = _attrgetter
_CLASS_METHOD_GETTER: Final[Callable[[str], ct.GetClassMethod]] = _attrgetter


def _binder(f1: ct.CallExprPrepare, f2: ct.GetMethod, /) -> ct.Binder[Incomplete]:
    def bind(
        ctx: ct.DispatchScopeAny[ct.Frame, ct.ET_co, ct.ST_co], /
    ) -> ct.BoundMethod[Any, ct.Frame, ct.ET_co | ct.ST_co]:
        return f2(f1(ctx))

    return bind


_GET_EXPR: Final[ct.GetExpr] = _attrgetter("_expr")
_GET_SCALAR: Final[ct.GetScalar] = _attrgetter("_scalar")


def _constructor_binder(
    f1: ct.CallNamespace, f2: ct.GetExpr | ct.GetScalar, f3: ct.GetClassMethod, /
) -> ct.Binder[Incomplete]:
    def bind(
        ctx: ct.DispatchScopeAny[ct.Frame, ct.ET_co, ct.ST_co], /
    ) -> ct.BoundMethod[Any, ct.Frame, ct.ET_co | ct.ST_co]:
        return f3(f2(f1(ctx)))

    return bind


@final
class DispatcherOptions:
    """Class-level configuration for how a `Dispatcher` should be built.

    Defined via the (optional) `dispatch` parameter at [subclass-definition time].

    Many expressions simply use the default:
    >>> from narwhals._plan import expressions as ir
    >>> from narwhals._plan._dispatch import DispatcherOptions
    >>> from narwhals._plan._nodes import node
    >>> from narwhals._plan.options import ExplodeOptions
    >>>
    >>> class Explode(ir.ExprIR):
    ...     #                  ^ # default `dispatch`
    ...     __slots__ = ("expr",)
    ...     expr: ir.ExprIR = node()

    >>> Explode.__expr_ir_dispatch__
    Dispatcher<explode>

    >>> Explode.__expr_ir_dispatch__.options
    DispatcherOptions(<default>)

    `dispatch` provides a bit more flexibility when you want it:

    >>> class Explode2(Explode, dispatch=DispatcherOptions.renamed("explodier")): ...
    >>> #                       ^^^^^^^^ custom `dispatch`

    >>> Explode2.__expr_ir_dispatch__
    Dispatcher<explodier>

    >>> Explode2.__expr_ir_dispatch__.options
    DispatcherOptions(override_name='explodier')

    Keep in mind that `options` are inherited:
    >>> class Explode21(Explode2): ...
    >>> Explode21.__expr_ir_dispatch__
    Dispatcher<explodier>

    So we'd need another override to get the default back:
    >>> class ExplodeWithOptions(Explode2, dispatch=DispatcherOptions()):
    ...     __slots__ = ("options",)
    ...     options: ExplodeOptions

    >>> ExplodeWithOptions.__expr_ir_dispatch__
    Dispatcher<explode_with_options>

    [subclass-definition time]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
    """

    __slots__ = ("accessor_name", "allow_dispatch", "constructor_name", "override_name")
    accessor_name: Accessor | None
    """Name of an (optional) expression namespace accessor.

    Required for expressions like:

        nw.col("a").str.len_chars()
        #           ^^^
    """

    allow_dispatch: bool
    """Whether the expression is supported at the compliant-level."""

    constructor_name: Constructs | None
    override_name: str
    """Manual override to the auto-generated expression dispatch method name.

    By default the name is derived from the [snake_case] transform of the [PascalCase] class name:

        >>> from narwhals._plan import expressions as ir
        >>> def show_dispatch_names(*tps: type[ir.ExprIR | ir.Function]) -> None:
        ...     length = max(len(tp.__name__) for tp in tps)
        ...     for tp in tps:
        ...         print(f"{tp.__name__:<{length}} -> {tp.__expr_ir_dispatch__.name}")
        >>>
        >>>
        >>> show_dispatch_names(ir.Cast, ir.BinaryExpr, ir.strings.StartsWith)
        Cast       -> cast
        BinaryExpr -> binary_expr
        StartsWith -> str.starts_with

    `override_name` provides an escape hatch for edge cases:

        >>> show_dispatch_names(ir.boolean.Not)
        Not    -> not_

    [snake_case]: https://en.wikipedia.org/wiki/Snake_case
    [PascalCase]: https://en.wikipedia.org/wiki/Naming_convention_(programming)#Examples_of_multiple-word_identifier_formats
    """

    def __init__(
        self,
        *,
        accessor_name: Accessor | None = None,
        allow_dispatch: bool = True,
        constructor_name: Constructs | None = None,
        override_name: str = "",
    ) -> None:
        self.accessor_name = accessor_name
        self.allow_dispatch = allow_dispatch
        self.constructor_name = constructor_name
        self.override_name = override_name

    @staticmethod
    def renamed(override_name: str, /) -> DispatcherOptions:
        """Override the auto-generated expression dispatch method name.

        Syntax sugar for `DispatcherOptions(override_name=override_name)`.
        """
        # NOTE: With `LenStar` & `Col` renamed - the remaining usage is:
        # - `boolean.Not` -> `not_()`
        # - `strings.ZFill` -> (str.) `zfill`
        # - `struct.FieldByName` -> (struct.) `field`
        return DispatcherOptions(override_name=override_name)

    def __repr__(self) -> str:
        parts = []
        if accessor := self.accessor_name:
            parts.append(f"accessor_name={accessor!r}")
        if not self.allow_dispatch:
            parts.append(f"allow_dispatch={self.allow_dispatch}")
        if constructor := self.constructor_name:
            parts.append(f"constructor_name={constructor!r}")
        if override := self.override_name:
            parts.append(f"override_name={override!r}")
        inner = (", ".join(parts)) if parts else "<default>"
        return f"{type(self).__name__}({inner})"

    def merge_with(self, child: DispatcherOptions | None) -> DispatcherOptions:
        """Propagate options to a subclassed `Function`.

        When `child` is None, it inherits all the options of self.

        If an `accessor_name` was defined, it is **sticky** and will be merged
        with options of `child`:

        >>> from narwhals._plan._function import Function
        >>> options = DispatcherOptions
        >>> class StringFunction(Function, dispatch=options(accessor_name="str")): ...
        >>> class ZFill(StringFunction, dispatch=options.renamed("zfill")): ...
        >>> get_dispatch_name(ZFill)
        'str.zfill'
        """
        options = child or self
        if accessor_name := (self.accessor_name or options.accessor_name):
            options = DispatcherOptions(
                override_name=options.override_name, accessor_name=accessor_name
            )
        return options


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


def get_dispatch_name(expr: ExprIR | type[Function], /) -> str:
    """Return the synthesized method name for `expr`.

    Note:
        Refers to the `Compliant*` method name, which may be *either* more general
        or more specialized than what the user called.
    """
    return (expr.function if is_function_expr(expr) else expr).__expr_ir_dispatch__.name
