from __future__ import annotations

import re
from collections.abc import Callable
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, final

from narwhals._plan._guards import is_function_expr
from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Never, Self, TypeAlias

    from narwhals._plan.compliant.typing import Ctx, FrameT_contra, R_co
    from narwhals._plan.expressions import ExprIR, Function, FunctionExpr
    from narwhals._plan.typing import Accessor, ExprIRT, FunctionT

    Node_contra = TypeVar(
        "Node_contra", bound="ExprIR | FunctionExpr[Any]", contravariant=True
    )

    class Binder(Protocol[Node_contra]):
        def __call__(
            self, ctx: Ctx[FrameT_contra, R_co], /
        ) -> BoundMethod[Node_contra, FrameT_contra, R_co]: ...

    class BoundMethod(Protocol[Node_contra, FrameT_contra, R_co]):
        def __call__(
            self, node: Node_contra, frame: FrameT_contra, name: str, /
        ) -> R_co: ...


__all__ = ["Dispatcher", "DispatcherOptions", "get_dispatch_name"]

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

        AttributeError: "<compliant-somthing> object has no attribute <method-or-namespace-or-accessor-name>"

    Instead, for a user-facing error we would have:

        NotImplementedError: "`ewm_mean` is not yet implemented for 'ArrowExpr'"

    And a developer-facing error might be:

        NotImplementedError: "`all_horizontal` has not been implemented at the compliant-level."
        "Hint: Try adding `CompliantNamespace.all_horizontal()`"
    """

    __slots__ = ("_name", "_options", "bind")

    # TODO @dangotbanned: Improve or remnove doc
    bind: Binder[Node]
    """Retrieve the implementation of this expression from `ctx`.

    Binds an instance method, most commonly via:

        expr: CompliantExpr
        method = getattr(expr, "method_name")
    """
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

        Namespaced-methods reflect the accessor in their name:

            ir.lists.NUnique.__expr_ir_dispatch__.name
            'list.n_unique'

        Generated names can always be overriden at class definition time:

            ir.Column.__expr_ir_dispatch__.name
            'col'
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
        ctx: Ctx[FrameT_contra, R_co],
        frame: FrameT_contra,
        name: str,
        /,
    ) -> R_co:
        """Evaluate this expression in `frame`, using implementation(s) provided by `ctx`."""
        try:
            method = self.bind(ctx)
        except AttributeError:
            raise self._not_implemented_error(ctx, "compliant") from None
        if result := method(node, frame, name):
            return result
        raise self._not_implemented_error(ctx, "context")

    @staticmethod
    def from_expr_ir(
        tp: type[ExprIRT], options: DispatcherOptions | Literal["no_dispatch"] | None, /
    ) -> Dispatcher[ExprIRT]:
        if options == "no_dispatch":
            options = DispatcherOptions(allow_dispatch=False)
        options = options or tp.__expr_ir_dispatch__.options
        if not options.allow_dispatch:
            return Dispatcher(tp.__name__, options=options)
        return Dispatcher._from_type(tp, options)

    @staticmethod
    def from_function(
        tp: type[FunctionT], options: DispatcherOptions | None, /
    ) -> Dispatcher[FunctionExpr[FunctionT]]:
        """Create a new `Function` dispatcher.

        When `options` is None, the options of the parent are inherited.

        If a parent defines an `accessor_name`, it is **sticky** and will be merged
        with options of a child:

        >>> from narwhals._plan._function import Function
        >>> options = DispatcherOptions
        >>> class StringFunction(Function, dispatch=options(accessor_name="str")): ...
        >>> class ZFill(StringFunction, dispatch=options.renamed("zfill")): ...
        >>> get_dispatch_name(ZFill)
        'str.zfill'
        """
        options_parent = tp.__expr_ir_dispatch__.options
        options = options or options_parent
        if accessor_name := (options_parent.accessor_name or options.accessor_name):
            options = DispatcherOptions(
                is_namespaced=options.is_namespaced,
                override_name=options.override_name,
                accessor_name=accessor_name,
            )
        return Dispatcher._from_type(tp, options)

    @staticmethod
    def _from_type(
        tp: type[Incomplete], options: DispatcherOptions, /
    ) -> Dispatcher[Incomplete]:
        name = options.override_name or _pascal_to_snake_case(tp.__name__)
        if ns := options.accessor_name:
            name = f"{ns}.{name}"
        getter = attrgetter(name)
        bind = _via_namespace(getter) if options.is_namespaced else getter
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
        bind: Binder[Node] | None = None,
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

    def _not_implemented_error(
        self, ctx: Ctx[Any, Any], /, missing: Literal["compliant", "context"]
    ) -> NotImplementedError:
        is_namespaced = self.options.is_namespaced
        if missing == "context":
            owner = ctx.__narwhals_namespace__() if is_namespaced else ctx
            msg = f"`{self.name}` is not yet implemented for {type(owner).__name__!r}"
        else:
            base_name = "Namespace" if is_namespaced else "Expr"
            msg = (
                f"`{self.name}` has not been implemented at the compliant-level.\n"
                f"Hint: Try adding `Compliant{base_name}.{self.name}()`"
            )
        return NotImplementedError(msg)


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

    __slots__ = ("accessor_name", "allow_dispatch", "is_namespaced", "override_name")
    accessor_name: Accessor | None
    """Name of an (optional) expression namespace accessor.

    Required for expressions like:

        nw.col("a").str.len_chars()
        #           ^^^
    """

    allow_dispatch: bool
    """Whether the expression is supported at the compliant-level.

    When `False`, **any** attempts to dispatch will raise a `TypeError`:
    >>> import narwhals._plan as nw
    >>> selector = nw.col("a", "b", "c")._ir
    >>> selector.dispatch(..., ..., ...)
    Traceback (most recent call last):
    TypeError: 'RootSelector' should not appear at the compliant-level.
    <BLANKLINE>
    Make sure to expand all expressions first, got:
    ncs.by_name('a', 'b', 'c', require_all=True)
    """

    is_namespaced: bool
    """True if expression dispatch routes through `__narwhals_namespace__`.

    Required for expressions like:

        nw.all_horizontal(...)
        # ^

    But not for methods *on* `Expr` like:

        nw.col("a").max()
        #          ^
    """

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

        >>> show_dispatch_names(ir.Column, ir.Literal, ir.boolean.Not)
        Column  -> col
        Literal -> lit
        Not     -> not_

    [snake_case]: https://en.wikipedia.org/wiki/Snake_case
    [PascalCase]: https://en.wikipedia.org/wiki/Naming_convention_(programming)#Examples_of_multiple-word_identifier_formats
    """

    def __init__(
        self,
        *,
        accessor_name: Accessor | None = None,
        allow_dispatch: bool = True,
        is_namespaced: bool = False,
        override_name: str = "",
    ) -> None:
        self.accessor_name = accessor_name
        self.allow_dispatch = allow_dispatch
        self.is_namespaced = is_namespaced
        self.override_name = override_name

    @staticmethod
    def namespaced(override_name: str = "", /) -> DispatcherOptions:
        """Route expression dispatch through `__narwhals_namespace__`.

        Syntax sugar for `DispatcherOptions(is_namespaced=True, override_name=override_name)`.
        """
        return DispatcherOptions(is_namespaced=True, override_name=override_name)

    @staticmethod
    def renamed(override_name: str, /) -> DispatcherOptions:
        """Override the auto-generated expression dispatch method name.

        Syntax sugar for `DispatcherOptions(override_name=override_name)`.
        """
        return DispatcherOptions(override_name=override_name)

    def __repr__(self) -> str:
        parts = []
        if accessor := self.accessor_name:
            parts.append(f"accessor_name={accessor!r}")
        if not self.allow_dispatch:
            parts.append(f"allow_dispatch={self.allow_dispatch}")
        if namespaced := self.is_namespaced:
            parts.append(f"is_namespaced={namespaced}")
        if override := self.override_name:
            parts.append(f"override_name={override!r}")
        inner = (", ".join(parts)) if parts else "<default>"
        return f"{type(self).__name__}({inner})"


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


def get_dispatch_name(expr: ExprIR | type[Function], /) -> str:
    """Return the synthesized method name for `expr`.

    Note:
        Refers to the `Compliant*` method name, which may be *either* more general
        or more specialized than what the user called.
    """
    dispatch: Dispatcher[Any]
    if is_function_expr(expr):
        from narwhals._plan import expressions as ir

        if isinstance(expr, (ir.RollingExpr, ir.AnonymousExpr)):
            dispatch = expr.__expr_ir_dispatch__
        else:
            dispatch = expr.function.__expr_ir_dispatch__
    else:
        dispatch = expr.__expr_ir_dispatch__
    return dispatch.name
