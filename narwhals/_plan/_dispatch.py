"""Making the connection between `CompliantColumn` and `ExprIR`.

## Implementation Notes
An earlier version implemented `Dispatch.__call__` using the descriptor protocol,
to make the signature more convenient for the caller:

    # Current
    node.__expr_ir_dispatch__(node, ctx, frame, name)
    # Descriptor
    node.__expr_ir_dispatch__(ctx, frame, name)

However, it required creating an additional function per-call - which was
considered a bad tradeoff (performance & complexity) for syntax sugar.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Final, Generic, Protocol, TypeVar, final

from narwhals._plan import common
from narwhals._plan._guards import is_function_expr
from narwhals._plan.common import pascal_to_snake_case
from narwhals._utils import deep_attrgetter

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Never, Self, TypeAlias

    from narwhals._plan import _expr_ir, expressions as ir
    from narwhals._plan.compliant import typing as ct
    from narwhals._plan.typing import Accessor, Constructs


__all__ = (
    "ConstructorDispatch",
    "Dispatch",
    "DispatcherOptions",
    "ExprIRDispatch",
    "FunctionDispatch",
    "FunctionExprDispatch",
    "get_dispatch_name",
)


Expr = TypeVar("Expr", bound="ir.ExprIR")
Expr_contra = TypeVar("Expr_contra", bound="ir.ExprIR", contravariant=True)
FExpr = TypeVar("FExpr", bound="ir.FunctionExpr[Any]")
ConExpr = TypeVar("ConExpr", bound="_expr_ir.Constructor")

_FromExpr = TypeVar("_FromExpr", bound="ir.ExprIR")
_FromConExpr = TypeVar("_FromConExpr", bound="_expr_ir.Constructor")
_FromFunction = TypeVar("_FromFunction", bound="ir.Function")


class Bind(Protocol[Expr_contra]):
    def __call__(
        self, ctx: ct.Caller[ct.E, ct.SC], /
    ) -> Bound[Expr_contra, ct.E, ct.SC]: ...


Bound: TypeAlias = "Callable[[Expr, ct.FrameAny, str], ct.E | ct.SC]"


# TODO @dangotbanned: Pick a focus for the docstring and start again
class Dispatch(Generic[Expr]):
    """Dispatch an expression to the compliant-level.

    Translates `ExprIR` and `Function` defintions into error-wrapped method calls, and stores in ``__expr_ir_dispatch__`.

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

    __slots__ = ("_bind", "_name")
    _name: str
    _bind: Bind[Expr]

    @property
    def name(self) -> str:
        """Compliant-level method name.

        They're often the lowercase transform of the class name:
        >>> import narwhals as nw
        >>> import narwhals._plan as nwp
        >>> from narwhals._plan import expressions as ir
        >>> expr = nwp.col("a").cast(nw.Int64())._ir
        >>> type(expr).__name__, get_dispatch_name(expr)
        ('Cast', 'cast')

        *PascalCase* becomes *snake_case*:
        >>> expr = nwp.col("a").over(order_by="b")._ir
        >>> type(expr).__name__, get_dispatch_name(expr)
        ('OverOrdered', 'over_ordered')

        Accessor methods reflect the full dotted path:
        >>> ir.lists.NUnique.__expr_ir_dispatch__.name
        'list.n_unique'

        For functions, generated names can be overridden at class definition time:
        >>> ir.boolean.Not.__expr_ir_dispatch__.name
        'not_'
        """
        return self._name

    @property
    def name_full(self) -> str:
        return f"CompliantExpr.{self.name}"

    def __repr__(self) -> str:
        return f"Dispatch<{self.name}>"

    def __call__(
        self, node: Expr, ctx: ct.Caller[ct.E, ct.SC], frame: ct.FrameAny, name: str, /
    ) -> ct.E | ct.SC:
        """Evaluate this expression in `frame`, using implementation(s) provided by `ctx`."""
        # NOTE: A `None` check is required for two related reasons.
        # 1. An un-implemented `Protocol` method body `...` returns `None`
        #    if you inherit but don't override it.
        #    So it won't raise an `AttributeError` *here*, but would as soon as you try to do anything with the result.
        # 2. Truth-testing `if (result := <call>): ...` can give a false-negative in the eager case,
        #    where `__len__() >= 0` produces `False`.
        #    https://docs.python.org/3/reference/datamodel.html#object.__len__
        if (result := _DISPATCHER_CALL(self, node, ctx, frame, name)) is not None:
            return result
        msg = f"`{self.name}` is not yet implemented for {type(ctx).__name__!r}"
        raise NotImplementedError(msg)

    def bind(self, ctx: ct.Caller[ct.E, ct.SC], /) -> Bound[Expr, ct.E, ct.SC]:
        return self._bind(ctx)

    @classmethod
    def root(cls, name: str, /) -> Self:
        self = cls.__new__(cls)
        self._name = name
        self._bind = _make_no_dispatch_error()
        return self

    @classmethod
    def from_type(cls: type[Dispatch[Any]], tp: type[Any], /) -> Dispatch[Any]:
        raise NotImplementedError


class _CompliantNew(Dispatch[Expr]):
    """Create a new compliant object (of the same kind) for each call."""

    def bind(self, ctx: ct.Caller[ct.E, ct.SC], /) -> Bound[Expr, ct.E, ct.SC]:
        tp = type(ctx)
        return super().bind(tp.__new__(tp))


@final
class FunctionDispatch(_CompliantNew[FExpr]):
    __slots__ = ("_options",)
    _options: DispatcherOptions

    @property
    def options(self) -> DispatcherOptions:
        """Configuration describing how this instance was built.

        When a `Function` is subclassed, this property is inherited and reused in a new
        `FunctionDispatch` instance.

        See `DispatcherOptions` for examples.
        """
        return self._options

    @classmethod
    def root(cls: type[FunctionDispatch[Any]], name: str, /) -> FunctionDispatch[Any]:
        self = super().root(name)
        self._options = DispatcherOptions()
        return self

    @classmethod
    def from_type(
        cls: type[FunctionDispatch[Any]],
        tp: type[_FromFunction],
        options: DispatcherOptions | None = None,
        /,
    ) -> FunctionDispatch[ir.FunctionExpr[_FromFunction]]:
        options = tp.__expr_ir_dispatch__.options.merge_with(options)
        self = cls.__new__(cls)
        name = options.override_name or pascal_to_snake_case(tp.__name__)
        if ns := options.accessor_name:
            name = f"{ns}.{name}"
        self._name = name
        self._bind = deep_attrgetter(self.name)
        self._options = options
        return self


_NARWHALS_CLASSES: Final = "__narwhals_classes__"


@final
class ConstructorDispatch(Dispatch[ConExpr]):
    __slots__ = ("_constructs",)
    _constructs: Constructs

    @property
    def name_full(self) -> str:
        return f"Compliant{self._constructs}.{self.name}"

    @classmethod
    def from_type(
        cls: type[ConstructorDispatch[Any]],
        tp: type[_FromConExpr],
        constructs: Constructs = "Expr",
        /,
    ) -> ConstructorDispatch[_FromConExpr]:
        self = cls.__new__(cls)
        self._name = pascal_to_snake_case(tp.__name__)
        self._bind = deep_attrgetter(_NARWHALS_CLASSES, constructs.lower(), self.name)
        self._constructs = constructs
        return self


class ExprIRDispatch(_CompliantNew[Expr]):
    """`(ExprIR - FunctionExpr)`."""

    @classmethod
    def from_type(
        cls: type[ExprIRDispatch[Any]], tp: type[_FromExpr], /
    ) -> ExprIRDispatch[_FromExpr]:
        self = cls.__new__(cls)
        self._name = pascal_to_snake_case(tp.__name__)
        self._bind = deep_attrgetter(self.name)
        return self


@final
class FunctionExprDispatch(ExprIRDispatch[FExpr]):
    __slots__ = ()

    @classmethod
    def root(
        cls: type[FunctionExprDispatch[Any]], name: str, /
    ) -> FunctionExprDispatch[Any]:
        self = cls.__new__(cls)
        self._name = pascal_to_snake_case(name)
        self._bind = deep_attrgetter(self.name)
        return self

    def __call__(
        self,
        node: FExpr | ir.FunctionExpr,
        ctx: ct.Caller[ct.E, ct.SC],
        frame: ct.FrameAny,
        name: str,
        /,
    ) -> ct.E | ct.SC:
        # NOTE: `node: FExpr | ir.FunctionExpr` is a trick to get the default of `FunctionT_co`.
        # This is a more useful type than `Any`:
        # `_: Any | FunctionDispatch[FunctionExpr[Function]] = node.function.__expr_ir_dispatch__`
        return node.function.__expr_ir_dispatch__(node, ctx, frame, name)


@final
class NoDispatch(Dispatch[Expr]):
    """`Alias`, `KeepName`, `RenameAlias`, `SelectorIR`."""

    __slots__ = ()

    @classmethod
    def from_type(
        cls: type[NoDispatch[Any]], tp: type[_FromExpr], /
    ) -> NoDispatch[_FromExpr]:
        self: NoDispatch[_FromExpr] = cls.__new__(cls)
        self._name = tp.__name__
        self._bind = _make_no_dispatch_error()
        return self


def _make_no_dispatch_error() -> Callable[[Any], Callable[..., Never]]:
    def _no_dispatch_error(node: ir.ExprIR, *_: Any) -> Never:
        msg = (
            f"{type(node).__name__!r} should not appear at the compliant-level.\n\n"
            f"Make sure to expand all expressions first, got:\n{node!r}"
        )
        raise TypeError(msg)

    def getter(_: Any, /) -> Callable[..., Never]:
        return _no_dispatch_error

    return getter


def _dispatch(
    self: Dispatch[Expr],
    node: Expr,
    ctx: ct.Caller[ct.E, ct.SC],
    frame: ct.FrameAny,
    name: str,
    /,
) -> ct.E | ct.SC:
    return self.bind(ctx)(node, frame, name)


def _dispatch_debug(
    self: Dispatch[Expr],
    node: Expr,
    ctx: ct.Caller[ct.E, ct.SC],
    frame: ct.FrameAny,
    name: str,
    /,
) -> ct.E | ct.SC:
    # Provides an opt-in hint for a development-time-only error
    # The original error *looks* like the problem is related to a single backend:
    #   `AttributeError: type object 'ArrowExpr' has no attribute 'lit'`
    try:
        method = self.bind(ctx)
    except AttributeError as err:
        msg = (
            f"`{self.name}` has not been implemented at the compliant-level.\n"
            f"Hint: Try adding `{self.name_full}()`"
        )
        raise NotImplementedError(msg) from err
    return method(node, frame, name)


_DISPATCHER_CALL: Final = (
    _dispatch if not os.environ.get(common.NW_DEV_ENV_NAME) else _dispatch_debug
)


@final
class DispatcherOptions:
    """Class-level configuration for how `FunctionDispatch` should be built.

    Defined via the (optional) `dispatch` parameter at [subclass-definition time].

    Many functions simply use the default:
    >>> from narwhals._plan import expressions as ir
    >>> from narwhals._plan._dispatch import DispatcherOptions
    >>> from narwhals._plan.options import ExplodeOptions
    >>>
    >>> class Explode(ir.Function): ...

    >>> Explode.__expr_ir_dispatch__
    Dispatch<explode>

    >>> Explode.__expr_ir_dispatch__.options
    DispatcherOptions(<default>)

    `dispatch` provides a bit more flexibility when you want it:

    >>> class Explode2(Explode, dispatch=DispatcherOptions.renamed("explodier")): ...

    >>> Explode2.__expr_ir_dispatch__
    Dispatch<explodier>

    >>> Explode2.__expr_ir_dispatch__.options
    DispatcherOptions(override_name='explodier')

    Keep in mind that `options` are inherited:
    >>> class Explode21(Explode2): ...
    >>> Explode21.__expr_ir_dispatch__
    Dispatch<explodier>

    So we'd need another override to get the default back:
    >>> class ExplodeWithOptions(Explode2, dispatch=DispatcherOptions()):
    ...     __slots__ = ("options",)
    ...     options: ExplodeOptions

    >>> ExplodeWithOptions.__expr_ir_dispatch__
    Dispatch<explode_with_options>

    [subclass-definition time]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
    """

    __slots__ = ("accessor_name", "override_name")
    accessor_name: Accessor | None
    """Name of an (optional) expression namespace accessor.

    Required for expressions like:

        nw.col("a").str.len_chars()
        #           ^^^
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

        >>> show_dispatch_names(ir.boolean.Not)
        Not    -> not_

    [snake_case]: https://en.wikipedia.org/wiki/Snake_case
    [PascalCase]: https://en.wikipedia.org/wiki/Naming_convention_(programming)#Examples_of_multiple-word_identifier_formats
    """

    def __init__(
        self, *, accessor_name: Accessor | None = None, override_name: str = ""
    ) -> None:
        self.accessor_name = accessor_name
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


def get_dispatch_name(expr: ir.ExprIR | type[ir.Function], /) -> str:
    """Return the synthesized method name for `expr`.

    Note:
        Refers to the `Compliant*` method name, which may be *either* more general
        or more specialized than what the user called.
    """
    return (expr.function if is_function_expr(expr) else expr).__expr_ir_dispatch__.name
