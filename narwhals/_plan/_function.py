"""Where the base `Function` lives.

## Implementation Notes
The design was adapted from an *older* version of (rust) polars.

[`dsl::function_expr::FunctionExpr`] became `Function`

```rust
pub enum Expr {                    // class ExprIR: ...
                                   // class Function: ...
                                   //
    Function {                     // class FunctionExpr(ExprIR):
        input: Vec<Expr>,          //     input: Seq[ExprIR]
        function: FunctionExpr,    //     function: Function
    //            ^^^^^^^^^^^^                      ^^^^^^^^
        options: FunctionOptions,  //     flags: FunctionFlags
    },
```

This *fleeting* version was bookended by 2 PRs landing a couple months apart:
- [refactor: Separate `FunctionOptions` from DSL calls]
- [refactor: Separate `FunctionExpr` and `IRFunctionExpr`]

Interestingly, that meant:
- we got all the definitions of [`FunctionOptions`]
- but didn't need to introduce [another layer] to access them at their [new home]

[`dsl::function_expr::FunctionExpr`]: https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L121-L1402
[refactor: Separate `FunctionOptions` from DSL calls]: https://github.com/pola-rs/polars/pull/22133
[refactor: Separate `FunctionExpr` and `IRFunctionExpr`]: https://github.com/pola-rs/polars/pull/23140
[`FunctionOptions`]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/options.rs#L54-L281
[new home]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/mod.rs#L994-L1238
[another layer]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/mod.rs#L258-L267
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from narwhals._plan._dispatch import Dispatcher, DispatcherOptions
from narwhals._plan._dtype import IntoResolveDType, ResolveDType
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._immutable import Immutable
from narwhals._plan.exceptions import function_expr_invalid_operation_error
from narwhals.dtypes import DType

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from typing_extensions import Self

    from narwhals._plan.expressions import ExprIR, FunctionExpr
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import Seq

__all__ = ["Function", "HorizontalFunction"]

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
namespaced = DispatcherOptions.namespaced
ELEMENTWISE = FunctionFlags.ELEMENTWISE


# TODO @dangotbanned: Introduce the concept of *arity* + replace `unwrap_input`
# TODO @dangotbanned: `RangeExpr` (and probably others) have input shape requirements
class Function(Immutable):
    r"""A general transformation applied to an expression.

    A `Function` is distinct from an expression but appears in many of them as:

        FunctionExpr[Function]

    **Instances** capture non-expression arguments to `Expr` methods:

    >>> import narwhals._plan as nw
    >>> from narwhals._plan import expressions as ir
    >>> from narwhals._plan.expressions import functions as F

    >>> expr = nw.col("a").shift(2)
    >>> expr_ir = expr._ir
    >>> isinstance(expr_ir, ir.FunctionExpr)
    True

    >>> print(f"Function(args) : {expr_ir.function}\nExprIR input(s): {expr_ir.input[0]}")
    Function(args) : Shift(n=2)
    ExprIR input(s): Column(name='a')

    Whereas **classes** encode most of the details, like...

    What properties does the function have?
    >>> F.Shift.__function_flags__
    <FunctionFlags.LENGTH_PRESERVING: 4>

    Which `CompliantExpr` method to call?
    >>> F.Shift.__expr_ir_dispatch__
    Dispatcher<shift>

    Does it transform the datatype?
    >>> F.Shift.__expr_ir_dtype__
    function.same_dtype()

    See Also:
        `narwhals._plan._function.py` doc for implementation notes
    """

    __function_flags__: ClassVar[FunctionFlags] = FunctionFlags.DEFAULT
    """Defines properties of the function.

    Flags tell us how a function transforms the shape of it's input:

        ┌───────────────────┬───────────────┬───────────┐
        │ Flag              ┆ Input         ┆ Output    │
        ╞═══════════════════╪═══════════════╪═══════════╡
        │ AGGREGATION       ┆ Column        ┆ Scalar    │
        │ ROW_SEPARABLE     ┆ Column        ┆ Unknown   │
        │ LENGTH_PRESERVING ┆ Column/Scalar ┆ Preserved │
        │ ELEMENTWISE       ┆ Column/Scalar ┆ Preserved │
        └───────────────────┴───────────────┴───────────┘

    And that's the main nugget we can use to answer the question:
    > Is this function valid *here*?

    See `FunctionFlags` for examples.

    To customize the behavior, use the `flags` **parameter** [when subclassing]:

        class FillNull(Function, flags=FunctionFlags.ELEMENTWISE): ...

    [when subclassing]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
    """

    __expr_ir_dispatch__: ClassVar[Dispatcher[FunctionExpr[Self]]] = Dispatcher()
    """Callable that dispatches to the appropriate compliant-level method.

    See `Dispatcher` and `DispatcherOptions` for examples.

    To customize the behavior, use the `dispatch` **parameter** [when subclassing]:

        class AllHorizontal(Function, dispatch=DispatcherOptions.namespaced()): ...

    Notes:
        Each class has their own `Dispatcher` instance, and inheritance is only on the `options` property.

    [when subclassing]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
    """

    __expr_ir_dtype__: ClassVar[ResolveDType[FunctionExpr[Self]]] = ResolveDType()
    """Callable defining how a `DType` is derived when `resolve_dtype` is called.

    If the logic fits an existing pattern, use the `dtype` **parameter** [when subclassing]:

        class FillNullWithStrategy(Function, dtype=ResolveDType.function.same_dtype()):
            __slots__ = ("limit", "strategy")
            strategy: FillNullStrategy
            limit: int | None

    See `IntoResolveDType` and `ResolveDType` for more examples.

    If nothing there *quite* scratches the itch, override `resolve_dtype` instead:

        class Pow(Function):
            def resolve_dtype(self, node: FunctionExpr, schema: FrozenSchema, /) -> DType:
                base = node.input[0].resolve_dtype(schema)
                if base.is_integer():
                    if (exp := node.input[1].resolve_dtype(schema)).is_float():
                        return exp
                return base

    [when subclassing]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
    """

    def is_elementwise(self) -> bool:
        return self.__function_flags__.is_elementwise()

    def _validate_input(self, input: Seq[ExprIR], /) -> Seq[ExprIR]:  # noqa: A002
        # NOTE: (Hacky) hook for arbitrary validation
        # Ideally this would be more declarative
        parent = input[0]
        if parent.is_scalar() and not self.is_elementwise():
            raise function_expr_invalid_operation_error(self, parent)
        return input

    def to_function_expr(self, *inputs: ExprIR) -> FunctionExpr[Self]:
        """Wrap this `Function` in a `FunctionExpr`.

        Arguments:
            *inputs: Expression arguments for the function.
                The first input is the root and responsible for the output name.
        """
        # NOTE: Defined as a method to allow these guys to override:
        # - `RollingWindow`, `MapBatches`, `RangeFunction`, `StructFunction`
        return _import_function_expr()(input=self._validate_input(inputs), function=self)

    def __init_subclass__(
        cls: type[Self],
        *,
        dispatch: DispatcherOptions | None = None,
        dtype: IntoResolveDType[Self] | None = None,
        flags: FunctionFlags | None = None,
        **_: Any,
    ) -> None:
        """Hook to [customize a new subclass] of `Function`.

        All parameters are optional and will be inherited when not provided to `__init_subclass__`.

        Arguments:
            dispatch: Defines how to build a `Dispatcher`.
                Stored in `__expr_ir_dispatch__.options`.

            dtype: Defines how a `DType` is derived when `resolve_dtype` is called.
                Stored in `__expr_ir_dtype__`.

                See `IntoResolveDType` and `ResolveDType` for usage.

                **Warning**: This functionality is considered **unstable**.
                Full support depends on [#3396].

            flags: Defines how a function transforms the shape of it's input.
                Stored in `__function_flags__`.

        [customize a new subclass]: https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
        [#3396]: https://github.com/narwhals-dev/narwhals/pull/3396
        """
        super().__init_subclass__(**_)
        if flags is not None:
            cls.__function_flags__ = flags
        cls.__expr_ir_dispatch__ = Dispatcher.from_function(cls, dispatch)
        if dtype is not None:
            if isinstance(dtype, DType):
                dtype = ResolveDType.just_dtype(dtype)
            elif not isinstance(dtype, ResolveDType):
                dtype = ResolveDType.function.visitor(dtype)
            # TODO @dangotbanned: fix mypy
            # error: Incompatible types in assignment (expression has type "FunctionVisitor[Any] | ResolveDType[Any]",
            # variable has type "ResolveDType[FunctionExpr[Self]]"
            cls.__expr_ir_dtype__ = dtype  # type: ignore[assignment]

    def __repr__(self) -> str:
        return self.__expr_ir_dispatch__.name

    def resolve_dtype(self, node: FunctionExpr[Any], schema: FrozenSchema, /) -> DType:
        """Get the data type of an expanded expression.

        Arguments:
            node: The expanded expression, wrapping this function.
            schema: The same schema used to project `node`.
        """
        return self.__expr_ir_dtype__(node, schema)


class HorizontalFunction(Function, flags=ELEMENTWISE, dispatch=namespaced()):
    """Transformations *across* columns.

    Special cases of [fold] or [reduce].

    ## Examples
    These functions use different semantics when expanding selectors.

    Say we have the following schema:
    >>> from tests.plan.utils import Frame
    >>> import narwhals._plan as nw

    >>> df = Frame.from_names("a", "b", "c")
    >>> df.schema
    Schema({'a': Int64, 'b': Int64, 'c': Int64})

    We expand multiple inputs into a single output:
    >>> before = nw.sum_horizontal(nw.all())
    >>> (reduced,) = df.project(before)
    >>> before._ir
    ncs.all().sum_horizontal()
    >>> reduced
    a=col('a').sum_horizontal([col('b'), col('c')])

    Whereas the more common form of expansion produces multiple outputs:
    >>> before = nw.all().clip("b")
    >>> before._ir
    ncs.all().clip_lower([col('b')])
    >>> df.project(before)  # doctest: +NORMALIZE_WHITESPACE
    (a=col('a').clip_lower([col('b')]),
     b=col('b').clip_lower([col('b')]),
     c=col('c').clip_lower([col('b')]))

    [fold]: https://docs.pola.rs/user-guide/expressions/folds/
    [reduce]: https://mathspp.com/blog/pydonts/the-power-of-reduce
    """


@cache
def _import_function_expr() -> type[FunctionExpr[Any]]:
    # NOTE: Very heavily used (`Function.to_function_expr`), but creates a cycle
    from narwhals._plan.expressions.expr import FunctionExpr

    return FunctionExpr
