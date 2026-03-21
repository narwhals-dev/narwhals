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

from typing import TYPE_CHECKING

from narwhals._plan._dispatch import Dispatcher, DispatcherOptions
from narwhals._plan._dtype import IntoResolveDType, ResolveDType
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._immutable import Immutable
from narwhals.dtypes import DType

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from typing_extensions import Self

    from narwhals._plan.expressions import ExprIR, FunctionExpr
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import Accessor

__all__ = ["Function", "HorizontalFunction"]

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
namespaced = DispatcherOptions.namespaced
ELEMENTWISE = FunctionFlags.ELEMENTWISE


# TODO @dangotbanned: Finish `Function` class doc
# TODO @dangotbanned: Add `ClassVar` docs
# TODO @dangotbanned: Introduce the concept of *arity* + replace `unwrap_input`
class Function(Immutable):
    r"""A general transformation applied to an expression.

    A `Function` is distinct from an expression but appears in many of them as:

        class FunctionExpr(ExprIR):
            input: tuple[ExprIR, ...]
            function: Function

    Instances capture non-expression arguments to `Expr` methods:

    >>> import narwhals._plan as nw
    >>> from narwhals._plan import expressions as ir

    >>> expr = nw.col("a").shift(2)
    >>> expr_ir = expr._ir
    >>> isinstance(expr_ir, ir.FunctionExpr)
    True

    >>> print(f"Function(args) : {expr_ir.function}\nExprIR input(s): {expr_ir.input[0]}")
    Function(args) : Shift(n=2)
    ExprIR input(s): Column(name='a')


    <!--TODO @dangotbanned: Finish this section -->

    Classes store all other details:
    - Which `CompliantExpr` method to call (`__expr_ir_dispatch__: Dispatcher`)
    - How the function changes the larger expression it is part of (`flags: FunctionFlags`)

    See Also:
        `narwhals._plan._function.py` doc for implementation notes
    """

    # TODO @dangotbanned: Naming convention isn't consistent
    _flags: ClassVar[FunctionFlags] = FunctionFlags.DEFAULT
    __expr_ir_dispatch__: ClassVar[Dispatcher[FunctionExpr[Self]]] = Dispatcher()
    __expr_ir_dtype__: ClassVar[ResolveDType[FunctionExpr[Self]]] = ResolveDType()

    @property
    def flags(self) -> FunctionFlags:
        return self._flags

    def is_scalar(self) -> bool:  # pragma: no cover
        return FunctionFlags.AGGREGATION in self.flags

    def to_function_expr(self, *inputs: ExprIR) -> FunctionExpr[Self]:
        from narwhals._plan.expressions.expr import FunctionExpr

        return FunctionExpr(input=inputs, function=self, flags=self.flags)

    def __init_subclass__(
        cls: type[Self],
        *,
        accessor: Accessor | None = None,
        flags: FunctionFlags | None = None,
        dispatch: DispatcherOptions | None = None,
        dtype: IntoResolveDType[Self] | None = None,
        **kwds: Any,
    ) -> None:
        super().__init_subclass__(**kwds)
        if flags is not None:
            cls._flags = flags
        cls.__expr_ir_dispatch__ = Dispatcher.from_function(cls, dispatch, accessor)
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


# TODO @dangotbanned: Add summary that's about the reduction behavior
# (selectors examples are fine later)
class HorizontalFunction(Function, flags=ELEMENTWISE, dispatch=namespaced()):
    """_summary_.

    These functions use different semantics when expanding selectors.

    ## Examples
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
    """
