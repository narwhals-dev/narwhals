"""Where the base `Function` lives.

### Implementation Notes
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
        options: FunctionOptions,  //     options: FunctionOptions
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

from narwhals._plan._dispatch import Dispatcher
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._immutable import Immutable
from narwhals._plan.common import replace
from narwhals._plan.options import FEOptions, FunctionOptions
from narwhals.dtypes import DType

if TYPE_CHECKING:
    from typing import Any, Callable, ClassVar

    from typing_extensions import Self

    from narwhals._plan.expressions import ExprIR, FunctionExpr
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import Accessor

__all__ = ["Function", "HorizontalFunction"]


# TODO @dangotbanned: Finish `Function` class doc
# TODO @dangotbanned: Add `ClassVar` docs
# TODO @dangotbanned: Introduce the concept of *arity* + replace `unwrap_input`
class Function(Immutable):
    r"""A general transformation applied to an expression.

    A `Function` is distinct from an expression but appears in many of them as:

        class FunctionExpr(ExprIR):
            function: Function

    Instances capture non-expression arguments to `Expr` methods:

        >>> import narwhals._plan as nwp
        >>> from narwhals._plan import expressions as ir

        >>> expr = nwp.col("a").shift(2)
        >>> expr_ir = expr._ir
        >>> isinstance(expr_ir, ir.FunctionExpr)
        True

        >>> print(
        ...     f"Function(args) : {expr_ir.function}\n"
        ...     f"ExprIR input(s): {expr_ir.input[0]}"
        ... )
        Function(args) : Shift(n=2)
        ExprIR input(s): Column(name='a')


    <!--TODO @dangotbanned: Finish this section -->

    Classes store all other details:
    - Which `CompliantExpr` method to call (`__expr_ir_dispatch__: Dispatcher`)
    - How the function changes the larger expression it is part of (`function_options: FunctionOptions`)

    See Also:
        `narwhals._plan._function.py` doc for implementation notes
    """

    # TODO @dangotbanned: Fix this wart
    # - `ExprIR` was updated a while back to remove `ClassVar[staticmethod]`
    # - Naming convention isn't consistent
    _function_options: ClassVar[staticmethod[[], FunctionOptions]] = staticmethod(
        FunctionOptions.default
    )
    __expr_ir_config__: ClassVar[FEOptions] = FEOptions.default()
    __expr_ir_dispatch__: ClassVar[Dispatcher[FunctionExpr[Self]]]
    __expr_ir_dtype__: ClassVar[ResolveDType[FunctionExpr[Self]]] = ResolveDType()

    @property
    def function_options(self) -> FunctionOptions:
        return self._function_options()

    @property
    def is_scalar(self) -> bool:
        return self.function_options.returns_scalar()

    def to_function_expr(self, *inputs: ExprIR) -> FunctionExpr[Self]:
        from narwhals._plan.expressions.expr import FunctionExpr

        return FunctionExpr(input=inputs, function=self, options=self.function_options)

    def __init_subclass__(
        cls: type[Self],
        *args: Any,
        accessor: Accessor | None = None,
        options: Callable[[], FunctionOptions] | None = None,
        config: FEOptions | None = None,
        dtype: DType | ResolveDType[Any] | Callable[[Self], DType] | None = None,
        **kwds: Any,
    ) -> None:
        super().__init_subclass__(*args, **kwds)
        if accessor_name := accessor or cls.__expr_ir_config__.accessor_name:
            config = replace(config or FEOptions.default(), accessor_name=accessor_name)
        if options:
            cls._function_options = staticmethod(options)
        if config:
            cls.__expr_ir_config__ = config
        cls.__expr_ir_dispatch__ = Dispatcher.from_function(cls)
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


class HorizontalFunction(
    Function, options=FunctionOptions.horizontal, config=FEOptions.namespaced()
): ...
