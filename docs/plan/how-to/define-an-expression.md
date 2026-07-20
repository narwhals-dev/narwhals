# How to define an expression

[ExprIR]: ../../api-reference-plan/ir/expr-ir/index.md

A feature request rolls in for an expression that Polars supports, but Narwhals does not.
You're here to understand how we fix that with [ExprIR].

By the end of this guide, you should be able to answer:

- Is [ExprIR] the right tool for this job?
- (TODO)

## Is [ExprIR] the right tool?
The default for implementing new expressions should be to add a new [`Function`][narwhals._plan._function.Function].
A [`Function`][narwhals._plan._function.Function] is representable as an expression when wrapping it with [`FunctionExpr`][narwhals._plan.expressions.function_expr.FunctionExpr]. 

Knowing when to use each *can be* reduced to **looking at what Polars does** and following along.

[where Polars defines the Expr enum]: https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/crates/polars-plan/src/dsl/expr/mod.rs#L57-L185
[^1]: the rust term is an [Enum variant](https://doc.rust-lang.org/rust-by-example/custom_types/enum.html)

First we find [where Polars defines the Expr enum] in rust. 
What this code means is not important - so do not worry if rust is unfamiliar - just think of this as a fancy [`enum.Enum`][] where members[^1] take arguments.

If we exclude all variants that we've implemented, everything that is left **are candidates** for [ExprIR]:


| Variant            | API                                           |
| ------------------ | --------------------------------------------- |
| `Eval`             | `pl.Expr.list.eval`                           |
| `StructEval`       | `pl.Expr.struct.with_fields`                  |
| `Element`          | `pl.element`                                  |
| `Field`            | `pl.field`                                    |
| `DataTypeFunction` | `pl.{dtype_of,self_dtype,struct_with_fields}` |
| `Rolling`          | `pl.Expr.rolling`                             |
| `Explode`          | `pl.Expr.{explode,list.explode}`              |
| `Gather`           | `pl.Expr.{gather,get}`                        |
| `Slice`            | `pl.Expr.slice`                               |

!!! tip
    If you can tell that the new expression doesn't look like one of those; nice one!  
    You can now move on to [How to define a function](./define-a-function.md)

[^2]: It is simpler to check the type of a single `ExprIR` vs traversing through [`FunctionExpr.args`][narwhals._plan.expressions.function_expr.FunctionExpr.args]

??? question "What makes those operations different?"
    [height is independent of input column(s)]: https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/crates/polars-sql/src/context.rs#L3354-L3361

    - `Eval` and `StructEval`
        - Provide a unique context for evaluating their expression arguments against elements/fields of their root expression
    - `Element` and `Field`
        - Valid only within their respective `*Eval` contexts
    - `DataTypeFunction`
        - Valid in specific UDF contexts, and can also be materialized to a `DataType` ([see also](https://docs.pola.rs/api/python/stable/reference/datatype_expr/index.html))
    - `Rolling`
        - Provides a unique context to define groups for window functions (like [`Over`][narwhals._plan.expressions.Over] and [`OverOrdered`][narwhals._plan.expressions.OverOrdered])
    - `Explode`, `Gather` and `Slice`
        - Length-changing expressions, where the [height is independent of input column(s)] ([see also](https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/crates/polars-plan/src/plans/aexpr/projection_height.rs#L110-L200))

    There are common themes, but the main questions to ask are:
    
    1. **What did Polars do?**
    2. Can we describe the operation using the tools [`Function`][narwhals._plan._function.Function] provides?
    3. If we extended [`Function`][narwhals._plan._function.Function] for some new operation, will this ever be reusable?
        1. Conversely, if the operation is unique, can we benefit from identifying it more easily? [^2]


Still here? Okay, let's try defining a new subclass of [ExprIR].

## Defining an expression
On the menu today is [`pl.Expr.explode`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.explode.html#polars.Expr.explode), which is described as:

> Explode a list expression.
> This means that every item is expanded to a new row.

[`Explode`](https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/crates/polars-plan/src/dsl/expr/mod.rs#L115-L119)


<!---TODO: python (polars)/rust code blocks

- What can we learn from each?
  - Takes a single expression (1x `expr: ExprIR = node()`)
- What is different and why?
  - 2x bool -> ExplodeOptions (introduces immutable options that can share logic between expr/series/*frame)
--->


<!---TODO: Describing the expression
- `is_scalar`?
- `is_length_preserving`?
- `changes_length`?
- dtype?
- repr
--->

<!---TODO: Add an `Expr` method to create it

End by attempting to use the method and showing the error telling us we haven't finished yet
-->


<!---TODO: Next steps (shared page for Compliant)-->
