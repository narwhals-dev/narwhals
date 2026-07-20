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
On the menu today is [`pl.Expr.explode`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.explode.html#polars.Expr.explode) which is described as:

> Explode a list expression.  
> This means that every item is expanded to a new row.


[^3]: Later, we will see that you cannot rely on the docs!

That description is short, yet gives us a lot of clues to what we need to model:

1. We take a single expression argument
2. That expression *should be* [^3] [`List`][narwhals.dtypes.List]-typed
3. We should expect `explode` to change the length of the output column

Now, we could jump into creating a subclass - but it would be a better idea to check if Polars can do any of the work for us:

=== "Python"

    [Python definition]: https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/py-polars/src/polars/expr/expr.py#L5383-L5386

    Starting with the most familiar, the [Python definition] gives us the signature we need for `Expr.explode`:

    ```py
    class Expr:
        def explode(self, *, empty_as_null: bool = True, keep_nulls: bool = True) -> Expr: ...
    ```

=== "Rust (PyO3)"

    Before we reach the promised land, we need to [go through](https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/crates/polars-python/src/expr/general.rs#L474-L483) [PyO3](https://pyo3.rs/).

    This step is usually not very exciting, but today it ~~is!~~ helps explain what an `ExplodeOptions` is:

    ```rs
    impl PyExpr {
        fn explode(&self, empty_as_null: bool, keep_nulls: bool) -> Self {
            self.inner  // (1)!
                .clone()
                .explode(ExplodeOptions {
                    empty_as_null,
                    keep_nulls,
                })
                .into()
        }
    }
    ```

    1. [`PyExpr.inner`](https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/crates/polars-python/src/expr/mod.rs#L41)

=== "Rust"

    [we made it!]: https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/crates/polars-plan/src/dsl/expr/mod.rs#L115-L118
    [current expression]: https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/crates/polars-plan/src/dsl/mod.rs#L219-L226

    Ah [we made it!] Along the way, some keyword-only arguments got packaged up into `ExplodeOptions` and the [current expression] became the `input`.

    While not the most common, [`input`][] is a python builtin function - so we will generally use `expr` in it's place:

    ```rs
    pub enum Expr {
        Explode {
            input: Arc<Expr>,
            options: ExplodeOptions,
        },
    }

    /// Explode the String/List column.
    impl Expr {  // (1)!
        pub fn explode(self, options: ExplodeOptions) -> Self {
            Expr::Explode {
                input: Arc::new(self),
                options,
            }
        }
    }
    ```

    1. I warned you about relying on the docs 😉, so we can use String too?


??? question "But what did X mean?"

    If you saw anything in rust along the way that wasn't mentioned - consider it noise.
    You will not be required to write or even fully understand the rust code that is shown.  
    It is here to help illustrate where ideas come from and where to look for answers when needed. 

    The rust codebase is enormous. The skill we need is cutting through the noise to *just the bits that help us*.



<!---TODO: Describing the expression
- `is_scalar`?
- `is_length_preserving`?
- `changes_length`?
- dtype?
    - https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/crates/polars-plan/src/dsl/mod.rs#L219-L220
    - https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/crates/polars-plan/src/plans/aexpr/schema.rs#L87-L98
- repr
    - https://github.com/pola-rs/polars/blob/0189b4682833082cf4dde3b263f564dcf4ae426a/crates/polars-plan/src/dsl/format.rs#L68-L84
--->

<!---TODO: Add an `Expr` method to create it

End by attempting to use the method and showing the error telling us we haven't finished yet
-->


<!---TODO: Next steps (shared page for Compliant)-->
