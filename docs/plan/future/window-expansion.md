# Improve Eager-only window expansion

The performance problem mentioned in [Window functions](../why.md#exprseries) can be solved, but first we need to understand it.

## Example

Here we have two queries. The only difference is how many columns are at the root of the expression:
```py
import narwhals._plan as nwp

data = {
    "group": ["a", "b", "a", "a", "b", "c"],
    "b": [1, 2, 1, 5, 3, 3],
    "c": [5, 4, 3, 6, 2, 1],
}

df = nwp.DataFrame.from_dict(data, backend="pyarrow")

root_single = df.with_columns(nwp.col("b").min().over("group"))
root_multiple = df.with_columns(nwp.col("b", "c").min().over("group"))
```

Let's translate each and see how things differ between `main` vs `narwhals._plan`

### `root_single`

=== "`main`"

    I'm being generous, since these steps are more complex on `main` ([1](https://github.com/narwhals-dev/narwhals/blob/7a9e5b9c622c762b180fda043b9550801fdce748/src/narwhals/_arrow/expr.py#L132-L192), 
    [2](https://github.com/narwhals-dev/narwhals/blob/c177d43c30d306f9a3a851bd378e18aaf5112344/src/narwhals/_arrow/group_by.py#L80-L191), 
    [3](https://github.com/narwhals-dev/narwhals/blob/c177d43c30d306f9a3a851bd378e18aaf5112344/src/narwhals/_expression_parsing.py#L72-L91))
    ```py
    expr_output = (
        df.select("group")
        .join(df.group_by("group").agg(nwp.col("b").min()), on="group")
        .drop("group")
    )
    df.with_columns(expr_output.get_column(name) for name in ["b"])
    ```

=== "`narwhals._plan`"

    Besides knowing we have a single column, they're the same
    ```py
    expr_output = (
        df.select("group")
        .join(df.group_by("group").agg(nwp.col("b").min()), on="group")
        .drop("group")
    )
    df.with_columns(expr_output.get_column("b"))
    ```

### `root_multiple`

=== "`main`"

    This is almost identical to `root_single`:
    ```py
    expr_output = (
        df.select("group")
        .join(df.group_by("group").agg(nwp.col("b", "c").min()), on="group")
        .drop("group")
    )
    df.with_columns(expr_output.get_column(name) for name in ["b", "c"])
    ```

=== "`narwhals._plan`"

    This is bad 😳  
    Our most expensive operators (`join`, `group_by`) are now inside a loop!
    PyArrow can parallelize `group_by`, but we can't utilize that if each aggregate is separated like this:
    ```py
    too_much_work = (
        df.select("group")
        .join(df.group_by("group").agg(nwp.col(name).min()), on="group")
        .drop("group")
        .get_column(name)
        for name in ("b", "c")
    )
    df.with_columns(too_much_work)
    ```

??? question "How did this happen?"

    If we switch over to Polars and compare the two query plans we can see expression expansion in action:

    ```py
    import polars as pl
    lf = pl.LazyFrame(data)
    root_single = lf.with_columns(pl.col("b").min().over("group"))
    root_multiple = lf.with_columns(pl.col("b", "c").min().over("group"))

    print(f"root_single\n{root_single.explain()}\n")
    print(f"root_multiple\n{root_multiple.explain()}")
    ```

    ```
    root_single
     WITH_COLUMNS:
     [col("b").min().over([col("group")])] 
      DF ["group", "b", "c"]; PROJECT */3 COLUMNS

    root_multiple
     WITH_COLUMNS:
     [col("b").min().over([col("group")]), col("c").min().over([col("group")])] 
      DF ["group", "b", "c"]; PROJECT */3 COLUMNS
    ```

    `narwhals._plan` is matching this behavior:

    ```py
    from narwhals._plan._expansion import prepare_projection

    expanded, _ = prepare_projection([nwp.col("b", "c").min().over("group")._ir], schema=df)
    expanded
    ```
    ```
    (b=col('b').min().over([col('group')]), c=col('c').min().over([col('group')]))
    ```

    **But** the outcome is problematic because each expanded expression will aggregate and join in multiple steps.

## Shouldn't we just stick to `main` then?
[^1]: and one I'm very keen to resolve

This is definitely a regression for these kinds of expressions on eager-only backends [^1].

But consider these guys:

```py
import narwhals as nw

problem_for_plan = nw.col("b", "c").min().over("group")
problem_for_both = nw.col("b").min().over("group"), nw.col("c").min().over("group")
```

On `main` and `narwhals._plan`, both inputs mean the same thing. But on `main`, the second has an entirely different performance profile. 

This is not documented and I'm not sure the current situation was intentional. 
`over` was introduced in (#109) and did not have any positive cases for multiple roots.

That was long ago, but in (#3152) support was added that rewrites expressions **into the form that I'm concerned about introducing**:

[docs/how_it_works.md?plain=1#L478-L488](https://github.com/narwhals-dev/narwhals/blob/7a9e5b9c622c762b180fda043b9550801fdce748/docs/how_it_works.md?plain=1#L478-L488)

> ```py
> (nw.col("a").sum() + nw.col("b").sum()).over("c")
> ```
> 
> then `+` is an elementwise operation and so can be swapped with `over`. We just need
> to take care to apply the `over` operation to all the arguments of `+`, so that we
> end up with
> 
> ```py
> nw.col("a").sum().over("c") + nw.col("b").sum().over("c")
> ```

The above expression *should have* been a red flag with our eager-only `over` support. 
IMO, rewritting an **expression** like this only comes from a mindset where the **query plan cannot be rewritten**.

For Pandas & PyArrow, a more preferable query would use half the number of (`join`, `group_by`)s:

```py
import polars as pl
lf = pl.LazyFrame(schema={"a": int, "b": int, "c": str})

user = lf.with_columns((pl.col("a").sum() + pl.col("b").sum()).over("c"))

_lhs = lf.group_by("c").agg(pl.col("a").sum())
_rhs = lf.group_by("c").agg(pl.col("b").sum())
main = (
    lf.drop("a")
    .join(_lhs.join(_rhs, on="c").select(pl.col("a") + pl.col("b"), "c"), on="c") # (1)!
    .select("a", "b", "c")
)

rewrite = (
    lf.drop("a")
    .join(
        lf.group_by("c")
        .agg(pl.col("a", "b").sum())
        .select(pl.col("a") + pl.col("b"), "c"),
        on="c",
    )
    .select("a", "b", "c")
)
```

1. The real thing on `main` may be more complex than this, but this captures 2x (`join`, `group_by`)s

So my answer is that `main` *happens* to have more ideal behavior for a subset of expressions, but it is not consistent.

Window functions for Pandas and PyArrow are unlike the rest of the expression machinery. 
What they are **already doing** is rewriting a query plan, but do so outside of what could be considered the Narwhals' data model.
Because of this, there is no visibility into how much work an expression might require. This has led to introducing things that I believe would have been rejected as proposals, had there been an understanding of the cost.

## How do we fix this?
We avoid regressing with `narwhals._plan` and improve the performance and predicatabilty of `main` (largely) in the same way. 

We fix this by taking what works from `main` and building the tools that express what we are doing for Pandas and PyArrow.
This was the reason for exploring `LogicalPlan`, as I'd like to express what we do for `over` in terms of `LogicalPlan` transforms.  
Alongside this, would be changes to expression expansion where a backend is brought into the loop. A common root expression is helpful in this case.

The benefits of a `LogicalPlan` go beyond this issue. Lazy-only backends are natively represented in this way. There we have many more cases of plan rewriting is masquerading as expression rewriting.

