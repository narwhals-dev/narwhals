# Avoiding the `UserWarning` error while using Pandas `group_by`

## Introduction

If you have ever experienced the

> UserWarning: Found complex group-by expression, which can't be expressed efficiently with the pandas API. If you can, please rewrite your query such that group-by aggregations are simple (e.g. mean, std, min, max, ...)

message while using the narwhals `group_by()` method this is for you. If you haven't, this is also for you as you might experience it and you need to know how to avoid it.

The `UserWarning` is thrown for exactly the reason that is given. The pandas API most likely cannot efficiently handle the complexity of the aggregation operations you are trying to run. Take the following two codes as an example.

When you run:

=== "Approach 1"
    ```python exec="true" source="above" result="python" session="df_ex1"
    import narwhals as nw
    import pandas as pd

    data = {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": [10, 20, 30, 40, 50]}

    df_pd = pd.DataFrame(data)


    @nw.narwhalify
    def approach_1(df):

        # Pay attention to this next line
        df = df.group_by("a").agg(d=(nw.col("b") + nw.col("c")).sum())

        return df


    print(approach_1(df_pd))
    ```

=== "Approach 2"
    ```python exec="true" source="above" result="python" session="df_ex2"
    import narwhals as nw
    import pandas as pd

    data = {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": [10, 20, 30, 40, 50]}

    df_pd = pd.DataFrame(data)


    @nw.narwhalify
    def approach_2(df):

        # Pay attention to this next line
        df = df.with_columns(d=nw.col("b") + nw.col("c")).group_by("a").agg(nw.sum("d"))

        return df


    print(approach_2(df_pd))
    ```


Both Approaches shown above return the exact same result, but Approach 1 is inefficient and returns the following error message.


```
UserWarning: Found complex group-by expression, which can't be expressed efficiently with the pandas API. If you can, please rewrite
your query such that group-by aggregations are simple (e.g. mean, std, min, max, ...).
```

You might rightly ask, what makes the first approach inefficient and the second approach efficient? It comes down to what the
pandas API lets us express.

## Approach 1
```python
# From line 11

return df.group_by("a").agg((nw.col("b") + nw.col("c")).sum().alias("d"))
```

To translate this to pandas, we would do:
```python
df.groupby("a").apply(lambda df: (df["b"] + df["c"]).sum().rename("d"))
```
Any time you use `apply` in pandas, that's a performance footgun - best to avoid it and use vectorised operations instead.
Let's take a look at how "approach 2" gets translated to pandas to see the difference.

## Approach 2
```python
# Line 11 in Approach 2

return df.with_columns(d=nw.col("b") + nw.col("c")).group_by("a").agg(nw.sum("d"))
```

This gets roughly translated to:
```python
df.assign(d=lambda df: df["b"] + df["c"]).groupby("a")["d"].sum()
```
Because we're using pandas' own API, as opposed to `apply` and a custom `lambda` function, then this is going to be efficient.

## Tips for Avoiding the `UserWarning`

To ensure efficiency and avoid warnings similar to those seen in Approach 1, we recommend that you follow these practices:

1. Decompose Complex Operations: Break down complex transformations into simpler steps. In this case, keep the `.agg` method simple. Compute new columns first, then use these columns in aggregation or other operations.
2. Avoid Redundant Computations: If an operation (like addition) is used multiple times, compute it once and store the result in a new column.
3. Leverage Built-in Functions: Use built-in functions provided by the DataFrame library. In this case, using the `with_columns()` method allows you to pre-compute before grouping and aggregation.

By following these guidelines, you can are sure to avoid the error: `UserWarning: Found complex group-by expression, which can't be expressed efficiently with the pandas API. If you can, please rewrite your query such that group-by aggregations are simple (e.g. mean, std, min, max, ...)`.

**_Happy grouping!_** 🫡
