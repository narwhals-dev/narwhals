# Machine Learning Patterns in Narwhals

There are two types of ML developers:
- Those who write different code for different DataFrame backends
- Those who want their code to work consistently everywhere

Narwhals aims to help the second group! Let's learn about the patterns that make this possible.

## 1. Backward Compatibility

When building ML pipelines, you want your code to remain stable across updates. Narwhals provides this through its stable API:

```python exec="1" source="material-block" session="ex1"
import narwhals.stable.v1 as nw
from narwhals.typing import IntoFrameT
import pandas as pd

# Example dataset
data = {"feature1": [1, 2, 3], "feature2": [4, 5, 6]}


def backward_compatible_workflow(df: IntoFrameT) -> IntoFrameT:
    """Use Narwhals stable.v1 API to process data."""
    # Convert to Narwhals lazy frame
    df_nw = nw.from_native(df)

    # Perform transformations
    df_transformed = df_nw.select(
        [
            nw.col("feature1").mean().alias("mean_feature1"),
            nw.col("feature2").sum().alias("sum_feature2"),
        ]
    )

    # Convert back to native format (e.g., Pandas)
    return df_transformed.to_native()
```

What makes this work? The stable API uses semantic versioning with additional guarantees:
1. `narwhals.stable.v1` is frozen - no breaking changes ever
2. New features only appear in new major versions (v2, v3, etc.)
3. Multiple versions can coexist, so you can migrate gradually:

```python exec="true" source="material-block" result="python" session="ex1"
# Testing the backward-compatible workflow
import pandas as pd

df_pd = pd.DataFrame(data)
result = backward_compatible_workflow(df_pd)

print("Result using Narwhals stable API v1:")
print(result)
```

## 2. The Collect-Then-Item Pattern

If you've ever seen errors like:
> AttributeError: 'LazyFrame' object has no attribute 'item'

or

> AttributeError: 'DataFrame' object has no attribute 'collect'

This pattern is for you. The challenge comes from how different backends handle materialization:

- Pandas (eager): Values are always materialized, just use `item()`
- Polars (mixed): Some operations are lazy, needs `collect()` then `item()`
- Dask (lazy): Everything is lazy until explicitly materialized

Here's how to handle all cases consistently:

=== "Incorrect Approach"
```python exec="1" source="material-block" session="ex2"
import narwhals as nw
from narwhals.typing import FrameT
import pandas as pd
import polars as pl
import dask.dataframe as dd

# Create sample ML dataset
data = {
    "numeric_feature": [1.5, 2.0, None, 4.0, 5.5],  # Has missing value
    "categorical_feature": ["A", "B", "A", "C", "B"],  # Needs encoding
}


@nw.narwhalify
def get_mean_wrong(df: FrameT, column: str) -> float:
    # This fails with Dask - no collect()
    return df.select([nw.col(column).mean()]).item()
```

=== "Also Incorrect"
    ```python exec="1" source="above" session="ex2"
    @nw.narwhalify
    def get_mean_also_wrong(df: FrameT, column: str) -> float:
        # This fails with Pandas - no collect() method
        return df.select([nw.col(column).mean()]).collect().item()
    ```

=== "Correct Approach"
    ```python exec="1" source="above" session="ex2"
    @nw.narwhalify
    def get_mean(df: FrameT, column: str) -> float:
        result = df.select([nw.col(column).mean()])
        # Check if we need collect()
        return result.item() if not hasattr(result, "collect") else result.collect().item()
    ```

Under the hood:
1. `hasattr(result, 'collect')` checks if we're dealing with a lazy frame
2. For eager frames (Pandas): Just use `item()` directly
3. For lazy frames (Dask): First `collect()` to materialize, then `item()`
4. For mixed frames (Polars): Same as lazy frames

Let's see it work across all backends:

```python exec="true" source="material-block" result="python" session="ex2"
# Start with any lazy backend (e.g., Dask)
df_pd = pd.DataFrame(data)
df_dask = dd.from_pandas(df_pd, npartitions=2)

# Test across backends
backends = {
    "Pandas (eager)": df_pd,
    "Polars (mixed)": pl.DataFrame(data),
    "Dask (lazy)": df_dask,
}

for name, df in backends.items():
    df_nw = nw.from_native(df)
    try:
        result = get_mean_wrong(df_nw, "numeric_feature")
        print(f"{name}: {result}")
    except Exception as e:
        print(f"{name}: Failed - {str(e)}")
```

## 3. Data Validation

When validating data for ML, you often need to handle custom objects or mixed types. This is especially tricky because:
- Different backends handle non-standard types differently
- You need different behavior in development vs production
- Error messages should be helpful for debugging

Narwhals solves this with two modes:

=== "Development Mode"
    ```python exec="1" source="material-block" session="ex3"
    import narwhals as nw
    from narwhals.typing import FrameT
    import pandas as pd

    # Create sample data for demonstration
    class CustomObject:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return f"Custom({self.value})"


    data = {
        "feature": [1, 2, 3],
        "custom": [CustomObject(1), CustomObject(2), CustomObject(3)],
    }
    df = pd.DataFrame(data)

    # Allow inspection of problematic data
    df_nw = nw.from_native(df, pass_through=True)
    ```

=== "Production Mode"
    ```python exec="1" source="above" session="ex3"
    # Strict validation for production
    df_nw = nw.from_native(df, pass_through=False)
    ```

What's happening under the hood?
1. Development mode (`pass_through=True`):
   - Wraps unsupported objects without conversion
   - Allows inspection of problematic data
   - Operations fail only when actually using bad columns
   
2. Production mode (`pass_through=False`):
   - Validates all columns immediately
   - Fails fast if any column has unsupported types
   - Prevents bad data from entering your pipeline

Let's see the difference:

```python exec="true" source="material-block" result="python" session="ex3"
class CustomObject:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Custom({self.value})"


data = {
    "feature": [1, 2, 3],
    "custom": [CustomObject(1), CustomObject(2), CustomObject(3)],
}
df = pd.DataFrame(data)

print("Development Mode:")
print("-" * 50)
df_dev = nw.from_native(df, pass_through=True)
print("1. Load data (succeeds):")
print(df_dev.to_native())

print("\n2. Use good column (succeeds):")
result = df_dev.select([nw.col("feature").mean()])
print(result.to_native())

print("\n3. Use bad column (fails helpfully):")
try:
    result = df_dev.select([nw.col("custom").mean()])
except Exception as e:
    print(f"Error: {str(e)}")

print("\nProduction Mode:")
print("-" * 50)
print("1. Load data (fails fast):")
try:
    df_prod = nw.from_native(df, pass_through=False)
except Exception as e:
    print(f"Error: {str(e)}")
```

## 4. Feature Engineering

When preprocessing features for ML, you need to handle both eager operations (like computing statistics) and lazy operations (like transformations). The challenge is doing this efficiently across backends.

Here's what can go wrong:

=== "Memory Inefficient"
    ```python exec="1" source="material-block" session="ex4"
    import narwhals as nw
    from narwhals.typing import FrameT
    import pandas as pd


    @nw.narwhalify
    def process_feature_inefficient(df: FrameT, column: str) -> FrameT:
        # Bad: Materializes entire column just to compute mean
        mean_val = df.select([nw.col(column)]).collect().to_numpy().mean()
        return df.select([nw.col(column).fill_null(mean_val)])
    ```

=== "Type Unsafe"
    ```python exec="1" source="above" session="ex4"
    @nw.narwhalify
    def process_feature_unsafe(df: FrameT, column: str) -> FrameT:
        # Bad: No type casting before computing mean
        mean_val = df.select([nw.col(column).mean()]).item()
        return df.select([nw.col(column).fill_null(mean_val)])
    ```

=== "Correct Approach"
    ```python exec="1" source="above" session="ex4"
    @nw.narwhalify(eager_only=True)
    def process_feature(df: FrameT, column: str) -> FrameT:
        # 1. Cast to correct type first
        # 2. Use collect-then-item pattern
        # 3. Keep final transformation lazy
        result = df.select([nw.col(column).cast(nw.Float64()).mean()])
        mean_val = (
            result.item() if not hasattr(result, "collect") else result.collect().item()
        )

        return df.select(
            [nw.col(column).cast(nw.Float64()).fill_null(mean_val).alias(column)]
        )
    ```

The correct approach:
1. Uses `eager_only=True` to signal immediate value needs
2. Casts to proper type before computing statistics
3. Uses collect-then-item pattern for materialization
4. Keeps final transformation lazy for optimization

Let's see it handle tricky data:

```python exec="true" source="material-block" result="python" session="ex4"
data = {
    "clean": [1.0, None, 3.0],
    "mixed": ["1.0", None, "3.0"],
    "invalid": ["1.0", "bad", "3.0"],
}
df = pd.DataFrame(data)
df_nw = nw.from_native(df)

print("Clean numeric data:")
print(process_feature(df_nw, "clean"))

print("\nMixed string/numeric:")
print(process_feature(df_nw, "mixed"))

print("\nInvalid data (fails safely):")
try:
    print(process_feature(df_nw, "invalid"))
except Exception as e:
    print(f"Error: {str(e)}")
```

## 5. Time Series Validation

Time series data adds extra complexity:
- Timestamps must be unique within groups
- Different backends handle timestamps differently
- Operations must stay lazy for large datasets

Here's how to handle it:

=== "Memory Inefficient"
    ```python exec="1" source="material-block" session="ex5"
    import narwhals as nw
    from narwhals.typing import FrameT
    import pandas as pd


    @nw.narwhalify
    def check_duplicates_inefficient(df: FrameT, id_col: str, time_col: str) -> FrameT:
        # Bad: Materializes entire frame to check duplicates
        return (
            df.collect()
            .group_by([id_col, time_col])
            .agg([nw.col(time_col).count().alias("count")])
            .filter(nw.col("count") > 1)
        )
    ```

=== "Correct Approach"
    ```python exec="1" source="above" session="ex5"
    @nw.narwhalify
    def check_duplicates(df: FrameT, id_col: str, time_col: str) -> FrameT:
        # 1. Group by stays lazy
        # 2. Count stays lazy
        # 3. Filter stays lazy
        counts = df.group_by([id_col, time_col]).agg(
            [nw.col(time_col).count().alias("count")]
        )
        return counts.filter(nw.col("count") > 1)
    ```

Under the hood:
1. `group_by` creates a lazy grouped frame
2. `agg` defines the computation but doesn't execute
3. `filter` adds to the computation plan
4. Final result stays lazy until needed

Let's test with real data:

```python exec="true" source="material-block" result="python" session="ex5"
import pandas as pd

# Create time series with known issues
dates = pd.date_range("2023-01-01", periods=3, freq="H")
data = {
    "id": [1, 1, 1, 2, 2],
    "timestamp": [
        dates[0],  # First timestamp
        dates[0],  # Duplicate for id=1
        dates[1],
        dates[0],  # Duplicate for id=2
        dates[2],
    ],
}
df = pd.DataFrame(data)
df_nw = nw.from_native(df)

print("Found duplicates:")
print(check_duplicates(df_nw, "id", "timestamp"))
```

## 6. Environment Management

Production ML pipelines need different environments for different stages:

=== "Development Environment"
    ```toml
    # Lean dependencies for production
    [tool.hatch.envs.default]
    dependencies = [
        "narwhals",
        "pandas"
    ]
    ```

=== "Testing Environment"
    ```toml
    # Full dependencies for testing
    [tool.hatch.envs.test]
    dependencies = [
        "narwhals",
        "pandas",
        "polars",
        "dask",
        "pytest",
        "hypothesis"
    ]
    ```

Why two environments?
1. Development:
   - Minimal dependencies
   - Faster builds
   - Smaller containers
   - Matches production

2. Testing:
   - All backends
   - Testing tools
   - Validation tools
   - Performance profiling

This ensures your code works everywhere while keeping production deployments lean.

**Happy coding!** 🚀
