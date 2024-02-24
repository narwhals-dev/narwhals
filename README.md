# Narwhals

Extremely lightweight compatibility layer between Polars, pandas, cuDF, and Modin.

Seamlessly support all four, without depending on any of them!

- ✅ **Just use** a subset of **the Polars API**, no need to learn anything new
- ✅ **No dependencies** (not even Polars), keep your library lightweight
- ✅ Separate **Lazy** and Eager APIs
- ✅ Use Polars **Expressions**

**Note: this is work-in-progress, and a bit of an experiment, don't take it too seriously**.

## Installation

```
pip install narwhals
```
Or just vendor it, it's only a bunch of pure-Python files.

## Usage

There are three steps to writing dataframe-agnostic code using Narwhals:

1. use `narwhals.to_polars_api` to wrap a pandas, Polars, cuDF, or Modin dataframe
   in the Polars API
2. use the subset of the Polars API defined in https://github.com/MarcoGorelli/narwhals/blob/main/narwhals/spec/__init__.py.
3. use `narwhals.to_original_object` to return an object to the user in their original
   dataframe flavour. For example:

   - if you started with pandas, you'll get pandas back
   - if you started with Polars, you'll get Polars back
   - if you started with Modin, you'll get Modin back
   - if you started with cuDF, you'll get cuDF back (and computation will happen natively on the GPU!)
   
## Example

Here's an example of a dataframe agnostic function:

```python
from typing import TypeVar
import pandas as pd
import polars as pl

from narwhals import translate_frame

AnyDataFrame = TypeVar("AnyDataFrame")


def my_agnostic_function(
    suppliers_native: AnyDataFrame,
    parts_native: AnyDataFrame,
) -> AnyDataFrame:
    suppliers, pl = translate_frame(suppliers_native, lazy_only=True)
    parts, _ = translate_frame(parts_native, lazy_only=True)
    result = (
        suppliers.join(parts, left_on="city", right_on="city")
        .filter(
            pl.col("color").is_in(["Red", "Green"]),
            pl.col("weight") > 14,
        )
        .group_by("s", "p")
        .agg(
            weight_mean=pl.col("weight").mean(),
            weight_max=pl.col("weight").max(),
        )
    )
    return result.collect().to_native()
```
You can pass in a pandas, Polars, cuDF, or Modin dataframe, the output will be the same!
Let's try it out:

```python
suppliers = {
    "s": ["S1", "S2", "S3", "S4", "S5"],
    "sname": ["Smith", "Jones", "Blake", "Clark", "Adams"],
    "status": [20, 10, 30, 20, 30],
    "city": ["London", "Paris", "Paris", "London", "Athens"],
}
parts = {
    "p": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "pname": ["Nut", "Bolt", "Screw", "Screw", "Cam", "Cog"],
    "color": ["Red", "Green", "Blue", "Red", "Blue", "Red"],
    "weight": [12.0, 17.0, 17.0, 14.0, 12.0, 19.0],
    "city": ["London", "Paris", "Oslo", "London", "Paris", "London"],
}

print("pandas output:")
print(
    my_agnostic_function(
        pd.DataFrame(suppliers),
        pd.DataFrame(parts),
    )
)
print("\nPolars output:")
print(
    my_agnostic_function(
        pl.LazyFrame(suppliers),
        pl.LazyFrame(parts),
    )
)
```

```
pandas output:
    s   p  weight_mean
0  S1  P6         19.0
1  S2  P2         17.0
2  S3  P2         17.0
3  S4  P6         19.0

Polars output:
shape: (4, 3)
┌─────┬─────┬─────────────┐
│ s   ┆ p   ┆ weight_mean │
│ --- ┆ --- ┆ ---         │
│ str ┆ str ┆ f64         │
╞═════╪═════╪═════════════╡
│ S1  ┆ P6  ┆ 19.0        │
│ S3  ┆ P2  ┆ 17.0        │
│ S4  ┆ P6  ┆ 19.0        │
│ S2  ┆ P2  ┆ 17.0        │
└─────┴─────┴─────────────┘
```
Magic! 🪄 

## Scope

- Do you maintain a dataframe-consuming library?
- Is there a Polars function which you'd like Narwhals to have, which would make your job easier?

If, I'd love to hear from you!

**Note**: this is **not** a "Dataframe Standard" project. It just translates a subset of the Polars
API to pandas-like libraries.

## Why "Narwhals"?

Because they are so awesome.
