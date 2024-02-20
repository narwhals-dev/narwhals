# narwhals

Extremely lightweight compatibility layer between Polars, pandas, cuDF, and Modin.

Seamlessly support all four, without depending on any of them!

- ✅ Just use a subset of the Polars API, no need to learn anything new
- ✅ **No dependencies** (not even Polars), keep your library lightweight
- ✅ Separate Lazy and Eager APIs
- ✅ Use the Polars Expressions API

**Note: this is work-in-progress, and a bit of an experiment, don't take it too seriously**.

## Installation

```
pip install narwhals
```
Or just vendor it, it's only a bunch of pure-Python files.

## Usage

There are three steps to writing dataframe-agnostic code using narwhals:

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

from narwhals import to_polars_api, to_original_object

AnyDataFrame = TypeVar("AnyDataFrame")


def my_agnostic_function(
    suppliers_native: AnyDataFrame,
    parts_native: AnyDataFrame,
) -> AnyDataFrame:
    suppliers, pl = to_polars_api(suppliers_native, version="0.20")
    parts, _ = to_polars_api(parts_native, version="0.20")
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
    return to_original_object(result.collect())
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
    s   p  weight_mean  weight_max
0  S1  P6         19.0        19.0
1  S2  P2         17.0        17.0
2  S3  P2         17.0        17.0
3  S4  P6         19.0        19.0

Polars output:
shape: (4, 4)
┌─────┬─────┬─────────────┬────────────┐
│ s   ┆ p   ┆ weight_mean ┆ weight_max │
│ --- ┆ --- ┆ ---         ┆ ---        │
│ str ┆ str ┆ f64         ┆ f64        │
╞═════╪═════╪═════════════╪════════════╡
│ S1  ┆ P6  ┆ 19.0        ┆ 19.0       │
│ S3  ┆ P2  ┆ 17.0        ┆ 17.0       │
│ S4  ┆ P6  ┆ 19.0        ┆ 19.0       │
│ S2  ┆ P2  ┆ 17.0        ┆ 17.0       │
└─────┴─────┴─────────────┴────────────┘
```
Magic! 🪄 

## Scope

If you maintain a dataframe-consuming library, then any function from the Polars API which you'd
like to be able to use is in-scope, so long as it can be supported without too much difficulty
for at least pandas, cuDF, and Modin.

Feature requests are more than welcome!

## Related Projects

- This is not Ibis. narwhals lets each backend do its own optimisations, and only provides
  a lightweight (~30 kilobytes) compatibility layer with the Polars API.
  Ibis applies its own optimisations to different backends, is a heavyweight
  dependency (~400 MB), and defines its own API.

- This is not intended as a DataFrame Standard. See the Consortium for Python Data API Standards
  for a more general and more ambitious project. Please only consider using narwhals if you only
  need to support Polars and pandas-like dataframes, and specifically want to tap into Polars'
  lazy and expressions features (which are out of scope for the Consortium's Standard).

## Why "Narwhals"?

Because they are so awesome.
