# Installation and quick start

## Installation

First, make sure you have [created and activated](https://docs.python.org/3/library/venv.html) a Python3.8+ virtual environment.

Then, run
```console
python -m pip install narwhals
```

Then, if you start the Python REPL and see the following:
```python
>>> import narwhals
>>> narwhals.__version__
'1.9.3'
```
then installation worked correctly!

## Quick start

### Prerequisites

Please start by following the [installation instructions](installation.md).

To follow along with the examples which follow, please install the following (though note that
they are not required dependencies - Narwhals only ever uses what the user passes in):

- [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [Polars](https://pola-rs.github.io/polars/user-guide/installation/)
- [PyArrow](https://arrow.apache.org/docs/python/install.html)

### Simple example

Create a Python file `t.py` with the following content:

```python exec="1" source="above" session="quickstart" result="python"
from __future__ import annotations

import pandas as pd
import polars as pl
import pyarrow as pa
import narwhals as nw
from narwhals.typing import IntoFrame


def my_function(df_native: IntoFrame) -> list[str]:
    df = nw.from_native(df_native)
    column_names = df.columns
    return column_names


data = {"a": [1, 2, 3], "b": [4, 5, 6]}
df_pandas = pd.DataFrame(data)
df_polars = pl.DataFrame(data)
table_pa = pa.table(data)

print("pandas output")
print(my_function(df_pandas))

print("Polars output")
print(my_function(df_polars))

print("PyArrow output")
print(my_function(table_pa))
```

If you run `python t.py` then your output should look like the above. This is the simplest possible example of a dataframe-agnostic
function - as we'll soon see, we can do much more advanced things.
Let's learn about what you just did, and what Narwhals can do for you!

Note: these examples are only using pandas and Polars. Please see the following to find the [supported libraries](extending.md).
