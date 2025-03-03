# Installation and quick start

## Installation

=== "UV"

    First, ensure you have installed [UV](https://github.com/astral-sh/uv), and make sure you have [created and activated](https://docs.astral.sh/uv/pip/environments/#python-environments) a Python 3.8+ virtual environment.

    If you haven't, you can follow our [_setting up your environment_](https://github.com/narwhals-dev/narwhals/blob/main/CONTRIBUTING.md#option-1-use-uv-recommended) guide.
    Then, run:

    ```terminal
    uv pip install narwhals
    ```

=== "Python's venv"

    First, ensure you have [created and activated](https://docs.python.org/3/library/venv.html) a Python 3.8+ virtual environment.

    Then, run:

    ```terminal
    python -m pip install narwhals
    ```

### Verifying the Installation

To verify the installation, start the Python REPL and execute:

```python
>>> import narwhals
>>> narwhals.__version__
'1.29.0'
```

If you see the version number, then the installation was successful!

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


def agnostic_get_columns(df_native: IntoFrame) -> list[str]:
    df = nw.from_native(df_native)
    column_names = df.columns
    return column_names


data = {"a": [1, 2, 3], "b": [4, 5, 6]}
df_pandas = pd.DataFrame(data)
df_polars = pl.DataFrame(data)
table_pa = pa.table(data)

print("pandas output")
print(agnostic_get_columns(df_pandas))

print("Polars output")
print(agnostic_get_columns(df_polars))

print("PyArrow output")
print(agnostic_get_columns(table_pa))
```

If you run `python t.py` then your output should look like the above.
This is the simplest possible example of a dataframe-agnostic function - as we'll soon
see, we can do much more advanced things.

Let's learn about what you just did, and what Narwhals can do for you!

!!! info

    These examples are using pandas, Polars, and PyArrow, however Narwhals
    supports other dataframe libraries (See [supported libraries](extending.md)).
