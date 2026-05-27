# Installation and quick start

## Installation

Narwhals requires **Python 3.10+** and has no required dependencies: it only ever uses the dataframe libraries you pass into it.

Pick the workflow that matches how you manage your project's dependencies.

=== "pip"

    First, [create and activate](https://docs.python.org/3/library/venv.html) a Python 3.10+ virtual environment, then run:

    ```terminal
    python -m pip install narwhals
    ```

=== "uv"

    If you're starting a new project with [uv](https://docs.astral.sh/uv/), add Narwhals to it with:

    ```terminal
    uv add narwhals
    ```

    This will create a virtual environment (if one doesn't already exist), add `narwhals` to your `pyproject.toml`, and update `uv.lock`.

    To install Narwhals into an existing virtual environment without touching `pyproject.toml`, use:

    ```terminal
    uv pip install narwhals
    ```


=== "Poetry"

    From within a [Poetry](https://python-poetry.org/) project, run:

    ```terminal
    poetry add narwhals
    ```

    This will add `narwhals` to your `pyproject.toml` and update `poetry.lock`.

### Installing with extras

Narwhals exposes optional extras that pull in a specific backend at a version known to be compatible. These are convenience pins, not requirements: if you already have the backend installed (or want to manage its version yourself), you can skip them.

Available extras include `pandas`, `polars`, `pyarrow`, `modin`, `dask`, `duckdb`, `ibis`, `pyspark`, `pyspark-connect`, `sqlframe`, `sql`, and `cudf` (Linux only).

Specify one or more extras in square brackets, for example:


=== "pip"

    ```terminal
    python -m pip install "narwhals[polars,pyarrow]"
    ```

=== "uv"

    ```terminal
    uv add "narwhals[polars,pyarrow]"
    ```

=== "Poetry"

    ```terminal
    poetry add "narwhals[polars,pyarrow]"
    ```

### Verifying the installation

To verify the installation, start the Python REPL and execute:

```python exec="yes" source="above" session="quickstart" result="python"
import narwhals

print(narwhals.__version__)
```

If you see the version number, then the installation was successful!

## Quick start

### Optional companion libraries

Narwhals has no required dataframe dependencies, but to follow along with the examples below you'll want at least one of:

- [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [Polars](https://pola-rs.github.io/polars/user-guide/installation/)
- [PyArrow](https://arrow.apache.org/docs/python/install.html)

### Simple example

Create a Python file `t.py` with the following content:

```python exec="yes" source="above" session="quickstart" result="python"
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
    supports other dataframe libraries (See [the home page](index.md) for supported libraries).
