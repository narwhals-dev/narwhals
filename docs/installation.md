# Installation and quick start

## Installation

Narwhals requires **Python 3.10+** and has no required dependencies: it only ever uses the dataframe libraries you pass into it.

Pick the workflow that matches how you manage your project's dependencies.

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

=== "pip"

    First, [create and activate](https://docs.python.org/3/library/venv.html) a Python 3.10+ virtual environment, then run:

    ```terminal
    python -m pip install narwhals
    ```

    Unlike `uv add` or `poetry add`, `pip install` does not touch `pyproject.toml`. If you're working on a project, you'll need to record the dependency there yourself.

=== "Poetry"

    From within a [Poetry](https://python-poetry.org/) project, run:

    ```terminal
    poetry add narwhals
    ```

    This will add `narwhals` to your `pyproject.toml` and update `poetry.lock`.

### Installing with extras

Narwhals exposes optional extras that pull in a specific backend at a version known to be compatible. These are convenience pins, not requirements: if you already have the backend installed (or want to manage its version yourself), you can skip them.

Available extras include `pandas`, `polars`, `pyarrow`, `modin`, `dask`, `duckdb`, `ibis`, `pyspark`, `pyspark-connect`, `sqlframe`, `sql`, and `cudf` (Linux only). For the authoritative list, see `[project.optional-dependencies]` in [`pyproject.toml`](https://github.com/narwhals-dev/narwhals/blob/main/pyproject.toml).

Specify one or more extras in square brackets, for example:


=== "uv"

    ```terminal
    uv add "narwhals[polars,pyarrow]"
    ```

=== "pip"

    ```terminal
    python -m pip install "narwhals[polars,pyarrow]"
    ```

=== "Poetry"

    ```terminal
    poetry add "narwhals[polars,pyarrow]"
    ```

!!! warning "Library authors: prefer a dev dependency group"

    `uv add "narwhals[polars,pyarrow]"` adds the extras to `[project.dependencies]`.

    If you publish your project, **every consumer is then forced to install polars and
    pyarrow**, which defeats Narwhals' "support all, depend on none" design.

    If you're building a library (rather than an application), keep `narwhals` as your
    runtime dependency and pin the backends only for development:

    === "uv"

        ```terminal
        uv add narwhals
        uv add --dev "narwhals[polars,pyarrow]"
        ```

    === "Poetry"

        ```terminal
        poetry add narwhals
        poetry add --group dev "narwhals[polars,pyarrow]"
        ```

    Dependency groups are not shipped in your distribution metadata, so consumers
    receive only `narwhals` and bring their own backend.

    Alternatively, re-expose the backends as your library's **own optional extras**, so
    consumers opt in explicitly (e.g. `pip install your-library[polars]`):

    ```toml
    [project.optional-dependencies]
    polars = ["narwhals[polars]"]
    pyarrow = ["narwhals[pyarrow]"]
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

!!! tip "Running with `uv` (no install needed)"

    If you have [uv](https://docs.astral.sh/uv/), you can skip the environment setup entirely by declaring [inline script dependencies](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies) at the top of `t.py`:

    ```python
    # /// script
    # dependencies = ["narwhals[pandas,polars,pyarrow]"]
    # ///
    ```

    Then run it with `uv run t.py`.

This is the simplest possible example of a dataframe-agnostic function - as we'll soon
see, we can do much more advanced things.

Let's learn about what you just did, and what Narwhals can do for you!

!!! info

    These examples are using pandas, Polars, and PyArrow, however Narwhals
    supports other dataframe libraries (See [the home page](index.md) for supported libraries).
