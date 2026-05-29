# Narwhals TPC-H queries

Utilities for running the TPC-H queries via Narwhals.

This is a [uv workspace] member (`narwhals-tpch`, import name `narwhals_tpch`) living
under [`packages/`](../). Before getting started, make sure you've followed the
instructions in [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

All commands below are run from the **repository root**. The `tpch` dependency-group
installs this package editable; `core-tests` plus the `dask`/`pandas` extras provide
the backends exercised by the test matrix.

## Generate data

Generating the data is required only the first time you want to run the queries (it is
also done automatically by the test suite if missing):

```bash
uv run --group core-tests --group tpch --extra dask --extra pandas \
    --module narwhals_tpch.generate_data --debug
```

## Run queries

* All queries for a given backend:

    ```bash
    uv run --group core-tests --group tpch --extra dask --extra pandas \
        pytest packages/narwhals-tpch/tests -k "polars"
    ```

* A given query for all the backends:

    ```bash
    uv run pytest packages/narwhals-tpch/tests -k "q1-"
    ```

    Why the extra dash? That's not a typo! `-k q1` would also match `q11`, `q12`, and
    so on. Adding a trailing dash prevents that.

* A given query for a single backend:

    ```bash
    uv run pytest packages/narwhals-tpch/tests -k "q1-pandas[pyarrow]"
    ```
