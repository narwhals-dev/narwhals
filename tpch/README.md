# Narwhals TPC-H queries

Utilities for running the TPC-H queries via Narwhals.

Before getting started, make sure you've followed the instructions in [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Prerequisites

The instructions below assume that your current working directory is the `tpch` folder
within the narwhals repository.

How to get here:

1. Clone the repo: `git clone git@github.com:YOUR-GITHUB-USERNAME/narwhals.git`
2. Move into the folder: `cd tpch`

## Generate data

To generate the data required to run TPCH queries you will need to run:

```bash
python generate_data.py
```

Note that this action is required only the first time you want to run the TPCH queries.

## Run queries

* To run all the queries for a given backend:

    ```bash
    pytest -k "polars"
    ```

* To run a given query for all the backends:

    ```bash
    pytest -k "q1-"
    ```

    Why there is an extra dash? That's not a typo! `-k q1` will match also `q11`, `q12`, and so on.
    Adding a dash at the end will prevent that.

* To run a given query for all the backends:
    
    ```bash
    pytest -k "q1-pandas[pyarrow]"
    ```
