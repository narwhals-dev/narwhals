# Narwhals TPC-H queries

Utilities for running the TPC-H queries via Narwhals.

Before getting started, make sure you've followed the instructions in
`CONTRIBUTING.MD`.

## Generate data

Run `python generate_data.py` from this folder.

## Run queries

* To run all the queries for a given backend:

    ```terminal
    pytest -k "polars"
    ```

* To run a given query for all the backends:

    ```terminal
    pytest -k "q1-"
    ```

    Why there is an extra dash? That's not a typo! `-k q1` will match also `q11`, `q12`, and so on.
    Adding a dash at the end will prevent that.

* To run a given query for all the backends:
    
    ```terminal
    pytest -k "q1-pandas[pyarrow]"
    ```
