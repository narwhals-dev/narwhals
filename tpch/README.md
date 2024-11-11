# Narwhals TPC-H queries

Utilities for running the TPC-H queries via Narwhals.

Before getting started, make sure you've followed the instructions in
`CONTRIBUTING.MD`.

## Generate data

Run `python generate_data.py` from this folder.

## Run queries

To run Q1, you can run `python -m execute.main q1` from this folder.

Please add query definitions in `queries`, and scripts to execute them
in `execute` (see `queries/q1.py` and `execute/q1.py` for examples).
