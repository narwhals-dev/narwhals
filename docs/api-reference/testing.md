# `narwhals.testing`

## Constructors

The pytest fixtures in `narwhals.testing.constructors` allow you to write tests that automatically run across all supported dataframe backends (pandas, polars, dask, etc.) that are installed.

::: narwhals.testing.constructors
    handler: python
    options:
      members:
        - eager_constructor
        - frame_constructor
        - lazy_constructor
