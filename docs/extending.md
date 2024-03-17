# How Narwhals can support your dataframe as well!

Currently, Narwhals recognises the following libraries as inputs:

- pandas
- Polars
- cuDF
- Modin

If you want your own library to be recognised too, you can either:

- open a PR (with tests)
- or, make sure that:

  - your Dataframe class contains a `__narwhals_dataframe__` method which,
    if called, returns a class with the same public methods as a Narwhals
    Dataframe
  - your Lazyframe class (if present) contains a `__narwhals_lazyframe__` method which,
    if called, returns a class with the same public methods as a Narwhals
    Lazyframe.
  - your Series class contains a `__narwhals_series__` method which,
    if called, returns a class with the same public methods as a Narwhals
    Series.

This is work-in-progress so don't expect it to work just yet, this is just the idea.
