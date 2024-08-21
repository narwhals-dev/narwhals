# Roadmap

Priorities, as of August 2024, are:

- Works towards supporting projects which have shown interest in Narwhals.
- Implement when/then/otherwise so that Narwhals is API-complete enough to complete all the TPC-H queries.
- Make Dask support complete-enough, at least to the point that it can execute TPC-H queries.
- Improve support for cuDF, which we can't currently test in CI (unless NVIDIA helps us out :wink:) but
  which we can and do test manually in Kaggle notebooks.
- Add extra docs and tutorials to make the project more accessible and easy to get started with.
- Look into extra backends, such as DuckDB and Ibis.
