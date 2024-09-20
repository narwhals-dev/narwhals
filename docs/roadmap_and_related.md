# Roadmap and related projects

## Roadmap

Priorities, as of September 2024, are:

- Works towards supporting projects which have shown interest in Narwhals.
- Add extra docs and tutorials to make the project more accessible and easy to get started with.
- Improve support for cuDF, which we can't currently test in CI (unless NVIDIA helps us out :wink:) but
  which we can and do test manually in Kaggle notebooks.
- Define a lazy-only layer of support which can include DuckDB, Ibis, and PySpark.

## Related projects

### Dataframe Interchange Protocol

Standardised way of interchanging data between libraries, see
[here](https://data-apis.org/dataframe-protocol/latest/index.html).

Narwhals builds upon it by providing one level of support to libraries which implement it -
this includes Ibis and Vaex. See [levels](levels.md) for details.

### Array API

Array counterpart to the DataFrame API, see [here](https://data-apis.org/array-api/2022.12/index.html).
