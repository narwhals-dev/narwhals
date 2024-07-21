# Related projects

## Dataframe Interchange Protocol

Standardised way of interchanging data between libraries, see
[here](https://data-apis.org/dataframe-protocol/latest/index.html).

Narwhals builds upon it by providing one level of support to libraries which implement it -
this includes Ibis and Vaex. See [levels](levels.md) for details.

## DataFrame API Standard

Now-discontinued project which aimed to "provide a standard interface that encapsulates implementation details of dataframe libraries. This will allow users and third-party libraries to write code that interacts and operates with a standard dataframe, and not with specific implementations.", see [here](https://data-apis.org/dataframe-api/draft/).

The Narwhals author was originally involved, but left due to irreconcilable differences in vision, and
the project was ultimately abandoned.

Some notable difference are:

- Narwhals just uses a subset of the Polars API, whereas the dataframe standard introduces a new API
- Narwhals supports expressions, and separates lazy and eager execution
- Narwhals is a standalone, independent project, whereas the dataframe standard aimed to be upstreamed
  and implemented by major dataframe libraries.

## Array API

Array counterpart to the DataFrame API, see [here](https://data-apis.org/array-api/2022.12/index.html).
