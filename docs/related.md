# Related projects

## Dataframe Interchange Protocol

Standardised way of interchanging data between libraries, see
[here](https://data-apis.org/dataframe-protocol/latest/index.html).

## DataFrame API Standard

Now-discontinued project which aimed to "provide a standard interface that encapsulates implementation details of dataframe libraries. This will allow users and third-party libraries to write code that interacts and operates with a standard dataframe, and not with specific implementations.", see [here](https://data-apis.org/dataframe-api/draft/).

The Narwhals author was originally involved, but left due to irreconcilable differences in vision.

Some notable difference are:

- Narwhals just uses a subset of the Polars API, whereas the dataframe standard introduces a new API
- Narwhals supports expressions, and separates lazy and eager execution
- Narwhals is a standalone, independent project, whereas the dataframe standard aims to be upstreamed
  and implemented by major dataframe libraries.

## Ibis

[Presents itself as a dataframe standard](https://voltrondata.com/resources/open-source-standards), and
dispatches to 20+ backends. Some differences with Narwhals are:

- Narwhals is aimed at library maintainers, Ibis more to end users
- Narwhals has zero required dependencies, whereas Ibis requires pandas and PyArrow. For users starting
  from non-pandas environments, the difference in the relative size increase is ~1000x
- Narwhals only supports 4 backends, Ibis more than 20
- Narwhals is focused on fundamental dataframe operations, Ibis on SQL backends
- Narwhals has negligible overhead for dataframe backends, while
  [Ibis' overhead can be significant](https://github.com/ibis-project/ibis/issues/9345)

The projects are not in competition and have different goals.

## Array API

Array counterpart to the DataFrame API, see [here](https://data-apis.org/array-api/2022.12/index.html).
