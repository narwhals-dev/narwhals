# Related projects

## Dataframe Interchange Protocol

Standardised way of interchanging data between libraries, see
[here](https://data-apis.org/dataframe-protocol/latest/index.html).

## DataFrame API Standard

Project which aims to "provide a standard interface that encapsulates implementation details of dataframe libraries. This will allow users and third-party libraries to write code that interacts and operates with a standard dataframe, and not with specific implementations.", see [here](https://data-apis.org/dataframe-api/draft/).

Some notable difference are:

- Narwhals just uses a subset of the Polars API, whereas the dataframe standard introduces a new API
- Narwhals supports expressions and separates lazy and eager execution
- Narwhals is a standalone, independent project, whereas the dataframe standard aims to be upstreamed
  and implemented by major dataframe libraries.

The projects are not in competition and have different goals.

## Ibis

[Presents itself as a dataframe standard](https://voltrondata.com/resources/open-source-standards), and
dispatches to 20+ backends. Some differences with Narwhals are:

- Narwhals is ~1000 times lighter
- Narwhals only supports 4 backends, Ibis more than 20
- Narwhals is limited to fundamental dataframe operations, Ibis includes more advanced and niche ones.

Again, the projects are not in competition and have different goals.

## Array API

Array counterpart to the DataFrame API, see [here](https://data-apis.org/array-api/2022.12/index.html).
