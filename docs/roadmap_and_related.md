# Roadmap and related projects

## Roadmap

See [roadmap discussion on GitHub](https://github.com/narwhals-dev/narwhals/discussions/1370)
for an up-to-date plan of future work.

## Related projects

### Dataframe Interchange Protocol

Standardised way of interchanging data between libraries, see
[here](https://data-apis.org/dataframe-protocol/latest/index.html).

Narwhals builds upon it by providing one level of support to libraries which implement it -
this includes Ibis and Vaex. See [extending](extending.md) for details.

### Array API

Array counterpart to the DataFrame API, see [here](https://data-apis.org/array-api/2022.12/index.html).

### PyCapsule Interface

Allows C extension modules to safely share pointers to C data structures with Python code and other C modules, encapsulating the pointer with a name and optional destructor to manage resources and ensure safe access, see [here](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html) for details. 

Narwhals supports exporting a DataFrame via the Arrow PyCapsule Interface.

### Ibis

Pitched as "The portable Dataframe library", Ibis provides a Pythonic frontend
to various SQL (as well as Polars LazyFrame) engines. Some differences with Narwhals are:

- Narwhals' main use case is for library maintainers wanting to support
  different dataframe libraries without depending on any whilst keeping
  things as lightweight as possible. Ibis is more targeted at end users
  and aims to be thought of as a Dataframe library akin to
  pandas / Polars / etc.
- Narwhals allows you to write a "Dataframe X in, Dataframe X out" function.
  Ibis allows materialising to pandas, Polars (eager), and PyArrow, but has
  no way to get back to the input type exactly (e.g. there's no way to
  start with a Polars LazyFrame and get back a Polars LazyFrame)
- Narwhals respects input data types as much as possible, Ibis doesn't
  support Categorical (nor does it distinguish between fixed-size-list and
  list)
- Narwhals separates between lazy and eager APIs, with the eager API
  provide very fine control over dataframe operations (slicing rows and
  columns, iterating over rows, getting values out of the dataframe as
  Python scalars). Ibis is more focused on lazy execution
- Ibis supports SQL engines (and can translate to SQL),
  Narwhals is more focused traditional dataframes where row-order is defined
  (although we are brainstorming a lazy-only level of support)
- Narwhals is extremely lightweight and comes with zero required dependencies,
  Ibis requires pandas and PyArrow for all backends
- Narwhals supports Dask, whereas Ibis has deprecated support for it

Although people often ask about the two tools, we consider them to be
very different and not in competition. Further efforts to clarify the
distinction are welcome 🙏!

