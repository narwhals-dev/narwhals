# Narwhals

-![](assets/image.png)

[![PyPI version](https://badge.fury.io/py/narwhals.svg)](https://badge.fury.io/py/narwhals)
[![Downloads](https://static.pepy.tech/badge/narwhals/month)](https://pepy.tech/project/narwhals)
[![Trusted publishing](https://img.shields.io/badge/Trusted_publishing-Provides_attestations-bright_green)](https://peps.python.org/pep-0740/)

Extremely lightweight and extensible compatibility layer between dataframe libraries!

- **Full API support**: cuDF, Modin, pandas, Polars, PyArrow
- **Lazy-only support**: Dask
- **Interchange-level support**: DuckDB, Ibis, Vaex, anything which implements the DataFrame Interchange Protocol

Seamlessly support all, without depending on any!

- ✅ **Just use** [a subset of **the Polars API**](./api-reference/index.md), no need to learn anything new
- ✅ **Zero dependencies**, Narwhals only uses what
  the user passes in so your library can stay lightweight
- ✅ Separate **lazy** and eager APIs, use **expressions**
- ✅ Support pandas' complicated type system and index, without
  either getting in the way
- ✅ **100% branch coverage**, tested against pandas and Polars nightly builds
- ✅ **Negligible overhead**, see [overhead](./overhead.md/)
- ✅ Let your IDE help you thanks to **full static typing**, see [`narwhals.typing`][]
- ✅ **Perfect backwards compatibility policy**,
  see [stable api](./backcompat.md/) for how to opt-in

## Who's this for?

Anyone wishing to write a library/application/service which consumes dataframes, and wishing to make it
completely dataframe-agnostic.

Let's get started!

## Roadmap

See [roadmap discussion on GitHub](https://github.com/narwhals-dev/narwhals/discussions/1370)
for an up-to-date plan of future work.
