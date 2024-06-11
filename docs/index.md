# Narwhals

![](assets/image.png)

Extremely lightweight compatibility layer between Polars, pandas, and more.

Seamlessly support both, without depending on either!

- ✅ **Just use** a subset of **the Polars API**, no need to learn anything new
- ✅ **Zero dependencies**, **zero 3rd-party imports**: Narwhals only uses what
  the user passes in, so you can keep your library lightweight
- ✅ Separate **lazy** and eager APIs, use **expressions**
- ✅ Support pandas' complicated type system and index, without
  either getting in the way
- ✅ **100% branch coverage**, tested against pandas and Polars nightly builds
- ✅ **Negligible overhead**, see [overhead](https://narwhals-dev.github.io/narwhals/overhead/)

## Who's this for?

Anyone wishing to write a library/application/service which consumes dataframes, and wishing to make it
completely dataframe-agnostic.

Let's get started!
