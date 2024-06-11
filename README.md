# Narwhals

<h1 align="center">
	<img
		width="400"
		alt="narwhals_small"
		src="https://github.com/narwhals-dev/narwhals/assets/33491632/26be901e-5383-49f2-9fbd-5c97b7696f27">
</h1>

[![PyPI version](https://badge.fury.io/py/narwhals.svg)](https://badge.fury.io/py/narwhals)

Extremely lightweight and extensible compatibility layer between Polars, pandas, Modin, and cuDF (and more!).

- [Read the documentation](https://narwhals-dev.github.io/narwhals/)
- [Chat with us on Discord!](https://discord.gg/V3PqtB4VA4)

Seamlessly support all, without depending on any!

- ✅ **Just use** a subset of **the Polars API**, no need to learn anything new
- ✅ **Zero dependencies**, **zero 3rd-party imports**: Narwhals only uses what
  the user passes in, so you can keep your library lightweight
- ✅ Separate **lazy** and eager APIs, use **expressions**
- ✅ Support pandas' complicated type system and index, without
  either getting in the way
- ✅ **100% branch coverage**, tested against pandas and Polars nightly builds
- ✅ **Negligible overhead**, see [overhead](https://narwhals-dev.github.io/narwhals/overhead/)

## Used by

Join the party!

- [scikit-lego](https://github.com/koaning/scikit-lego)
- [scikit-playtime](https://github.com/koaning/scikit-playtime)
- [timebasedcv](https://github.com/FBruzzesi/timebasedcv)

## Installation

- pip (recommended, as it's the most up-to-date)
  ```
  pip install narwhals
  ```
- conda-forge (also fine, but the latest version may take longer to appear)
  ```
  conda install -c conda-forge narwhals
  ```

## Usage

There are three steps to writing dataframe-agnostic code using Narwhals:

1. use `narwhals.from_native` to wrap a pandas/Polars/Modin/cuDF
   DataFrame/LazyFrame in a Narwhals class
2. use the [subset of the Polars API supported by Narwhals](https://narwhals-dev.github.io/narwhals/api-reference/)
3. use `narwhals.to_native` to return an object to the user in its original
   dataframe flavour. For example:

   - if you started with pandas, you'll get pandas back
   - if you started with Polars, you'll get Polars back
   - if you started with Modin, you'll get Modin back (and compute will be distributed)
   - if you started with cuDF, you'll get cuDF back (and compute will happen on GPU)

## What about Ibis?

Like Ibis, Narwhals aims to enable dataframe-agnostic code. However, Narwhals comes with **zero** dependencies,
is about as lightweight as it gets, and is aimed at library developers rather than at end users. It also does
not aim to support as many backends, instead preferring to focus on dataframes. So, which should you use?

- If you need a SQL frontend in Python: Ibis!
- If you're a library maintainer and want a lightweight and minimal-overhead layer to get cross-dataframe library support: Narwhals!

Here is the package size increase which would result from installing each tool in a non-pandas
environment:

![image](https://github.com/MarcoGorelli/narwhals/assets/33491632/a8dfba78-feb1-48c1-960a-5b9b03585fa5)

## Example

See the [tutorial](https://narwhals-dev.github.io/narwhals/basics/dataframe/) for several examples!

## Scope

- Do you maintain a dataframe-consuming library?
- Do you have a specific Polars function in mind that you would like Narwhals to have in order to make your work easier?

If you said yes to both, we'd love to hear from you!

**Note**: You might suspect that this is a secret ploy to infiltrate the Polars API everywhere.
Indeed, you may suspect that.

## Why "Narwhals"?

[Coz they are so awesome](https://youtu.be/ykwqXuMPsoc?si=A-i8LdR38teYsos4).

Thanks to [Olha Urdeichuk](https://www.fiverr.com/olhaurdeichuk) for the illustration!
