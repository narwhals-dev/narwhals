# Overhead

Narwhals converts Polars syntax to non-Polars dataframes.

So, what's the overhead of running "pandas" vs "pandas via Narwhals"?

Based on experiments we've done, the answer is: it's negligible.
Sometimes it's even negative, because of how careful we are in Narwhals
to avoid unnecessary copies and index resets. Here are timings from the
TPC-H queries, comparing running pandas directly vs running pandas via Narwhals:

![Comparison of pandas vs "pandas via Narwhals" timings on TPC-H queries showing neglibile overhead](https://github.com/user-attachments/assets/bbd6fcaf-5c25-46a6-8c03-9ce42efca787)

[Complete code to reproduce](https://www.kaggle.com/code/marcogorelli/narwhals-vs-pandas-overhead-tpc-h-s2).

## Plotly's story

One big difference between Plotly v5 and Plotly v6 is the handling of non-pandas inputs:

- In v5, Plotly would convert non-pandas inputs to pandas.
- In v6, Plotly operates on non-pandas inputs natively (via Narwhals).

We expected that this would bring a noticeable performance benefit for non-pandas inputs,
but that there may be some slight overhead for pandas.

Instead, we observed that things got noticeably faster for both non-pandas inputs and for
pandas ones!

- Polars plots got 3x, and sometimes even more than 10x, faster.
- pandas plots were typically no slower, but sometimes ~20% faster.

Full details on [Plotly's write-up](https://plotly.com/blog/chart-smarter-not-harder-universal-dataframe-support/).

## Overhead for DuckDB, PySpark, and other lazy backends

For lazy backends, Narwhals respects the backends' laziness and always keeps
everything lazy. Narwhals never evaluates a full query unless you ask it to
(with `.collect()`).

In order to mimic Polars' behaviour, there are some places
where Narwhals does need to inspect dataframes' schemas. This is typically
cheap, as it does not require reading a full dataset into memory and can often just
be done from metadata alone. However, it's not completely free, especially if your
data lives on the cloud. To minimise the overhead, when Narwhals needs to evaluate
schemas or column names, it makes sure to cache them.
