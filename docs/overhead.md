# Overhead

Narwhals converts Polars syntax to non-Polars dataframes.

So, what's the overhead of running pandas vs pandas via Narwhals?

Based on experiments we've done, the answer is: it's negligible. Here
are timings from the TPC-H queries, comparing running pandas directly
vs running pandas via Narwhals:

![Comparison of pandas vs "pandas via Narwhals" timings on TPC-H queries showing neglibile overhead](https://github.com/narwhals-dev/narwhals/assets/33491632/71029c26-4121-43bb-90fb-5ac1c16ab8a2)

[Here](https://www.kaggle.com/code/marcogorelli/narwhals-tpc-h-results-s-2-w-native)'s the code to
reproduce the plot above, check the input
sources for notebooks which run each individual query, along with
the data sources.

On some runs, the Narwhals code makes things marginally faster, on others
marginally slower. The overall picture is clear: with Narwhals, you
can support both Polars and pandas APIs with little to no impact on either.
