# Complete example

We're going to write a dataframe-agnostic "Standard Scaler". This class will have
`fit` and `transform` methods (like `scikit-learn` transformers), and will work
agnostically for pandas and Polars.

We'll need to write two methods:

- `fit`: find the mean and standard deviation for each column from a given training set;
- `transform`: scale a given dataset with the mean and standard deviations calculated
  during `fit`.

The `fit` method is a bit complicated, so let's start with `transform`.
Suppose we've already calculated the mean and standard deviation of each column, and have
stored them in attributes `self.means` and `self.std_devs`.

## Transform method

We're going to take in a dataframe, and return a dataframe of the same type.
Therefore, we use `@nw.narwhalify_method` (the counterpart to `@nw.narwhalify` which is
meant to be used for methods):

```python
import narwhals as nw

class StandardScaler:
    @nw.narwhalify_method
    def transform(self, df):
        return df.with_columns(
            (nw.col(col) - self._means[col]) / self._std_devs[col]
            for col in df.columns
        )
```

Note that all the calculations here can stay lazy if the underlying library permits it,
so we don't pass in any extra keyword-arguments such as `eager_only`, we just use the
default `eager_only=False`.

## Fit method

Unlike the `transform` method, `fit` cannot stay lazy, as we need to compute concrete values
for the means and standard deviations.

To be able to get `Series` out of our `DataFrame`, we'll pass `eager_only=True` to `nw.from_native`.
This is because Polars doesn't have a concept of lazy `Series`, and so Narwhals
doesn't either.

Note how here, we're not returning a dataframe to the user - we just take a dataframe in, and
store some internal state. Therefore, we use `nw.from_native` explicitly, as opposed to using the
utility `@nw.narwhalify_method` decorator.

```python
import narwhals as nw

class StandardScaler:
    def fit(self, df_any):
        df = nw.from_native(df_any, eager_only=True)
        self._means = {col: df[col].mean() for col in df.columns}
        self._std_devs = {col: df[col].std() for col in df.columns}
```

## Putting it all together

Here is our dataframe-agnostic standard scaler:
```python exec="1" source="above" session="tute-ex1"
import narwhals as nw

class StandardScaler:
    def fit(self, df_any):
        df = nw.from_native(df_any, eager_only=True)
        self._means = {col: df[col].mean() for col in df.columns}
        self._std_devs = {col: df[col].std() for col in df.columns}

    @nw.narwhalify_method
    def transform(self, df):
        return df.with_columns(
            (nw.col(col) - self._means[col]) / self._std_devs[col]
            for col in df.columns
        )
```

Next, let's try running it. Notice how, as `transform` doesn't use
any eager-only features, so we can pass a Polars LazyFrame to it and have it
stay lazy!

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="tute-ex1"
    import pandas as pd

    df_train = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 7]})
    df_test = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 7]})
    scaler = StandardScaler()
    scaler.fit(df_train)
    print(scaler.transform(df_test))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="tute-ex1"
    import polars as pl

    df_train = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 7]})
    df_test = pl.LazyFrame({'a': [1, 2, 3], 'b': [4, 5, 7]})
    scaler = StandardScaler()
    scaler.fit(df_train)
    print(scaler.transform(df_test).collect())
    ```
