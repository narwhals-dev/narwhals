# Complete example

We're going to write a dataframe-agnostic "Standard Scaler". This class will have
`fit` and `transform` methods (like `scikit-learn` transformers), and will work
agnosstically for pandas and Polars.

We'll need to write two methods:

- `fit`: find the mean and standard deviation for each column from a given training set;
- `transform`: scale a given dataset with the mean and standard deviations calculated
  during `fit`.

The `fit` method is a bit complicated, so let's start with `transform`.
Suppose we've already calculated the mean and standard deviation of each column, and have
stored them in attributes `self.means` and `self.std_devs`.

## Transform method

The general strategy will be:

1. Initialise a Narwhals DataFrame by passing your dataframe to `nw.DataFrame`.
2. Express your logic using the subset of the Polars API supported by Narwhals.
3. If you need to return a dataframe to the user in its original library, call `narwhals.to_native`.

```python
import narwhals as nw

class StandardScalar:
    def transform(self, df):
        df = nw.DataFrame(df)
        df = df.with_columns(
            (nw.col(col) - self._means[col]) / self._std_devs[col]
            for col in df.columns
        )
        return nw.to_native(df)
```

Note that all the calculations here can stay lazy if the underlying library permits it.
For Polars, the return value is a `polars.LazyFrame` - it is the caller's responsibility to
call `.collect()` on the result if they want to materialise its values.

## Fit method

Unlike the `transform` method, `fit` cannot stay lazy, as we need to compute concrete values
for the means and standard deviations.

To be able to get `Series` out of our `DataFrame`, we'll need to use `narwhals.DataFrame` (as opposed to
`narwhals.LazyFrame`), as Polars doesn't have a concept of lazy `Series`.

```python
import narwhals as nw

class StandardScalar:
    def fit(self, df):
        df = nw.DataFrame(df)
        self._means = {df[col].mean() for col in df.columns}
        self._std_devs = {df[col].std() for col in df.columns}
```

## Putting it all together

Here is our dataframe-agnostic standard scaler:
```python exec="1" source="above" session="tute-ex1"
import narwhals as nw

class StandardScaler:
    def fit(self, df):
        df = nw.DataFrame(df)
        self._means = {col: df[col].mean() for col in df.columns}
        self._std_devs = {col: df[col].std() for col in df.columns}

    def transform(self, df):
        df = nw.LazyFrame(df)
        df = df.with_columns(
            (nw.col(col) - self._means[col]) / self._std_devs[col]
            for col in df.columns
        )
        return nw.to_native(df)
```

Next, let's try running it. Notice how, as `transform` doesn't use
any eager-only features, so we can make it completely lazy!

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
