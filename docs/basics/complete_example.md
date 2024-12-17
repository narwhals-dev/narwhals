# Complete example

We're going to write a dataframe-agnostic "Standard Scaler". This class will have
`fit` and `transform` methods (like `scikit-learn` transformers), and will work
agnostically for pandas and Polars.

We'll need to write two methods:

- `fit`: find the mean and standard deviation for each column from a given training set;
- `transform`: scale a given dataset with the mean and standard deviations calculated
  during `fit`.

## Fit method

Unlike the `transform` method, which we'll write below, `fit` cannot stay lazy,
as we need to compute concrete values for the means and standard deviations.

To be able to get `Series` out of our `DataFrame`, we'll pass `eager_only=True` to `nw.from_native`.
This is because Polars doesn't have a concept of lazy `Series`, and so Narwhals
doesn't either.

We can specify that in the `@nw.narwhalify` decorator by setting `eager_only=True`, and
the argument will be propagated to `nw.from_native`.

=== "from/to_native"
    ```python
    from typing import Self
    import narwhals as nw
    from narwhals.typing import IntoDataFrameT


    class StandardScaler:
        def fit(self: Self, df: IntoDataFrameT) -> Self:
            df_nw = nw.from_native(df, eager_only=True)
            self._means = {col: df_nw[col].mean() for col in df_nw.columns}
            self._std_devs = {col: df_nw[col].std() for col in df_nw.columns}
            self._columns = df_nw.columns
            return self
    ```

=== "@narwhalify"
    ```python
    from typing import Self
    import narwhals as nw
    from narwhals.typing import DataFrameT


    class StandardScaler:
        @nw.narwhalify(eager_only=True)
        def fit(self: Self, df: DataFrameT) -> Self:
            self._means = {col: df[col].mean() for col in df.columns}
            self._std_devs = {col: df[col].std() for col in df.columns}
            self._columns = df.columns
            return self
    ```

## Transform method

We're going to take in a dataframe, and return a dataframe of the same type:

=== "from/to_native"
    ```python
    from typing import Self
    import narwhals as nw
    from narwhals.typing import IntoFrameT


    class StandardScaler:
        ...

        def transform(self: Self, df: IntoFrameT) -> IntoFrameT:
            df_nw = nw.from_native(df)
            return df_nw.with_columns(
                (nw.col(col) - self._means[col]) / self._std_devs[col]
                for col in self._columns
            ).to_native()
    ```

=== "@narwhalify"
    ```python
    from typing import Self
    import narwhals as nw
    from narwhals.typing import FrameT


    class StandardScaler:
        ...

        @nw.narwhalify
        def transform(self: Self, df: FrameT) -> FrameT:
            return df.with_columns(
                (nw.col(col) - self._means[col]) / self._std_devs[col]
                for col in self._columns
            )
    ```

Note that all the calculations here can stay lazy if the underlying library permits it,
so we don't pass in any extra keyword-arguments such as `eager_only`, we just use the
default `eager_only=False`.

## Putting it all together

Here is our dataframe-agnostic standard scaler:

=== "from/to_native"
    ```python
    from typing import Self
    import narwhals as nw
    from narwhals.typing import IntoDataFrameT
    from narwhals.typing import IntoFrameT


    class StandardScaler:
        def fit(self: Self, df: IntoDataFrameT) -> Self:
            df_nw = nw.from_native(df, eager_only=True)
            self._means = {col: df_nw[col].mean() for col in df_nw.columns}
            self._std_devs = {col: df_nw[col].std() for col in df_nw.columns}
            self._columns = df_nw.columns
            return self

        def transform(self: Self, df: IntoFrameT) -> IntoFrameT:
            df_nw = nw.from_native(df)
            return df_nw.with_columns(
                (nw.col(col) - self._means[col]) / self._std_devs[col]
                for col in self._columns
            ).to_native()
    ```

=== "@narwhalify"
    ```python exec="1" source="above" session="standard-scaler-example"
    from typing import Self
    import narwhals as nw
    from narwhals.typing import DataFrameT
    from narwhals.typing import FrameT


    class StandardScaler:
        @nw.narwhalify(eager_only=True)
        def fit(self: Self, df: DataFrameT) -> Self:
            self._means = {col: df[col].mean() for col in df.columns}
            self._std_devs = {col: df[col].std() for col in df.columns}
            self._columns = df.columns
            return self

        @nw.narwhalify
        def transform(self: Self, df: FrameT) -> FrameT:
            return df.with_columns(
                (nw.col(col) - self._means[col]) / self._std_devs[col]
                for col in self._columns
            )
    ```

Next, let's try running it. Notice how, as `transform` doesn't use
any eager-only features, so we can pass a Polars LazyFrame to it and have it
stay lazy!

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="standard-scaler-example"
    import pandas as pd

    df_train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})
    df_test = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})
    scaler = StandardScaler()
    scaler.fit(df_train)
    print(scaler.transform(df_test))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="standard-scaler-example"
    import polars as pl

    df_train = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})
    df_test = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 7]})
    scaler = StandardScaler()
    scaler.fit(df_train)
    print(scaler.transform(df_test).collect())
    ```
