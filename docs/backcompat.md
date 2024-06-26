# Perfect backwards compatibility

Ever upgraded a package, only to find that it breaks all your tests because of an intentional
API change? Did you end up having to litter your code with statements such as

```python
if parse_version(pdx.__version__) < parse_version('1.3.0'):
    df = df.brewbeer()
elif parse_version('1.3.0') <= parse_version(pdx.__version__) < parse_version('1.5.0'):
    df = df.brew_beer()
else:
    df = df.brew_drink('beer')
```
?

Now imagine multiplying that complexity over all the dataframe libraries you want to support...

Narwhals offers a simple solution, inspired by Rust.

## Narwhals' Stable API

Narwhals implements a subset of the Polars API. What will Narwhals do if/when Polars makes
a backwards-incompatible change? Would you need to update your Narwhals code?

To understand the solution, let's go through an example. Suppose that, hypothetically, in Polars 2.0,
`polars.Expr.cum_sum` was to be renamed to `polars.Expr.cumulative_sum`. In Narwhals, we
have `narwhals.Expr.cum_sum`. Does this mean that Narwhals will also rename its method,
and deprecate the old one? The answer is...no!

Narwhals offers a `StableAPI` object, which allows you to write your code once and forget about
it. That is to say, if you write your code like this:

```python
from narwhals import StableAPI

nw = StableAPI('1.0')

@narwhalify
    def func(df):
        return df.with_columns(nw.col('a').cum_sum())
```

then we, in Narwhals, promise that your code will keep working, even in newer versions of Polars
after they have renamed their method, so long as you used `StableAPI('1.0')`. If you use
`StableAPI('2.0')`, then the method would be called `cumulative_sum` instead.

## `import narwhals as nw` or `nw = StableAPI(api_version)`?

Which should you use? In general we recommend:

- when prototyping, use `import narwhals as nw`, so you iterate quickly
- once you're happy with what you've got and what to release something production-ready and stable,
  when switch out your `import narwhals as nw` usage for `nw = StableAPI(api_version)`. You may
  want to instantiate this once in your project.
