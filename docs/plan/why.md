# Why?

## Background
[This sentiment] has been a driving force behind this project:

[This sentiment]: https://github.com/narwhals-dev/narwhals/pull/2483#issuecomment-2866902903

> *What do you think about the ways other packages represent operations/expressions?*  
> ...  
> I feel like there are some lessons we could learn from how other models solve *similar* problems 🙂

1 year and many lines of code **read** later - I know we have more to learn from those that came before us.

Since ([#2483]) we've had another rewrite in ([#3152]) that updated the internals of [`narwhals.expr.Expr`][] [^1] and [`CompliantExpr`](https://github.com/narwhals-dev/narwhals/blob/967e6108416b532314d6f2b2b47bdc7b6442662c/src/narwhals/_compliant/expr.py) [^2].  
*But* these internals are still quite different to what you'll find reading the source of [related projects].

[^1]: See [how it works](./how_it_works.md#nodes)
[^2]: See [`_expression_parsing.py`](https://github.com/narwhals-dev/narwhals/blob/967e6108416b532314d6f2b2b47bdc7b6442662c/src/narwhals/_expression_parsing.py)

[#2483]: https://github.com/narwhals-dev/narwhals/pull/2483
[#3152]: https://github.com/narwhals-dev/narwhals/pull/3152
[related projects]: ./inspired.md

### Isn't being different what makes Narwhals popular?
There's no doubt that Narwhals is popular, this is indeed wild 🙌

[![Downloads](https://static.pepy.tech/badge/narwhals/month)](https://pepy.tech/project/narwhals)

[early adopter]: https://github.com/vega/altair/pull/3384#issuecomment-2188349276
[keep growing]: ../index.md

The reasons to choose Narwhals [keep growing], but as an [early adopter] I still believe these are key:

> 1. The author (@MarcoGorelli) is a maintainer for both `pandas` and `polars`
> 2. It has **zero dependencies**
> 3. Uses a [single API](https://narwhals-dev.github.io/narwhals/api-reference/), which could potentially simplify a lot of ... compatibility code

!!! info

    *What* Narwhals does for it's users and *how* we do it are two very different things. 
    If you're already convinced by the promise of Narwhals - the latter may never be a factor.


### When *what?* meets *how?*
But when they collide, it tends to be something like:

1. [Trying to understand how the code works, by reading it](https://discord.com/channels/1235257048170762310/1235257049626181656/1501887540813234176)
2. [Trying to implement a feature that doesn't fit into current model](https://github.com/narwhals-dev/narwhals/pull/3460#discussion_r2777195110)
3. Requesting a feature that doesn't fit into current model
    1. https://github.com/narwhals-dev/narwhals/issues/2704
    2. https://github.com/narwhals-dev/narwhals/issues/3310
    3. https://github.com/narwhals-dev/narwhals/issues/1610
    4. https://github.com/narwhals-dev/narwhals/issues/3000
    5. https://github.com/narwhals-dev/narwhals/issues/2869
    6. https://github.com/narwhals-dev/narwhals/issues/2722
    7. https://github.com/narwhals-dev/narwhals/discussions/3425
    8. https://github.com/narwhals-dev/narwhals/discussions/2816


## The problem(s)
### *"An expression is a function from a DataFrame to a sequence of Series."*

In [How it works - theory](./how_it_works.md#theory), this quote is hard to miss but how true is it?

We'll explore a few angles:

1. What are we talking about?
    1. [What is a Series?](#what-is-a-series)
    2. [A function from a ~~Data~~ Frame?](#a-function-from-a-data-frame)
2. [How does our definition compare to Polars?](#how-does-our-definition-compare-to-polars)
3. **TODO:** [How other projects in the ecosystem represent expressions?](#how-that-compares-to-the-ecosystem)

#### What is a Series?

What began as just [pl.Series] and [`pd.Series`][pandas.Series], has grown to mean many different things.  

Let's see how each backend compares to [pl.Series]:

| Backend       | Class [^1]                                 | Named? [^2]     | Order? [^3]                   | Row-separable? [^4]           | Columnar? [^5]                     |
| ------------- | ------------------------------------------ | --------------- | ----------------------------- | ----------------------------- | ---------------------------------- |
| Polars        | [pl.Series]                                | :lucide-check:  | :lucide-check:                | :lucide-check:                | :lucide-check:                     |
| Pandas(-Like) | [`pd.Series`][pandas.Series]               | :lucide-check:  | :lucide-check:                | :lucide-triangle-alert: [^6]  | :lucide-check:                     |
| PyArrow       | [`pa.ChunkedArray`][pyarrow.ChunkedArray]  | :lucide-x: [^7] | :lucide-check:                | :lucide-check:                | :lucide-check:                     |
| DuckDB        | [`duckdb.Expression`][]                    | :lucide-check:  | :lucide-x:                    | :lucide-x:                    | :lucide-check:                     |
| PySpark       | [`pyspark.sql.Column`][]                   | :lucide-check:  | :lucide-x:                    | :lucide-x:                    | :lucide-x:                         |
| Ibis          | [ibis.ir.Column] [^8]                      | :lucide-check:  | :lucide-x:                    | :lucide-x:                    | :lucide-circle-question-mark: [^9] |
| Dask          | [`dx.Series`][dask.dataframe.Series] [^10] | :lucide-check:  | :lucide-circle-question-mark: | :lucide-circle-question-mark: | :lucide-circle-question-mark:      |

Some backends are clearly more *Series-like* than others.  
If the only issue were the name, we could simply tweak the docs and be home in time for dinner:

> *An expression is a function from a DataFrame to a sequence of ~~Series~~ columns.*

*Sadly*, I've got more teeth to pull at - but at least we have some base definitions!

[pl.Series]: https://docs.pola.rs/api/python/stable/reference/series/index.html
[ibis.ir.Column]: https://ibis-project.org/reference/expression-generic#ibis.expr.types.generic.Column

[^1]: What class(es) do we use?
[^2]: Does it have an accessible name?
[^3]: Is order guaranteed?
[^4]: See [`FunctionFlags.ROW_SEPARABLE`][narwhals._plan._flags.FunctionFlags.ROW_SEPARABLE]
[^5]: See https://en.wikipedia.org/wiki/Data_orientation#Column-oriented
[^6]: When careful with [`pd.Index`][pandas.Index]
[^7]: Maps directly to the lower-level [polars `#!rust ChunkedArray`](https://docs.rs/polars/latest/polars/#chunkedarray), which is unnamed
[^8]: [`*Scalar`](https://ibis-project.org/concepts/datatypes) classes are also user-facing
[^9]: Backend-dependent
[^10]: Just a very fancy expression, only a "Series" in name

#### A function from a ~~Data~~ Frame?

I'll save you the blabbing about LazyFrame, that's too easy!

The interesting part is how much we need to do to squeeze every backend into this `Callable[[Frame], Sequence[Series]]` shape:

<!--TODO @dangotbanned: oh boy, docs are hard 

- Pandas - do homework
- PyArrow - Just focus on `pyarrow.compute`
- DuckDB, PySpark - Lean into how they compare to PyArrow
- Ibis - idk, this is correct but how useful is it here?

-->

| Backend         | Notes                                                                                                                                                                              |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pandas(-Like)   | You can apply functions to dataframes, but we don't use any of those APIs AFAICT.<br> There's also horizontal with `axis=1` (but same dealio).                                     |
| PyArrow         | Best to think of it as array-first + some table operations.<br> In terms of expressions, the majority of the API we use is `pyarrow.compute` (most of those are scalar functions). |
| DuckDB, PySpark | APIs where functions are applied to expressions and they have no inherent context until you put them in one.                                                                       |
| Ibis            | More table-bound, but every function results in 1 output.<br> Also supports selectors, but unlike Polars they are not first-class expressions.                                     |

I do not see a connection between what we do and how our backends work.  
They all support `Callable[[Column], Column]` and we use this for none of them.

##### What about scalars?
Consider these two queries:

``` py
import narwhals as nw

df = nw.DataFrame.from_dict({"a": [1, 2], "b": [3, 4]}, backend="pyarrow")
literal = nw.lit(5)

one = df.select(literal)
two = df.with_columns(literal)
```

In either `one` or `two`, is `literal` *a function from a DataFrame to a sequence of Series*?

I would argue that it has no relation to functions, frames, series (and definitely not a sequence of them).  
But don't take my word for it, let's look at how trying to fit `lit` into this box plays out.

[ArrowNamespace.lit]: https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_arrow/namespace.py#L67-L82

[ArrowNamespace.lit] takes a python scalar and:

``` py
class ArrowNamespace:
    def lit(self, value: PythonLiteral, dtype: IntoDType | None) -> ArrowExpr:
        def _lit_arrow_series(_: ArrowDataFrame) -> ArrowSeries: # (1)!
            arrow_series = ArrowSeries.from_iterable( # (2)!
                data=[value], name="literal", context=self # (3)!
            )
            if dtype:
                return arrow_series.cast(dtype) # (4)!
            return arrow_series

        return self._expr(
            lambda df: [_lit_arrow_series(df)], # (5)!
            evaluate_output_names=lambda _df: ["literal"], # (6)!
            alias_output_names=None,
            version=self._version,
        )
```

1.  Creates a function that returns the scalar as a length-1 `ArrowSeries` with the name `"literal"`
2.  `ArrowSeries.from_iterable` is non-trivial
3.  Uses a constant name, and we need to track this outside of [`pa.ChunkedArray`][pyarrow.ChunkedArray]
4.  Optionally [casting the series](https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_arrow/series.py#L549-L550), which creates another `ArrowSeries` that ["preserves broadcast"](https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_arrow/series.py#L148-L149)
5.  Creates another function to pass in a (unused) dataframe to the first function, returning a `list`
6.  Creates another function to pass in a (unused) dataframe, and returns `["literal"]`


Woah. That was ... almost entirely unrelated to both queries, wasn't it?  
Here's what stood out to me:

- PyArrow natively supports scalars ([`pa.Scalar`][pyarrow.Scalar]), but we don't use it
- Why are we creating 3 functions each time we want to wrap a scalar?
    - And why are we passing a dataframe where it is never used?
- Why are we providing the name `"literal"` here, and why was it twice?
- Why are we creating list(s) for a single output expression?

!!! question

    Perhaps this is an outlier? 
    Or maybe PyArrow is difficult to work with?

[PandasLikeNamespace.lit]: https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_pandas_like/namespace.py#L84-L132
[DaskNamespace.lit]: https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_dask/namespace.py#L62-L90
[IbisNamespace.lit]: https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_ibis/namespace.py#L127-L143
[DuckDBNamespace.lit]: https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_duckdb/namespace.py#L135-L159
[SparkLikeNamespace.lit]: https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_spark_like/namespace.py#L107-L153

- [PandasLikeNamespace.lit]
    - Gets more complex, but this time it *does use* the dataframe for (something related to?) the `index`
- [DaskNamespace.lit]
    - Slightly less complex than Pandas, *does use* the dataframe for `npartitions`
- [IbisNamespace.lit]
    - *Only* 2 functions? might be the least complex yet - maybe it is smooth sailing from here?
- [DuckDBNamespace.lit]
    - We now need the dataframe to create a "deferred time zone"
    - We create another function that does the same thing, but has a required, unused "window inputs"
- [SparkLikeNamespace.lit]
    - Like DuckDB, but with `Implementation` branching


`lit` *should be* close to the simplest expression to implement.  
However each backend requires *creating* at least two new functions per-call.  
    
1. We **know** that the result of this expression will always have the name `"literal"`.
2. We **know** that the result will always be a single column.
3. We **know** that the column will be considered scalar (in broadcasting terms).

But these details get repeated in various forms in every implementation.

!!! success "Solved"

    See [What's new? - First-class scalars](../whats-new/behind-the-scenes/#first-class-scalars) for how we can build our way out of this.


##### Summary
> *"An expression is a function from a DataFrame to a sequence of Series."*

As we've seen, this model is not native to any backend we support.

Is this because Polars is unique?

#### How does our definition compare to Polars?
[Expression expansion]: https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/#expression-expansion

Ahh, feels like home.

So the first thing you might say is

> um actually, it is because [Expression expansion], ever heard of it?

*Indeed*, polars has expression expansion - and yet this is how it defines expressions in the rust docs:

!!! quote "[Polars (rust) expressions](https://docs.rs/polars/latest/polars/#expressions)"

    Polars expressions ... are a functional mapping of `#!rust Fn(Series) -> Series` ...  


This is **not** a subtle difference, these guys are barely related

```py
def narwhals(frame: Frame) -> Sequence[Series]: ...
def polars(series: Series) -> Series: ...  # (1)!
```

1.  yes, just like all other backends support natively

??? info "A trip down memory lane"

    [^13]: `v1.1.0` and `v1.31.0` are meaningful events *to me*, but you could pick any version really 😅

    If we take a step back, what's interesting is that this concept has been in Narwhals since before it was *Narwhals*.  
    Here's 4 snapshots of what that signature describes [^13]:

    - (v0.1.0) [`polars_api_compat` is born](https://github.com/narwhals-dev/narwhals/blob/4add1be80a5a6d553e22a3375ea61695ab4f5843/polars_api_compat/utils.py#L180-L217)
    - (v1.1.0) [Altair adopts Narwhals](https://github.com/narwhals-dev/narwhals/blob/6e0c3adb95ddf5324b29a2c1453e5139e2936263/narwhals/_expression_parsing.py#L176-L224)
    - (v1.31.0) [`narwhals._compliant` joins the party](https://github.com/narwhals-dev/narwhals/blob/668bbf698e5dee67576b9a003144d03f15b65310/narwhals/_compliant/expr.py#L348-L431)
    - (v2.23.0) [Not much has changed here since](https://github.com/narwhals-dev/narwhals/blob/93652c3abedb9c0e519926397f157426e8caaf86/src/narwhals/_compliant/expr.py#L323-L393)

    I know the bar is high to change the world.

To put it bluntly, we *don't* have expression expansion.  
Because we don't have it - yet have APIs that look like we do - it means every expression method 
checks if it is "multi-output" and does many different dances around carrying unresolved selections everywhere.

This extra complexity is created by Narwhals, but it doesn't have to be this way since Polars shows us how fix it.

!!! success "Solved"

    - [What's new? - More selectors, allow them in more places](../whats-new/on-the-surface/#more-selectors-allow-them-in-more-places)
    - [What's new? - A rich representation for expressions](../whats-new/behind-the-scenes/#a-rich-representation-for-expressions)


#### How that compares to the ecosystem

## Can this really not be solved with more rewrites?

## Wrapping up with something cheerful
<!--TODO @dangotbanned: Plan section -->
