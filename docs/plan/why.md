# Why?

## Background
[This sentiment] has been a driving force behind this project:

[This sentiment]: https://github.com/narwhals-dev/narwhals/pull/2483#issuecomment-2866902903

> *What do you think about the ways other packages represent operations/expressions?*  
> ...  
> I feel like there are some lessons we could learn from how other models solve *similar* problems 🙂

1 year and many lines of code **read** later - I know we have more to learn from those that came before us.

Despite *another rewrite* that followed ([#2483]) in ([#3152]) - the internals of [`narwhals.expr.Expr`][] [^1] and [`CompliantExpr`] [^2] 
are unlike what you'll find reading the source of [related projects].

[^1]: See [how it works](./how_it_works.md#nodes)
[^2]: See [`_expression_parsing.py`](https://github.com/narwhals-dev/narwhals/blob/967e6108416b532314d6f2b2b47bdc7b6442662c/src/narwhals/_expression_parsing.py)

[#2483]: https://github.com/narwhals-dev/narwhals/pull/2483
[#3152]: https://github.com/narwhals-dev/narwhals/pull/3152
[related projects]: ./inspired.md
[`CompliantExpr`]: https://github.com/narwhals-dev/narwhals/blob/967e6108416b532314d6f2b2b47bdc7b6442662c/src/narwhals/_compliant/expr.py

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
<!--TODO @dangotbanned: Plan section -->

## Can this really not be solved with more rewrites?
<!--TODO @dangotbanned: Plan section 

- Timeline of marco's "change the world" rewrites 😉
  - when
  - changes: surface-level
  - changes: internals
  - changes: cumulative
-->


## Wrapping up with something cheerful
<!--TODO @dangotbanned: Plan section -->
