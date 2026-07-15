# What is `narwhals._plan`?

`narwhals._plan` explores what Narwhals can be if we bite off a little more of Polars.

??? info

    This project ([#2571]) has been running on/off for ~1 year, prompted by an [off-hand PR review comment].

    **You should expect rough edges and feature gaps.**
    
    But if you were curious enough to unfold me - then you might just the person to help keep things moving 👼.

    Check out [Mind the gap] and [Related issues] to get involved

    [#2571]: https://github.com/narwhals-dev/narwhals/issues/2571
    [off-hand PR review comment]: https://github.com/narwhals-dev/narwhals/pull/2483#issuecomment-2866902903
    [Mind the gap]: ./mind-the-gap.md
    [Related issues]: ./related-issues.md#related-issues

## On the surface
- Extended/new APIs
- Improved error messages
- Increased API completeness for PyArrow
- Closer behavior alignment to Polars

## Behind the scenes
- A rich representation for expressions
- Fine-grained protocols, which compose to provide useful defaults
- First-class scalars
- Every backend is a plugin
