---
search:
  exclude: true
---

# API Completeness

In the following section it is possible to check which method is implemented for which
class and backend.

!!! info

    - By design, Polars supports all the methods of the Narwhals API.
    - "pandas-like" means pandas, cuDF and Modin.
    - "spark-like" means PySpark (including Spark Connect) and SQLFrame.
    - Backends provided by [plugins](../extending.md) (such as Daft) are not covered by
      these tables.
