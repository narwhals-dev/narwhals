# Thread safety

## TL;DR

- **Narwhals objects are safe to share between threads for reading**: methods return new
  objects; nothing mutates the receiver, nor the native object passed to
  [`from_native`](../api-reference/narwhals.md#narwhals.from_native).
- **Writing is the backend's business**: Narwhals has no in-place mutation API, but once you
  reach the native object (`to_native`, `to_numpy`, ...) the backend's rules apply.
- **DuckDB needs one connection per thread**: the only backend with a rule that changes how
  you write Narwhals code, see [DuckDB](#duckdb).

!!! tip
    All of this holds on both the GIL-enabled and the [free-threaded] build. A race is a bug
    either way; free-threading just makes it likely enough to notice.

## Global state inside Narwhals

| State | Thread-safe? | Notes |
| ----- | ------------ | ----- |
| Function caches <br/>(`backend_version`, dtype conversions, plugin discovery, ...) | Yes | [`lru_cache` is threadsafe][lru_cache] and the cached functions are pure,<br/>so a concurrent cold cache costs at most a duplicated computation |
| Constant lookup tables <br/>(dtype mappings, interval units, ...) | Yes | Read-only after import |
| Configuration / registries / mutable singletons | N/A | Narwhals has none |
| `narwhals.sql`'s DuckDB catalog | Yes, per thread | See [`narwhals.sql`](#narwhalssql) |

### Backend detection

Narwhals does not import your backend in order to recognise its objects: you must already
have imported it to have an object to pass, so it looks the module up in `sys.modules`
instead. It *does* import on your behalf when you name a backend
(`nw.from_dict(..., backend="pandas")`) or convert between them (`to_polars`, `to_arrow`,
...). Since a module is [visible in `sys.modules` before its body finishes executing][loading],
import your backends on the main thread rather than racing that first import from a pool.

## Sharing Narwhals objects

All of the following are safe to share between threads:

| Object | Why |
| --- | --- |
| `DataFrame`, `LazyFrame`, `Series` | Every method returns a new object.<br/>Lazily-cached metadata (schema, column names) is computed idempotently,<br/>so a concurrent first access only repeats work |
| `Expr` | A chain of immutable nodes. `Expr.over` copies before it rewrites,<br/>so reusing one expression in several queries at once cannot corrupt it |
| `GroupBy` (the result of `DataFrame.group_by(...)`) | Aggregation state is passed along the call, not stashed on the object |
| `Schema` | A `dict` subclass: safe to read concurrently, but do not mutate a shared instance |

!!! warning
    Sharing is safe because nobody writes: if a worker hands the native object to something
    that mutates it in place, Narwhals' view goes stale too - `from_native` does not copy.

## Backends

| Backend | Reads | Writes to the native object |
| --- | --- | --- |
| Polars | Safe | Operations return new objects, though immutability is [not enforced][polars-immutable] |
| PyArrow | Safe | [Arrow data is immutable][arrow-immutable] |
| pandas (NumPy-backed) | Safe | **Not safe** through the NumPy buffer (`.values`, `.to_numpy()`):<br/>[mutating a shared array races][numpy-thread-safety]. pandas' own writes go<br/>through [copy-on-write], the default since pandas 3.0 |
| pandas (Arrow-backed) | Safe | [copy-on-write]: a write produces new buffers rather than mutating shared ones |
| DuckDB | One connection per thread, see below | N/A |
| PySpark, SQLFrame, Ibis, Dask | Delegated to the session / scheduler | Consult the backend's own docs |

## DuckDB

Every thread needs [its own `.cursor()`, a thread-local connection to the same
database][multiple threads]:

```python
import duckdb
import narwhals as nw


def worker(con: duckdb.DuckDBPyConnection) -> None:
    cursor = con.cursor()  # one per thread
    lf = nw.from_native(cursor.sql("select * from my_table"))
    print(lf.filter(nw.col("a") > 1).collect())
```

1. **A cursor does not inherit `LOCAL`-[scoped][duckdb-config] settings**: `TimeZone` is
   one of them, so use `SET GLOBAL`, or set it on every cursor. It matters in Narwhals
   because DuckDB keeps the time zone in the connection rather than the dtype:
   `collect_schema()` reports a time-zone-aware column using *the cursor's* time zone.
2. **A relation belongs to the connection that created it**: combining relations from two
   connections is [rejected outright][duckdb-join-relation], so build frames you intend to
   `join` or `concat` in the thread that uses them.
3. **`collect_schema()` on frames from a shared connection is safe, executing on one is
   not**: a time-zone-aware dtype is the one case where Narwhals queries your connection
   *implicitly*, and it serializes that query. Explicit execution (`collect`, `to_arrow`,
   ...) runs on the connection you gave it and follows DuckDB's rules.

[`scan_csv`](../api-reference/narwhals.md#narwhals.scan_csv) and
[`scan_parquet`](../api-reference/narwhals.md#narwhals.scan_parquet) read through DuckDB's
process-global default connection unless you pass `connection=con.cursor()`.

### `narwhals.sql`

[`narwhals.sql.table`](../api-reference/sql.md) keeps a module-level DuckDB connection and
gives each thread its own cursor on it. Since a cursor is a connection to the same database,
the catalog is shared process-wide and `name` must be unique across threads; and by (2)
above, tables created in *different* threads cannot be joined.

## Free-threaded Python

Narwhals is pure Python with no compiled extensions, so importing it never re-enables the
GIL. Whether *your* stack supports the [free-threaded] build depends on the backend: check
that its wheels are built for `cp314t` (or whichever version you run).

CI covers it in two ways: the whole suite runs again under [pytest-run-parallel] with
`--parallel-threads=4`, which surfaces races through global state, and
`tests/free_threading_test.py` stresses every guarantee on this page. If you find a race,
please [open an issue](https://github.com/narwhals-dev/narwhals/issues) with a script that
reproduces it, ideally on a free-threaded build or with `sys.setswitchinterval(1e-7)`.

[arrow-immutable]: https://arrow.apache.org/docs/python/data.html
[copy-on-write]: https://pandas.pydata.org/docs/user_guide/copy_on_write.html
[duckdb-config]: https://duckdb.org/docs/stable/configuration/overview
[duckdb-join-relation]: https://github.com/duckdb/duckdb/blob/fabf1d60bb0565032ad7d48e64f689fdbf616719/src/main/relation/join_relation.cpp#L25-L26
[free-threaded]: https://docs.python.org/3/howto/free-threading-python.html
[loading]: https://docs.python.org/3/reference/import.html#loading
[lru_cache]: https://docs.python.org/3/library/functools.html#functools.lru_cache
[multiple threads]: https://duckdb.org/docs/stable/guides/python/multiple_threads
[numpy-thread-safety]: https://numpy.org/doc/stable/reference/thread_safety.html
[polars-immutable]: https://github.com/pola-rs/polars/issues/17447
[pytest-run-parallel]: https://github.com/Quansight-Labs/pytest-run-parallel
