# pizzlies

A pure-Python, zero-dependency, eager dataframe library whose native frame is a plain
`dict[str, list]`, with the [Narwhals](https://github.com/narwhals-dev/narwhals) API.

```console
pip install pizzlies
```

## Standalone

`pizzlies` re-exports Narwhals' API, with its constructors bound to this backend:

```python
import pizzlies as pz

df = pz.from_dict({"a": [1, 2, 3], "b": [4.0, None, 6.0]})
df.with_columns(c=pz.col("a") * 2).select(pz.col("c").sum(), pz.col("b").mean())
df.to_native()  # {'a': [1, 2, 3], 'b': [4.0, None, 6.0]}
```

`pz.from_dict`, `pz.from_dicts`, `pz.from_numpy`, `pz.from_arrow`, `pz.new_series`,
`pz.read_csv` and `pz.scan_csv` create `pizzlies`-backed objects. Everything else (`pz.col`,
`pz.when`, `pz.concat`, `pz.selectors`, the dtypes, ...) is Narwhals' own, so see the
[Narwhals docs](https://narwhals-dev.github.io/narwhals/) for the full API.

## As a Narwhals plugin

Narwhals discovers the plugin automatically, so a dict of lists can be passed straight to
`nw.from_native`, and `backend="pizzlies"` (or `backend=pz.BACKEND`) works wherever Narwhals
accepts a `backend`:

```python
import narwhals as nw

df = nw.from_native({"a": [1, 2, 3]}, eager_only=True)
df.select(nw.col("a").sum()).to_native()  # {'a': [6]}
```

Note that, once installed, `pizzlies` claims every plain `dict` passed to `nw.from_native`.

## Size and import time

`pizzlies` is small and quick to import, but it's pure Python, so it's much slower than
pandas or Polars on anything but small data.

| | Installed size | Import time |
| --- | ---: | ---: |
| `pizzlies` (+ `narwhals`) | 0.3 MB (+ 2.1 MB) | 40 ms |
| `polars` (+ `polars-runtime-32`) | 183 MB | 125 ms |
| `pyarrow` | 171 MB | 144 ms |
| `pandas` (+ `numpy`) | 96 MB | 365 ms |

The `pizzlies` wheel is 74 KB. Import times are the median of 40 fresh interpreters, from
wheel installs; almost all of `pizzlies`' is importing Narwhals (34 ms on its own).

Measured with Python 3.13 on Linux x86-64 (AMD Ryzen AI 9 465), with pandas 3.1, Polars
2.1 and PyArrow 26. Results depend on your hardware, so measure your own workload.
`benchmark.py` reproduces the import times, see its docstring.

## Limitations

- **Parquet:** `pz.read_parquet` uses a small pure-Python reader. It handles flat and
  nested (list, struct and map) columns, the `PLAIN`, dictionary, `RLE`,
  `BYTE_STREAM_SPLIT` and `DELTA_*` encodings, and uncompressed, snappy, gzip, LZ4 and
  zstd compression. Zstd, Polars' default, needs Python 3.14+ or
  `pip install 'pizzlies[zstd]'`. Other files (e.g. brotli-compressed ones) raise
  `NotImplementedError`: read them with pyarrow and convert them,
  `pz.from_arrow(pq.read_table(path))`. Corrupt data raises `ValueError`, and page
  checksums are verified when the writer stored them, though only for the columns read.
- **Dtypes** are inferred from the Python values, so e.g. integers are always `Int64`,
  and an empty or all-null column's dtype is `Unknown` (so dtype selectors don't select
  it). `Map` dtypes aren't supported, so maps are read as lists of
  `{"key": ..., "value": ...}` structs, as Polars does.
- **NaN:** comparisons follow IEEE 754, as in Python: e.g. `NaN == NaN` and `NaN > 1.0`
  are false, where Polars treats NaN as equal to itself and greater than any other
  value. Sorting, `unique`, `group_by` and joins do treat NaN as Polars does.
- **No columns, no rows:** a frame is a dict of lists, so one with no columns has no
  rows either, e.g. `df.drop(*df.columns).with_columns(a=pz.lit(1))` is empty.
- **Time zones:** like Python, and unlike Polars, `pizzlies` ignores `fold` when comparing
  time-zone aware datetimes, so the two occurrences of an ambiguous wall time (e.g. 01:30
  when clocks go back) are equal in group-bys, joins and `unique`.

## Development

The public API lives in `pizzlies/__init__.py`, and the Narwhals-compliant implementation in
the private `pizzlies._plugin`. It only imports from Narwhals' public API
(`narwhals.compliant`, `narwhals.typing`, `narwhals.utils`, ...): anything it needs from
Narwhals' private modules is vendored in `pizzlies/_plugin/_vendored.py`. The exception is
expression metadata (`Expr._metadata`), which the plugin reads directly and which has no
stability guarantees, so the plugin needs to be kept in step with the Narwhals versions it
supports.

The package has its own environment and lock file, and is not part of the root uv workspace:
once installed, it claims every plain `dict`, which would clash with the in-repo
`test-plugin`.

`run_tests.py` runs, in this package's `.venv`, both this package's own tests and Narwhals'
test suite against the plugin. Narwhals tests which are known to fail (mostly
not-yet-implemented methods) are listed in `known_failures.txt` and deselected:

```console
python packages/pizzlies/run_tests.py           # excludes known failures
python packages/pizzlies/run_tests.py --all     # includes known failures
python packages/pizzlies/run_tests.py --update  # regenerate known_failures.txt
```

`--update` exits non-zero if any test was added to the baseline, so new failures can't be
absorbed into it unnoticed.
