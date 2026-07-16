# Sparse columns

pandas can store a column as a [`SparseArray`](https://pandas.pydata.org/docs/user_guide/sparse.html),
which compresses repeated occurrences of a single *fill value* (typically `0` or `NaN`) to save memory.

No other Narwhals backend has an equivalent concept.

Narwhals treats sparsity as a **storage detail, not a logical dtype**: a sparse column has the same Narwhals dtype as
its dense equivalent. Beyond that, keeping columns sparse is a **best-effort, pandas-only convenience**.

## Reading the schema

A `Sparse[<subtype>]` column reports the Narwhals dtype of its dense `<subtype>` (e.g. `Sparse[int64, 0]` maps to `Int64`),
rather than `Unknown`:

```python exec="yes" source="above" session="sparse"
import narwhals as nw
import pandas as pd

frame = pd.DataFrame(
    {
        "a": pd.arrays.SparseArray([0, 1, 0, 2]),
        "b": pd.arrays.SparseArray([0.0, 1.5, 0.0, 2.5]),
    }
)
```

```python exec="yes" source="material-block" result="python" session="sparse"
print("native dtypes: ", frame.dtypes.to_dict())
print("narwhals schema:", nw.from_native(frame).schema)
```

## Casting

`cast` keeps a column sparse when it can do so *without altering the data*, and densifies otherwise.
A sparse column stays sparse when the target is a numeric or temporal NumPy dtype and the source fill
value is representable in it; it densifies when:

* the target is not a NumPy dtype (nullable, PyArrow-backed) as pandas' `SparseDtype` can only wrap NumPy dtypes;
* the target is a string / object dtype: casting a sparse column to these is a no-op that would leave the stored values unconverted;
* the source fill value cannot be represented in the target (e.g. a `NaN` fill cast to an integer subtype), since no
    fill would preserve the compressed entries.

```python exec="yes" source="material-block" result="python" session="sparse"
s = nw.from_native(frame["a"], series_only=True)

print("cast to Float64:", nw.to_native(s.cast(nw.Float64)).dtype)
print("cast to String:", nw.to_native(s.cast(nw.String)).dtype)
```

!!! note
    Sparse *input* is accepted and preserved on a best-effort basis, for pandas only.
    
    There is no sparse output on any other backend, and Narwhals only explicitly preserves sparsity in `cast`.
    Other operations delegate to pandas, which may or may not keep the column sparse.
