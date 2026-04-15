# `narwhals.testing`

## Asserts

::: narwhals.testing
    handler: python
    options:
      members:
        - assert_frame_equal
        - assert_series_equal

## `pytest` plugin

Narwhals register a pytest plugin that exposes parametrized fixtures with callables
to build native frames from a column-oriented python `dict`.

### Available fixtures

| Fixture | Backends |
|---|---|
| `constructor` | every selected backend (eager + lazy) |
| `constructor_eager` | only eager backends |

The selection is controlled by two CLI options:

* `--constructors=pandas,polars[lazy],duckdb`: comma-separated list.
    Defaults to [`DEFAULT_CONSTRUCTORS`][narwhals.testing.constructors.DEFAULT_CONSTRUCTORS]
    intersected with the backends installed in the current environment.
* `--all-cpu-constructors`: shortcut for "every CPU backend that is installed".
* `--use-external-constructor`: Skip narwhals.testing's parametrisation and let
    another plugin provide the `constructor*` fixtures.

Set the `NARWHALS_DEFAULT_CONSTRUCTORS` environment variable to override the default
list (useful e.g. when running under `cudf.pandas`).

### Quick start

The plugin auto-loads as soon as you `pip install narwhals`. Just write a test:

```python
from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.testing.typing import ConstructorEager, Data


def test_shape(constructor_eager: ConstructorEager) -> None:
    data: Data = {"x": [1, 2, 3]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    assert df.shape == (3, 1)
```

The fixtures are parametrised against every supported backend that is installed
in the current environment. Filter the matrix on the command line:

```bash
pytest --constructors="pandas,polars[lazy]"
pytest --all-cpu-constructors
```

### Type aliases

::: narwhals.testing.typing
    handler: python
    options:
      members:
        - Constructor
        - ConstructorEager
        - ConstructorLazy
        - Data
